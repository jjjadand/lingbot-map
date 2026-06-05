"""LingBot-MAP Real-Time 3D Reconstruction — demo_realtimeV2.py

Merges the best rendering (depth unprojection from demo_realtimeV) with
advanced features from demo_realtime.py (loop detection, sky mask, video input,
keyframe gating, MP4 export).

Core rendering (same as demo_realtimeV / official demo.py):
  1. depth unprojection: depth → camera coords → world coords
     (via depth_to_cam_coords_points + closed_form_inverse_se3)
  2. colors: original camera frame (pixel-aligned RGB)
  3. camera poses: c2w from pose_enc (extrinsics w2c → invert → c2w)
  4. confidence: depth_conf, absolute threshold (default 1.5, like official PointCloudViewer)

Additional features:
  - Loop detection: auto-detect return-to-start and stop
  - Sky mask: filter out sky pixels using depth threshold
  - Video file input: run on a recorded video file instead of live camera
  - Keyframe gating: only show frames with significant camera motion
  - MP4 export: rendered flythrough with Open3D (requires open3d)

Usage:
    python demo_realtimeV2.py --model_path lingbot-map.pt --video_device /dev/video0

    # With loop detection
    python demo_realtimeV2.py --model_path lingbot-map.pt --video_device /dev/video0 \
        --enable_loop_detection --loop_threshold 0.5

    # On video file
    python demo_realtimeV2.py --model_path lingbot-map.pt --video_file /path/to/video.mp4

    # Jetson with GStreamer
    python demo_realtimeV2.py --model_path lingbot-map.pt --video_device /dev/video0 \
        --use_gstreamer --image_width 640 --image_height 384 --fps 20

Stop recording:
    1. Ctrl+C (manual stop)
    2. GUI: click "Stop & Export" in the viewer
    3. Auto-stop: --max_frames (auto-stop after N frames)
    4. Loop detection: auto-detect loop closure and stop
       (--loop_threshold distance, --loop_skip_frames N)
"""

import argparse
import glob
import math
import os
import socket
import sys
import threading
import time
import signal

if "--compile" not in sys.argv:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
_repo_root = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", os.path.join(_repo_root, ".cache"))
os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(_repo_root, ".cache", "torch_extensions"))

import cv2
import numpy as np
import torch
from PIL import Image
import trimesh
import gc

from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import (
    closed_form_inverse_se3,
    closed_form_inverse_se3_general,
    depth_to_cam_coords_points,
)
from lingbot_map.vis.glb_export import integrate_camera_into_scene, get_opengl_conversion_matrix
from scipy.spatial.transform import Rotation as SRT


# =============================================================================
# Image Preprocessing (exact demo.py crop mode)
# =============================================================================

def preprocess_frame(frame_rgb, image_size, patch_size=14):
    """PIL BICUBIC resize → center-crop → ToTensor(), matching demo.py crop mode."""
    from torchvision import transforms as TF

    img = Image.fromarray(frame_rgb)
    w_orig, h_orig = img.size

    new_w = image_size
    new_h = round(h_orig * (new_w / w_orig) / patch_size) * patch_size

    img_r = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
    img_t = TF.ToTensor()(img_r)

    if new_h > image_size:
        start_y = (new_h - image_size) // 2
        img_t = img_t[:, start_y: start_y + image_size]

    return img_t


# =============================================================================
# Camera Capture
# =============================================================================

def _detect_jetson():
    try:
        with open("/proc/device-tree/model") as f:
            return "nvidia" in f.read().lower() or "jetson" in f.read().lower()
    except Exception:
        return False


class UVCCapture:
    def __init__(self, device="/dev/video0", width=640, height=384, fps=20,
                 pixel_format="MJPG", use_gstreamer=False, capture_fps=None, verbose=True):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.pixel_format = pixel_format
        self.use_gstreamer = use_gstreamer
        self.capture_fps = capture_fps or fps
        self.verbose = verbose
        self._cap = None
        self.is_jetson = _detect_jetson()
        self._interval = 1.0 / self.capture_fps
        self._last = 0.0
        self._open()

    def _open(self):
        opened = False
        if self.use_gstreamer and self.is_jetson:
            pipeline = (
                f"v4l2src device={self.device} ! "
                f"image/jpeg,framerate={self.capture_fps}/1 ! "
                f"jpegdec ! videoconvert ! "
                f"video/x-raw,format=RGB ! "
                f"appsink emit-signals=true drop=true sync=false"
            )
            self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if self._cap.isOpened():
                if self.verbose:
                    print("[Camera] GStreamer opened")
                opened = True

        if not opened:
            self._cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open {self.device}")
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if self.pixel_format == "MJPG":
                self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            elif self.pixel_format == "YUYV":
                self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
            self._cap.set(cv2.CAP_PROP_FPS, self.capture_fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if self.verbose:
                w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                af = self._cap.get(cv2.CAP_PROP_FPS)
                print(f"[Camera] V4L2: {w}x{h} @ {af:.0f}fps")

    def read_throttled(self):
        now = time.time()
        if now - self._last < self._interval:
            return False, None
        ret, frame = self._cap.read()
        if not ret:
            return False, None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._last = now
        return True, frame_rgb

    def release(self):
        if self._cap:
            self._cap.release()


class VideoFileCapture:
    """Read frames from a video file instead of a camera device."""

    def __init__(self, video_path, loop=False, verbose=True):
        self.video_path = video_path
        self.loop = loop
        self.verbose = verbose
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.verbose:
            print(f"[VideoFile] {video_path}: {self.width}x{self.height} @ {self.fps:.1f}fps, {self.total_frames} frames")
        self._released = False

    def read_throttled(self):
        if self._released:
            return False, None
        ret, frame = self._cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return True, frame_rgb
        if self.loop:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return True, frame_rgb
        return False, None

    def release(self):
        self._released = True
        if self._cap:
            self._cap.release()


# =============================================================================
# Loop Detection
# =============================================================================

class LoopDetector:
    """Detect loop closures based on camera position proximity.

    When the camera returns near a previously visited position (within
    ``threshold`` distance), a loop closure is detected.
    """

    def __init__(self, threshold=0.5, min_interval=30, min_history=30, window_size=100):
        self.threshold = threshold
        self.min_interval = min_interval
        self.min_history = min_history
        self.frames = []
        self.recent_window = []
        self.frame_idx = 0

    def update(self, c2w):
        if self.frame_idx < self.min_history:
            self.frame_idx += 1
            self.frames.append((self.frame_idx, c2w))
            self.recent_window.append((self.frame_idx, c2w))
            if len(self.recent_window) > self.min_interval:
                self.recent_window.pop(0)
            return False

        pos = c2w[:3, 3]
        self.frames.append((self.frame_idx, c2w))
        self.recent_window.append((self.frame_idx, c2w))
        if len(self.recent_window) > self.min_interval:
            self.recent_window.pop(0)

        loop_detected = False
        for hist_idx, (fidx, old_c2w) in enumerate(self.frames[:-1]):
            if self.frame_idx - fidx < self.min_interval:
                continue
            old_pos = old_c2w[:3, 3]
            dist = float(np.linalg.norm(pos - old_pos))
            if dist < self.threshold:
                print(f"[LoopDetect] Loop closure at frame {self.frame_idx} "
                      f"matches frame {fidx} (dist={dist:.3f})")
                loop_detected = True
                break

        self.frame_idx += 1
        return loop_detected

    def reset(self):
        self.frames = []
        self.recent_window = []
        self.frame_idx = 0


# =============================================================================
# Depth-based 3D unprojection (official demo method)
# =============================================================================

def unproject_depth_frame(depth_np, extrinsic_np, intrinsic_np, sky_threshold=50.0):
    """Unproject a single depth frame to world coordinates.

    Uses the EXACT same math as official PointCloudViewer:
      1. depth → camera coords (via intrinsic)
      2. camera coords → world coords (via c2w = inverse(extrinsic))

    Args:
        depth_np: [H, W] float32 depth in meters
        extrinsic_np: [3, 4] camera extrinsic (w2c, OpenCV convention)
        intrinsic_np: [3, 3] camera intrinsic matrix
        sky_threshold: depth values above this are masked as sky (default 50.0m)

    Returns:
        world_points: [H, W, 3] world coordinates
        cam_coords: [H, W, 3] camera coordinates (for debugging)
    """
    depth_masked = depth_np.copy()
    if sky_threshold > 0:
        depth_masked = np.where(depth_np > sky_threshold, 0.0, depth_np)

    cam_coords = depth_to_cam_coords_points(depth_masked, intrinsic_np)
    c2w = closed_form_inverse_se3(extrinsic_np[None])[0]

    R = c2w[:3, :3]
    t = c2w[:3, 3]
    world_points = cam_coords @ R.T + t

    return world_points, cam_coords


# =============================================================================
# Keyframe gating helpers
# =============================================================================

def _rotation_angle_deg(c2w_a, c2w_b):
    if c2w_a is None or c2w_b is None:
        return 180.0
    r_rel = c2w_a[:3, :3].T @ c2w_b[:3, :3]
    trace = float(np.trace(r_rel))
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _should_accept_keyframe(last_c2w, curr_c2w,
                             translation_thresh=0.0025, rotation_thresh_deg=6.0):
    def _ensure_4x4(m):
        if m is None:
            return None
        m = np.asarray(m)
        if m.shape == (4, 4):
            return m
        if m.shape == (3, 4):
            m2 = np.eye(4, dtype=m.dtype)
            m2[:3] = m
            return m2
        if m.ndim == 3 and m.shape[-2:] == (4, 4):
            return m[-1]
        return None

    last_c2w = _ensure_4x4(last_c2w)
    curr_c2w = _ensure_4x4(curr_c2w)

    if curr_c2w is None:
        return False, 0.0, 0.0
    if last_c2w is None:
        return True, 0.0, 180.0

    t_delta = float(np.linalg.norm(curr_c2w[:3, 3] - last_c2w[:3, 3]))
    r_delta = _rotation_angle_deg(last_c2w, curr_c2w)
    accept = (t_delta >= translation_thresh) or (r_delta >= rotation_thresh_deg)
    return accept, t_delta, r_delta


# =============================================================================
# Export (matches glb_export.py approach + MP4)
# =============================================================================

def export_reconstruction(frames, out_dir, conf_pct=50.0, ds_factor=10,
                         glb=True, npz_out=True, export_video=False):
    """Export accumulated frames to GLB and/or NPZ and/or MP4.

    Points are centered on the first camera position (like glb_export.apply_scene_alignment).
    """
    os.makedirs(out_dir, exist_ok=True)
    result = {}

    if not frames:
        print("[Export] No frames")
        return result

    print(f"[Export] {len(frames)} frames...")

    pts_list, cols_list, conf_list = [], [], []
    c2w_list = []
    images_out = []

    for pc_data in frames:
        if isinstance(pc_data, dict):
            pc = pc_data.get("pc")
            color = pc_data.get("color")
            conf = pc_data.get("conf")
            c2w = pc_data.get("c2w")
            image = pc_data.get("image")
        elif isinstance(pc_data, (tuple, list)) and len(pc_data) >= 4:
            pc, color, conf, c2w = pc_data[0], pc_data[1], pc_data[2], pc_data[3]
            image = pc_data[4] if len(pc_data) > 4 else None
        else:
            continue

        if pc is None or len(pc) == 0:
            continue

        pts_arr = np.asarray(pc).reshape(-1, 3)
        cols_arr = (np.asarray(color).reshape(-1, 3) if color is not None and len(color) > 0
                    else np.zeros((len(pts_arr), 3), dtype=np.uint8))
        conf_arr = np.asarray(conf).flatten() if conf is not None and len(conf) > 0 \
                   else np.ones(len(pts_arr), dtype=np.float32)

        if conf_pct <= 100:
            tval = np.percentile(conf_arr, conf_pct)
            mask = (conf_arr >= tval) & (conf_arr > 1e-5)
        else:
            mask = (conf_arr >= conf_pct) & (conf_arr > 1e-5)

        pts = pts_arr[mask]
        cols = cols_arr[mask]
        if len(pts) == 0:
            continue

        if ds_factor > 1:
            idx = np.arange(0, len(pts), ds_factor)
            pts, cols = pts[idx], cols[idx]

        pts_list.append(pts)
        cols_list.append(cols)
        c2w_list.append(c2w)
        if image is not None:
            images_out.append(image)

    if not pts_list:
        print("[Export] No valid points")
        return result

    all_pts = np.concatenate(pts_list)
    try:
        all_cols = np.concatenate(cols_list)
    except ValueError:
        all_cols = np.zeros((len(all_pts), 3), dtype=np.uint8)

    c2w_first = c2w_list[0]
    if c2w_first is not None:
        R0, t0 = c2w_first[:3, :3], c2w_first[:3, 3]
        R0_inv = R0.T
        t0_inv = -R0_inv @ t0
        aligned_pts = all_pts @ R0_inv.T + t0_inv
    else:
        center = all_pts.mean(axis=0)
        aligned_pts = all_pts - center

    if len(aligned_pts) > 100:
        lo = np.percentile(aligned_pts, 5, axis=0)
        hi = np.percentile(aligned_pts, 95, axis=0)
        scene_scale = max(np.linalg.norm(hi - lo), 0.1)
    else:
        scene_scale = 1.0

    if all_cols.dtype != np.uint8:
        if all_cols.max() <= 1.0:
            all_cols = (all_cols * 255).astype(np.uint8)
        else:
            all_cols = all_cols.astype(np.uint8)

    if glb:
        print("[Export] Building GLB...")
        scene = trimesh.Scene()
        pc_mesh = trimesh.PointCloud(vertices=aligned_pts, colors=all_cols)
        scene.add_geometry(pc_mesh, geom_name="point_cloud")

        import matplotlib
        colormap = matplotlib.colormaps.get_cmap("gist_rainbow")
        for i, c2w in enumerate(c2w_list):
            if c2w is None:
                continue
            R0, t0 = c2w[:3, :3], c2w[:3, 3]
            R0_inv = R0.T
            t0_inv = -R0_inv @ t0
            c2w_v = np.eye(4, dtype=np.float64)
            c2w_v[:3, :3] = R0_inv
            c2w_v[:3, 3] = t0_inv

            rgba_c = colormap(i / max(len(c2w_list) - 1, 1))
            cam_color = tuple(int(255 * x) for x in rgba_c[:3])

            opengl = get_opengl_conversion_matrix()
            align_rot = np.eye(4, dtype=np.float64)
            align_rot[:3, :3] = SRT.from_euler("y", 180, degrees=True).as_matrix()
            cam_transform = c2w_v @ opengl @ align_rot

            integrate_camera_into_scene(scene, cam_transform, cam_color, scene_scale)

        path = os.path.join(out_dir, "scene.glb")
        scene.glb(path)
        result["glb"] = path
        print(f"[Export] GLB: {path}")

    if npz_out:
        path = os.path.join(out_dir, "reconstruction.npz")
        np.savez(path, points=aligned_pts, colors=all_cols,
                 c2w_arr=np.stack(c2w_list) if c2w_list else np.zeros((0, 4, 4)),
                 images=images_out if images_out else np.array([]))
        result["npz"] = path
        print(f"[Export] NPZ: {path}")

    if export_video:
        print("[Export] Rendering MP4 (requires open3d)...")
        mp4_path = os.path.join(out_dir, "reconstruction.mp4")
        try:
            _export_video_fallback(aligned_pts, all_cols, c2w_list, mp4_path)
        except Exception as e:
            print(f"[Export] MP4 export failed: {e}")
        else:
            result["mp4"] = mp4_path

    print(f"[Export] Done: {out_dir}")
    return result


def _export_video_fallback(points, colors, c2w_arr, output_path):
    """Render a simple MP4 flythrough using Open3D offscreen rendering."""
    try:
        import open3d as o3d
    except ImportError:
        print("[Export] open3d not installed, skipping MP4 export")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if len(colors) > 0:
        if colors.max() > 1:
            colors = colors.astype(np.float32) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1280, height=720)
    vis.add_geometry(pcd)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (1280, 720))

    for i, c2w in enumerate(c2w_arr):
        if i % 10 != 0:
            continue
        c2w_copy = c2w.copy()
        c2w_copy[:3, 3] = c2w_copy[:3, 3] - points.mean(axis=0)
        vis.poll_events()
        vis.update_renderer()
        img = vis.capture_screen_float_buffer(do_render=True)
        img = (np.asarray(img) * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img)

    out.release()
    vis.destroy_window()
    print(f"[Export] MP4 saved: {output_path}")


# =============================================================================
# Viser Viewer — matches PointCloudViewer pattern (same as demo_realtimeV)
# =============================================================================

def start_viewer(host="0.0.0.0", port=8080, max_frames=300):
    state = {
        "frames": [],
        "running": True,
        "stop_requested": False,
        "conf_pct": 1.5,
        "downsample_factor": 10,
        "point_size": 0.012,
        "max_viewer_frames": max_frames,
        "show_all_frames": True,
        "current_timestep": 0,
        "show_trajectory": True,
        "show_camera_feed": True,
    }
    pc_handle = [None]
    cam_handles = {}
    traj_handle = [None]
    feed_handle = [None]
    actual_port = [None]

    def viewer_thread_fn():
        import viser
        import viser.transforms as _tf

        for cand in [port, port + 1, port + 2]:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", cand))
                actual_port[0] = cand
                break
            except OSError:
                continue
        else:
            actual_port[0] = port

        print(f"[Viewer] Starting on {host}:{actual_port[0]}...")
        server = viser.ViserServer(host=host, port=actual_port[0], _local_only=False)
        print(f"[Viewer] Open: http://127.0.0.1:{actual_port[0]}")

        try:
            server.scene.configure_default_lights(enabled=True, cast_shadow=False)
            server.scene.add_frame("/world_axes", show_axes=True, axes_length=0.5)
        except Exception:
            pass

        server.gui.configure_theme(control_layout="collapsible")

        # Confidence: absolute threshold (matching official PointCloudViewer)
        server.gui.add_slider("Confidence Threshold", 0.1, 5.0, step=0.01,
                              initial_value=1.5).on_update(
            lambda v: state.update({"conf_pct": float(v.target.value)}))

        # Downsample
        server.gui.add_slider("Downsample", 1, 32, step=1,
                              initial_value=10).on_update(
            lambda v: state.update({"downsample_factor": int(v.target.value)}))

        # Point size
        server.gui.add_slider("Point Size", 0.000001, 0.05, step=0.000001,
                              initial_value=0.012).on_update(
            lambda v: state.update({"point_size": float(v.target.value)}))

        # Show all frames (accumulate) or single frame
        server.gui.add_checkbox("Show All Frames (Accumulate)", True).on_update(
            lambda v: state.update({"show_all_frames": v.target.value}))

        server.gui.add_checkbox("Show Trajectory", True).on_update(
            lambda v: state.update({"show_trajectory": v.target.value}))

        server.gui.add_checkbox("Show Camera Feed", True).on_update(
            lambda v: state.update({"show_camera_feed": v.target.value}))

        server.gui.add_button("Stop & Export").on_click(
            lambda _: state.update({"stop_requested": True}))

        def scene_center_scale(frames):
            if not frames:
                return np.zeros(3), 1.0

            cam_pos = []
            for f in frames:
                c2w = f.get("c2w")
                if c2w is not None:
                    cam_pos.append(c2w[:3, 3].copy())

            if not cam_pos:
                return np.zeros(3), 1.0

            positions = np.array(cam_pos)
            center = positions.mean(axis=0)
            if len(positions) > 1:
                extent = np.ptp(positions, axis=0)
                scale = np.linalg.norm(extent)
            else:
                scale = 1.0
            return center, max(scale, 0.1)

        def render_scene():
            frames = state["frames"]
            if not frames:
                return

            center, scale = scene_center_scale(frames)
            conf_pct = state["conf_pct"]
            ds = max(1, state["downsample_factor"])
            ps = state["point_size"]
            show_all = state.get("show_all_frames", True)
            show_traj = state.get("show_trajectory", True)
            show_feed = state.get("show_camera_feed", True)

            if show_all:
                visible = frames
            else:
                ts = state.get("current_timestep", 0)
                visible = [frames[ts]] if ts < len(frames) else []

            if not visible:
                return

            all_pts, all_cols, all_conf = [], [], []
            cam_centered = []

            for f in visible:
                pc = f.get("pc")
                color = f.get("color")
                conf = f.get("conf")
                c2w = f.get("c2w")

                if pc is None or len(pc) == 0:
                    continue

                all_pts.append(np.asarray(pc).reshape(-1, 3))
                conf_arr = np.asarray(conf).flatten() if conf is not None else np.ones(len(pc))
                all_conf.append(conf_arr)

                col_arr = np.asarray(color).reshape(-1, 3) if color is not None else np.zeros((len(pc), 3))
                all_cols.append(col_arr)

                if c2w is not None:
                    cam_centered.append(c2w[:3, 3].copy())

            if not all_pts:
                return

            conf_flat = np.concatenate(all_conf)
            if len(conf_flat) == 0:
                return

            if conf_pct <= 100:
                thresh_val = np.percentile(conf_flat, conf_pct)
            else:
                thresh_val = conf_pct

            fp, fc = [], []
            for pts, cols, cf in zip(all_pts, all_cols, all_conf):
                n = len(pts)
                if cf.shape[0] != n:
                    cf = np.resize(cf.flatten(), n)
                mask = (cf >= thresh_val) & (cf > 1e-5)
                pf = pts[mask]
                ca = cols[mask] if len(cols) == n else cols
                if len(pf) == 0:
                    continue
                if ds > 1:
                    idx = np.arange(0, len(pf), ds)
                    pf, ca = pf[idx], ca[idx]
                if len(pf) > 0:
                    fp.append(pf)
                    fc.append(ca)

            if not fp:
                return

            g_pts = np.concatenate(fp).astype(np.float32)
            g_cols = np.concatenate(fc)
            if len(g_cols) != len(g_pts):
                g_cols = np.zeros((len(g_pts), 3), dtype=np.float32)

            valid = np.isfinite(g_pts).all(axis=1)
            g_pts = g_pts[valid]
            g_cols = g_cols[valid]

            g_s = (g_pts - center) / scale
            if cam_centered:
                cam_c = ((np.array(cam_centered) - center) / scale).astype(np.float32)
            else:
                cam_c = np.zeros((0, 3), dtype=np.float32)

            if g_cols.dtype == np.uint8:
                g_cols = g_cols.astype(np.float32) / 255.0
            elif g_cols.max() > 1.5:
                g_cols = g_cols.astype(np.float32) / 255.0
            else:
                g_cols = g_cols.astype(np.float32)
            g_cols = np.clip(g_cols, 0.0, 1.0)

            with server.atomic():
                try:
                    if pc_handle[0] is None:
                        pc_handle[0] = server.scene.add_point_cloud(
                            "_pc",
                            points=g_s,
                            colors=g_cols,
                            point_size=ps,
                            point_shading="flat",
                        )
                    else:
                        pc_handle[0].points = g_s
                        pc_handle[0].colors = g_cols
                        pc_handle[0].point_size = ps
                except Exception as e:
                    print(f"[Viewer] PC update error: {e}")

                vis_keys = {i for i in range(len(visible)) if i % ds == 0}
                for i, c_v in zip(range(len(visible)), cam_c):
                    if i not in vis_keys:
                        continue
                    c2w = visible[i].get("c2w")
                    if c2w is None:
                        continue
                    key = i
                    try:
                        R_centered = c2w[:3, :3]
                        t_centered = (c2w[:3, 3] - center) / scale
                        wxyz = _tf.SO3.from_matrix(R_centered).wxyz
                        pos = tuple(float(x) for x in t_centered)

                        if key not in cam_handles:
                            cam_handles[key] = server.scene.add_camera_frustum(
                                f"_c{key}",
                                fov=0.8,
                                aspect=1.4,
                                scale=scale * 0.05,
                                color=(0, 1, 0),
                                wxyz=wxyz,
                                position=pos,
                                variant="wireframe",
                            )
                        else:
                            cam_handles[key].wxyz = wxyz
                            cam_handles[key].position = pos
                            cam_handles[key].visible = True
                    except Exception:
                        pass

                for key in list(cam_handles.keys()):
                    if key not in vis_keys:
                        try:
                            cam_handles[key].visible = False
                        except Exception:
                            pass

                if show_traj and len(cam_c) >= 2:
                    try:
                        n = len(cam_c)
                        segs = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.int32)
                        t_vals = np.linspace(0, 1, max(n - 1, 1))
                        cmap = np.zeros((max(n - 1, 1), 3), dtype=np.float32)
                        cmap[:, 0] = np.clip(np.abs(t_vals * 6 - 3) - 1, 0, 1)
                        cmap[:, 1] = np.clip(2 - np.abs(t_vals * 6 - 2), 0, 1)
                        cmap[:, 2] = np.clip(2 - np.abs(t_vals * 6 - 4), 0, 1)

                        if traj_handle[0] is None:
                            traj_handle[0] = server.scene.add_line_set(
                                "_traj",
                                points=cam_c,
                                segments=segs,
                                colors=cmap,
                                line_width=2.0,
                            )
                        else:
                            traj_handle[0].points = cam_c
                            traj_handle[0].segments = segs
                            traj_handle[0].colors = cmap
                    except Exception:
                        pass
                elif traj_handle[0]:
                    try:
                        traj_handle[0].visible = False
                    except Exception:
                        pass

                latest_img = None
                for f in reversed(visible):
                    img = f.get("image")
                    if img is not None:
                        latest_img = img
                        break

                if show_feed and latest_img is not None:
                    try:
                        arr = np.asarray(latest_img)
                        if arr.dtype == np.float32 and arr.max() <= 1.0:
                            arr = (arr * 255).astype(np.uint8)
                        elif arr.dtype != np.uint8:
                            arr = arr.clip(0, 255).astype(np.uint8)
                        if arr.shape[-1] != 3:
                            arr = arr[:, :, :3]

                        if feed_handle[0] is None:
                            feed_handle[0] = server.gui.add_image(
                                image=arr,
                                label="Camera Feed",
                            )
                        else:
                            feed_handle[0].image = arr
                    except Exception:
                        pass
                elif not show_feed and feed_handle[0] is not None:
                    try:
                        feed_handle[0].visible = False
                    except Exception:
                        pass

        last_rep = [0]
        while state["running"]:
            try:
                render_scene()
            except Exception as e:
                print(f"[Viewer] render error: {e}")
            n = len(state["frames"])
            now = time.time()
            if n != last_rep[0] and now - last_rep[0] > 5:
                print(f"[Viewer] {n} frames accumulated")
                last_rep[0] = n
            time.sleep(0.25)

    t = threading.Thread(target=viewer_thread_fn, daemon=True)
    t.start()
    return t, state, actual_port


# =============================================================================
# Main inference loop
# =============================================================================

def run(args):
    device = torch.device(args.device if hasattr(args, "device") else
                         "cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"[Device] {device}")

    # ---- Load model ----
    from lingbot_map.models.gct_stream import GCTStream
    print("[Model] Building...")
    model = GCTStream(
        img_size=args.image_size,
        patch_size=args.patch_size,
        enable_3d_rope=args.enable_3d_rope,
        max_frame_num=args.max_frame_num,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        kv_cache_scale_frames=args.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=args.use_sdpa,
        camera_num_iterations=args.camera_num_iterations,
    )
    if args.model_path:
        print(f"[Model] Loading: {args.model_path}")
        ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        del ckpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] {gpu_mem:.1f} GB, dtype={dtype}")
    else:
        dtype = torch.float32

    if dtype != torch.float32 and getattr(model, "aggregator", None):
        print(f"[Model] Casting to {dtype}")
        model.aggregator = model.aggregator.to(dtype=dtype)
    model = model.to(device).eval()

    # Canvas dims (demo.py crop mode)
    raw_h = round(args.image_size * 360 / 640)
    canvas_h = round(raw_h / args.patch_size) * args.patch_size
    canvas_w = args.image_size
    print(f"[Canvas] {canvas_w}x{canvas_h} (image_size={args.image_size})")

    scale_n = max(1, args.num_scale_frames)
    kv_reset = getattr(args, 'kv_reset_interval', 200)

    # ---- Warmup ----
    print(f"[Warmup] {canvas_w}x{canvas_h}, scale={scale_n}...")
    dummy = torch.zeros((1, scale_n, 3, canvas_h, canvas_w),
                        dtype=torch.float32, device=device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        model.inference_streaming(dummy, num_scale_frames=scale_n)
    model.clean_kv_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Camera / Video capture ----
    buf, buf_lock = [], threading.Lock()
    stop_cap = threading.Event()
    cap_err = [None]

    if args.video_file:
        def cap_loop():
            try:
                cap = VideoFileCapture(args.video_file, loop=args.loop_video)
                while not stop_cap.is_set():
                    ret, f = cap.read_throttled()
                    if ret:
                        with buf_lock:
                            buf.append(f.copy())
                            while len(buf) > 10:
                                buf.pop(0)
                    else:
                        time.sleep(0.05)
                cap.release()
            except Exception as e:
                cap_err[0] = e
    else:
        def cap_loop():
            try:
                cap = UVCCapture(
                    device=args.video_device,
                    width=args.image_width, height=args.image_height,
                    fps=args.fps, pixel_format=args.pixel_format,
                    use_gstreamer=args.use_gstreamer,
                    capture_fps=args.capture_fps,
                )
                while not stop_cap.is_set():
                    ret, f = cap.read_throttled()
                    if ret:
                        with buf_lock:
                            buf.append(f.copy())
                            while len(buf) > 10:
                                buf.pop(0)
                    else:
                        time.sleep(0.005)
                cap.release()
            except Exception as e:
                cap_err[0] = e

    threading.Thread(target=cap_loop, daemon=True).start()

    # ---- Viewer ----
    vstate = None
    try:
        _, vstate, _ = start_viewer(
            host=args.server_ip, port=args.port,
            max_frames=args.max_viewer_frames)
    except Exception as e:
        print(f"[Viewer] Failed to start: {e}")
        import traceback; traceback.print_exc()

    # ---- Loop detection ----
    loop_detector = None
    if args.enable_loop_detection:
        loop_detector = LoopDetector(
            threshold=args.loop_threshold,
            min_interval=args.loop_min_interval,
            min_history=args.loop_min_history,
        )

    print("=" * 55)
    print("LingBot-MAP Real-Time V2 (depth unprojection + merged features)")
    print("=" * 55)
    src = f"video={os.path.basename(args.video_file)}" if args.video_file else args.video_device
    print(f"  Source     : {src}")
    print(f"  Canvas     : {canvas_w}x{canvas_h}")
    print(f"  Scale      : {scale_n}  Camera iters: {args.camera_num_iterations}")
    print(f"  KV reset   : every {kv_reset} frames")
    if loop_detector:
        print(f"  Loop detect: enabled (threshold={args.loop_threshold})")
    if args.enable_keyframe_gate:
        print(f"  Keyframe   : enabled (trans>{args.keyframe_trans_thresh}, rot>{args.keyframe_rot_thresh_deg}deg)")
    print("=" * 55)
    print("Press Ctrl+C to stop.\n")

    viewer_frames = []
    frame_idx = 0
    fps_win = []
    last_t = time.perf_counter()
    total_t = 0.0
    fps_rep = time.time()
    fps_cnt = 0
    stop_reason = "user"
    gc_interval = 20
    last_keyframe_c2w = None

    try:
        while True:
            if args.max_frames and frame_idx >= args.max_frames:
                stop_reason = "max_frames"
                break
            if vstate and vstate.get("stop_requested"):
                stop_reason = "gui"
                break
            if cap_err[0]:
                raise cap_err[0]

            with buf_lock:
                if not buf:
                    time.sleep(0.001)
                    continue
                frame_rgb = buf.pop(0)

            img_t = preprocess_frame(frame_rgb, args.image_size, args.patch_size)
            img_t = img_t.to(device)

            t0 = time.perf_counter()

            # ---- Bootstrap phase ----
            if frame_idx < scale_n:
                if len(buf) + 1 < scale_n - frame_idx:
                    with buf_lock:
                        buf.insert(0, frame_rgb.copy())
                    time.sleep(0.01)
                    continue

                boot_frames = [frame_rgb]
                with buf_lock:
                    while len(boot_frames) < scale_n:
                        if buf:
                            boot_frames.append(buf.pop(0))
                        else:
                            break

                boot_t = torch.stack([
                    preprocess_frame(f, args.image_size, args.patch_size).to(device)
                    for f in boot_frames
                ], dim=0).unsqueeze(0)

                model.clean_kv_cache()
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                    preds = model.forward(
                        boot_t,
                        num_frame_for_scale=scale_n,
                        num_frame_per_block=scale_n,
                        causal_inference=True,
                    )

                preds_cpu = {}
                for k, v in preds.items():
                    if isinstance(v, torch.Tensor):
                        preds_cpu[k] = v.detach().cpu()
                    else:
                        preds_cpu[k] = v

                depth = preds_cpu.get("depth")
                depth_conf = preds_cpu.get("depth_conf")
                pose_enc = preds_cpu.get("pose_enc")

                if depth is not None:
                    depth_np = depth[-1, -1, :, :, 0].numpy()
                else:
                    depth_np = None

                if depth_conf is not None:
                    conf_np = depth_conf[-1, -1].numpy().flatten().astype(np.float32)
                else:
                    conf_np = None

                pe = pose_enc[-1, -1].numpy() if pose_enc is not None else None

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                infer_t = time.perf_counter() - t0

                bootstrap_done = True
                print(f"[Bootstrap] Done at frame {frame_idx}, processing last of {scale_n} frames")

            else:
                # ---- Streaming phase ----
                fb = img_t.unsqueeze(0).unsqueeze(0)

                if (frame_idx - scale_n) % kv_reset == 0 and frame_idx > scale_n:
                    print(f"[KVReset] Reset at frame {frame_idx}")
                    model.clean_kv_cache()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                        _ = model.forward(fb, num_frame_for_scale=1,
                                          num_frame_per_block=1, causal_inference=True)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    frame_idx += 1
                    continue

                with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                    preds = model.forward(
                        fb,
                        num_frame_for_scale=scale_n,
                        num_frame_per_block=1,
                        causal_inference=True,
                    )

                preds_cpu = {}
                for k, v in preds.items():
                    if isinstance(v, torch.Tensor):
                        preds_cpu[k] = v.detach().cpu()
                    else:
                        preds_cpu[k] = v

                depth = preds_cpu.get("depth")
                depth_conf = preds_cpu.get("depth_conf")
                pose_enc = preds_cpu.get("pose_enc")

                if depth is not None:
                    depth_np = depth[-1, -1, :, :, 0].numpy()
                else:
                    depth_np = None

                if depth_conf is not None:
                    conf_np = depth_conf[-1, -1].numpy().flatten().astype(np.float32)
                else:
                    conf_np = None

                pe = pose_enc[-1, -1].numpy() if pose_enc is not None else None

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                infer_t = time.perf_counter() - t0

            # ---- Convert pose encoding to extrinsic/intrinsic ----
            if pe is not None:
                pe_t = torch.from_numpy(pe).float().unsqueeze(0).unsqueeze(0)
                ext, intr = pose_encoding_to_extri_intri(pe_t, (canvas_h, canvas_w))

                ext_4x4 = torch.zeros(ext.shape[:-2] + (4, 4), dtype=ext.dtype)
                ext_4x4[..., :3, :4] = ext
                ext_4x4[..., 3, 3] = 1.0
                ext_4x4 = closed_form_inverse_se3_general(ext_4x4)
                c2w_np = ext_4x4.squeeze().numpy()
                ext_np = ext[-1, -1].numpy()
                intr_np = intr[-1, -1].numpy() if intr is not None else None
            else:
                c2w_np = None
                ext_np = None
                intr_np = None

            has_nan = c2w_np is not None and (np.any(np.isnan(c2w_np)) or np.any(np.isinf(c2w_np)))

            # ---- Depth unprojection to world coordinates ----
            world_pts = None
            cam_coords = None

            sky_thresh = args.sky_threshold if args.mask_sky else 0.0

            if depth_np is not None and not has_nan and ext_np is not None and intr_np is not None:
                try:
                    world_pts, cam_coords = unproject_depth_frame(
                        depth_np, ext_np, intr_np, sky_threshold=sky_thresh
                    )
                except Exception as e:
                    print(f"[Unproject] Error: {e}")

            # ---- Color from original camera frame ----
            H, W = depth_np.shape if depth_np is not None else (canvas_h, canvas_w)
            # frame_rgb is already RGB from cv2.cvtColor(frame, COLOR_BGR2RGB)
            # Just resize to match depth dimensions (no BGR→RGB needed)
            color_np = cv2.resize(frame_rgb.astype(np.float32) / 255.0, (W, H)).reshape(-1, 3).astype(np.float32)

            # ---- Assemble frame data ----
            if world_pts is not None:
                pc = world_pts.reshape(-1, 3).astype(np.float32)
            else:
                pc = np.zeros((0, 3), dtype=np.float32)
                color_np = np.zeros((0, 3), dtype=np.float32)
                conf_np = np.zeros(0, dtype=np.float32)

            # ---- Keyframe gating ----
            accepted = False
            t_delta, r_delta = 0.0, 0.0
            if args.enable_keyframe_gate:
                accepted, t_delta, r_delta = _should_accept_keyframe(
                    last_keyframe_c2w, c2w_np,
                    translation_thresh=args.keyframe_trans_thresh,
                    rotation_thresh_deg=args.keyframe_rot_thresh_deg,
                )
            else:
                accepted = c2w_np is not None

            if accepted:
                frame_data = {
                    "pc": pc,
                    "color": color_np,
                    "conf": conf_np,
                    "c2w": c2w_np,
                    "image": frame_rgb,
                }
                viewer_frames.append(frame_data)
                last_keyframe_c2w = c2w_np.copy() if c2w_np is not None else last_keyframe_c2w
                if len(viewer_frames) > args.max_viewer_frames:
                    viewer_frames.pop(0)
            elif frame_idx % 20 == 0:
                print(f"[Viewer] skip frame {frame_idx}: "
                      f"(dt={t_delta:.6f}, dr={r_delta:.2f}deg)")

            # ---- Loop detection ----
            if loop_detector is not None and c2w_np is not None:
                if loop_detector.update(c2w_np):
                    stop_reason = "loop_closure"
                    print(f"[Inference] Loop closure at frame {frame_idx}")
                    break

            # ---- Update viewer ----
            if vstate is not None:
                vstate["frames"] = list(viewer_frames)

            # ---- FPS tracking ----
            fps_cnt += 1
            dt = time.perf_counter() - last_t
            last_t = time.perf_counter()
            total_t += dt
            fps_win.append(1.0 / dt if dt > 0 else 0)
            if len(fps_win) > 30:
                fps_win.pop(0)

            now = time.time()
            if now - fps_rep >= 5.0:
                avg = sum(fps_win) / len(fps_win) if fps_win else 0
                overall = fps_cnt / total_t if total_t > 0 else 0
                print(f"  FPS: {avg:.1f} (infer={infer_t*1000:.0f}ms) frames={frame_idx} "
                      f"viewer={len(viewer_frames)}")
                fps_rep = now

            frame_idx += 1

            if frame_idx % gc_interval == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    except KeyboardInterrupt:
        stop_reason = "keyboard"
        print("\n[Inference] Interrupted")

    print(f"[Inference] Stopped: {stop_reason}")
    stop_cap.set()
    if vstate:
        vstate["running"] = False
    time.sleep(0.5)

    out_dir = args.output_dir or f"realtimeV2_{stop_reason}"
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(os.getcwd(), out_dir)

    print(f"[Export] {len(viewer_frames)} frames -> {out_dir}")
    export_reconstruction(
        viewer_frames, out_dir,
        conf_pct=args.conf_threshold,
        ds_factor=args.downsample_factor,
        glb=args.export_glb,
        npz_out=args.export_npz,
        export_video=args.export_video,
    )

    overall = fps_cnt / total_t if total_t > 0 else 0
    print(f"\n  Overall FPS: {overall:.1f}")


# =============================================================================
# Entry point
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="LingBot-MAP Real-Time V2 (depth unprojection + merged features)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g = p.add_argument_group("Camera / Input")
    g.add_argument("--device", default="cuda")
    g.add_argument("--video_device", default="/dev/video0")
    g.add_argument("--video_file", default=None,
                   help="Run on a video file instead of camera (mutually exclusive with --video_device)")
    g.add_argument("--loop_video", action="store_true",
                   help="Loop the video file (for --video_file)")
    g.add_argument("--image_width", type=int, default=640)
    g.add_argument("--image_height", type=int, default=384)
    g.add_argument("--fps", type=int, default=20)
    g.add_argument("--capture_fps", type=int, default=None)
    g.add_argument("--pixel_format", default="MJPG")
    g.add_argument("--use_gstreamer", action="store_true")

    g = p.add_argument_group("Model")
    g.add_argument("--model_path", required=True)
    g.add_argument("--image_size", type=int, default=518)
    g.add_argument("--patch_size", type=int, default=14)
    g.add_argument("--num_scale_frames", type=int, default=8)
    g.add_argument("--enable_3d_rope", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--max_frame_num", type=int, default=256)
    g.add_argument("--kv_reset_interval", type=int, default=200)
    g.add_argument("--kv_cache_sliding_window", type=int, default=64)
    g.add_argument("--use_sdpa", action="store_true")
    g.add_argument("--camera_num_iterations", type=int, default=2)

    g = p.add_argument_group("Viewer")
    g.add_argument("--server_ip", default="0.0.0.0")
    g.add_argument("--port", type=int, default=8080)
    g.add_argument("--conf_threshold", type=float, default=1.5)
    g.add_argument("--downsample_factor", type=int, default=10)
    g.add_argument("--max_viewer_frames", type=int, default=300)
    g.add_argument("--point_size", type=float, default=0.012)

    g = p.add_argument_group("Stop")
    g.add_argument("--max_frames", type=int, default=None)
    g.add_argument("--enable_loop_detection", action="store_true")
    g.add_argument("--loop_threshold", type=float, default=0.5)
    g.add_argument("--loop_min_interval", type=int, default=30)
    g.add_argument("--loop_min_history", type=int, default=30)
    g.add_argument("--enable_keyframe_gate", action="store_true",
                   help="Only show frames when camera moves significantly")
    g.add_argument("--keyframe_trans_thresh", type=float, default=0.0025,
                   help="Min translation for accepting frame (default: 0.0025)")
    g.add_argument("--keyframe_rot_thresh_deg", type=float, default=6.0,
                   help="Min rotation deg for accepting frame (default: 6.0)")

    g = p.add_argument_group("Filter")
    g.add_argument("--mask_sky", action="store_true",
                   help="Filter out sky pixels using depth threshold")
    g.add_argument("--sky_threshold", type=float, default=50.0,
                   help="Depth above this (meters) is treated as sky (default: 50.0)")

    g = p.add_argument_group("Export")
    g.add_argument("--output_dir", default=None)
    g.add_argument("--export_glb", action="store_true")
    g.add_argument("--export_npz", action="store_true")
    g.add_argument("--export_video", action="store_true",
                   help="Export MP4 flythrough (requires open3d)")

    args = p.parse_args()

    if args.enable_loop_detection and args.loop_min_history < args.loop_min_interval:
        args.loop_min_history = args.loop_min_interval

    if args.video_file and args.video_device != "/dev/video0":
        print("[Warn] Both --video_file and --video_device specified; using --video_file")

    run(args)


if __name__ == "__main__":
    main()

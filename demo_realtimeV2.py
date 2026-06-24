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

    # Skip a static intro so bootstrap can estimate scale from motion frames
    python demo_realtimeV2.py --model_path lingbot-map.pt --video_file /path/to/video.mp4 \
        --video_start_frame 400

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

    def __init__(self, video_path, loop=False, verbose=True, frame_skip=1, start_frame=0,
                 throttle=True, throttle_fps=None):
        self.video_path = video_path
        self.loop = loop
        self.verbose = verbose
        self.frame_skip = max(1, frame_skip)
        self._skip_counter = 0
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.start_frame = max(0, min(int(start_frame), max(0, self.total_frames - 1)))
        if self.start_frame > 0:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            if self.verbose:
                print(f"[VideoFile] Start frame: {self.start_frame} (T={self.start_frame/self.fps:.1f}s)")
        if self.verbose:
            print(f"[VideoFile] {video_path}: {self.width}x{self.height} @ {self.fps:.1f}fps, {self.total_frames} frames")
        if self.frame_skip > 1:
            print(f"[VideoFile] Frame skip: every {self.frame_skip} frames (effective ~{self.fps/self.frame_skip:.1f} fps)")
        # Throttle cap_loop so the buffer holds frames from --video_start_frame,
        # not the end of the video. Without throttling, cap_loop races to the end
        # in a few seconds and bootstrap uses frames from the end (often stationary).
        self._throttle = bool(throttle)
        self._throttle_fps = float(throttle_fps) if throttle_fps else self.fps
        self._throttle_interval = 1.0 / max(self._throttle_fps, 0.1)
        self._last_read = 0.0
        if self.verbose and self._throttle:
            print(f"[VideoFile] Throttle: cap_loop reads at {self._throttle_fps:.1f} FPS "
                  f"({self._throttle_interval*1000:.1f}ms between frames)")
        self._released = False

    def read_throttled(self):
        """Read the next frame, optionally throttled to a target FPS.

        When throttling is enabled, sleeps until the next frame slot is due
        (real video FPS, or the override --cap_read_fps). This keeps cap_loop
        from racing ahead to the end of the video and starving the bootstrap
        of motion-rich frames at the start.

        Returns (True, frame_rgb) on success.
        Returns (False, None) on EOF or error.
        """
        if self._released:
            return False, None
        if self._throttle:
            while True:
                now = time.time()
                remaining = self._throttle_interval - (now - self._last_read)
                if remaining <= 0:
                    break
                time.sleep(min(0.005, remaining))
        while True:
            ret, frame = self._cap.read()
            if not ret:
                if self.loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self._cap.read()
                    if not ret:
                        return False, None
                else:
                    return False, None
            self._skip_counter += 1
            if self._skip_counter % self.frame_skip == 0:
                break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._last_read = time.time()
        return True, frame_rgb

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

    def __init__(self, threshold=0.5, min_interval=90, min_history=90, window_size=100,
                 warmup_frames=50):
        # Defaults bumped from 30 -> 90 because at ~1.4 FPS pipeline the
        # trajectory drifts ~0.01 m/frame, so 30 frames is only ~0.3 m of
        # motion — way under the 0.5 m threshold. That produced constant
        # false-positive loop closures during normal forward motion.
        self.threshold = threshold
        self.min_interval = min_interval
        self.min_history = min_history
        self.warmup_frames = warmup_frames  # skip loop checks for first N frames
        self.frames = []
        self.recent_window = []
        self.frame_idx = 0
        self.warmed_up = False

    def update(self, c2w):
        # Always record the frame so reset()/visualization can use history,
        # but skip the loop check during warmup.
        if self.frame_idx < self.min_history:
            self.frame_idx += 1
            self.frames.append((self.frame_idx, c2w))
            self.recent_window.append((self.frame_idx, c2w))
            if len(self.recent_window) > self.min_interval:
                self.recent_window.pop(0)
            return False

        if not self.warmed_up and self.frame_idx >= self.warmup_frames:
            self.warmed_up = True
            print(f"[LoopDetect] Warmup done at frame {self.frame_idx} — "
                  f"loop detection is now active")

        pos = c2w[:3, 3]
        self.frames.append((self.frame_idx, c2w))
        self.recent_window.append((self.frame_idx, c2w))
        if len(self.recent_window) > self.min_interval:
            self.recent_window.pop(0)

        # Don't fire loop closures during the warmup window. This prevents
        # false positives right after bootstrap (when positions can still be
        # noisy / rescaling) and during the early "drifting forward" phase.
        if not self.warmed_up:
            self.frame_idx += 1
            return False

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
        """Clear all history. Use when bootstrap is re-run (e.g. after a
        low-scale bootstrap) so the broken bootstrap positions don't pollute
        the loop-closure search."""
        self.frames = []
        self.recent_window = []
        self.frame_idx = 0
        self.warmed_up = False


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
# Motion quality checks
# =============================================================================

def _detect_pose_jump(c2w_prev, c2w_curr, t_thresh=0.5, r_thresh_deg=30.0):
    """Detect abnormal pose jumps between consecutive frames.

    Large jumps indicate the pose predictor lost tracking, so the frame's
    point cloud should be heavily downsampled or skipped.
    """
    if c2w_prev is None or c2w_curr is None:
        return False, 0.0, 0.0
    t_delta = float(np.linalg.norm(c2w_curr[:3, 3] - c2w_prev[:3, 3]))
    r_delta = _rotation_angle_deg(c2w_prev, c2w_curr)
    is_jump = (t_delta > t_thresh) or (r_delta > r_thresh_deg)
    return is_jump, t_delta, r_delta


def detect_depth_degradation(depth_np, conf_np=None, ref_depth=None):
    """Detect when depth prediction has degraded (motion blur, tracking loss).

    Checks:
    - NaN/Inf ratio in depth map
    - Depth standard deviation (too low or too high = bad)
    - Drift from reference depth (if provided)

    Returns:
        bool: True if depth is degraded
        dict: diagnostic info
    """
    if depth_np is None:
        return True, {}

    info = {}
    valid_mask = np.isfinite(depth_np) & (depth_np > 0)
    valid_ratio = np.sum(valid_mask) / depth_np.size
    info["valid_ratio"] = valid_ratio

    valid_flat = valid_mask.flatten()
    valid_depth = depth_np.flatten()[valid_flat]
    if len(valid_depth) == 0:
        return True, info

    depth_mean = float(np.mean(valid_depth))
    depth_std = float(np.std(valid_depth))
    info["depth_mean"] = depth_mean
    info["depth_std"] = depth_std

    nan_ratio = 1.0 - valid_ratio
    info["nan_ratio"] = nan_ratio

    if conf_np is not None and len(conf_np) > 0:
        conf_arr = conf_np.flatten()[valid_flat]
        conf_mean = float(np.mean(conf_arr)) if len(conf_arr) > 0 else None
        info["conf_mean"] = conf_mean
    else:
        conf_mean = None

    degraded = False
    bad_count = 0
    if nan_ratio > 0.5:
        bad_count += 1
    if depth_std < 0.05 and depth_mean > 0.5:
        bad_count += 1
    if bad_count >= 2:
        degraded = True

    if ref_depth is not None:
        ref_valid = np.isfinite(ref_depth) & (ref_depth > 0)
        if ref_valid.any() and valid_mask.any():
            drift = abs(depth_mean - float(np.mean(ref_depth[ref_valid])))
            info["depth_drift"] = drift
            if drift > 5.0:
                degraded = True

    # Only flag as degraded if multiple indicators are bad (conservative)
    bad_count = 0
    if nan_ratio > 0.5:
        bad_count += 1
    if depth_std < 0.05 and depth_mean > 0.5:
        bad_count += 1
    if bad_count >= 2:
        degraded = True

    return degraded, info


def depth_consistency_check(world_pts, depth_ref, c2w_new, c2w_ref, intr, thresh=0.5):
    """Check 3D point consistency by reprojecting into reference frame.

    For each 3D point in world_coords, project it into the reference camera
    and compare the reprojected depth with the reference depth map.
    Points with large depth discrepancy are marked as unreliable.

    Args:
        world_pts: [H, W, 3] world coordinates from new frame
        depth_ref: [H, W] reference depth map
        c2w_new: [4, 4] new frame camera-to-world
        c2w_ref: [4, 4] reference frame camera-to-world
        intr: [3, 3] intrinsic matrix
        thresh: max relative depth error (default 0.5m)

    Returns:
        mask: [H*W] boolean array, True = consistent point
    """
    H, W = world_pts.shape[:2]
    world_flat = world_pts.reshape(-1, 3)

    R_ref = c2w_ref[:3, :3]
    t_ref = c2w_ref[:3, 3]
    w2c_ref = np.linalg.inv(c2w_ref)
    R_new_inv = c2w_new[:3, :3].T
    t_new = c2w_new[:3, 3]

    cam_pts_ref = (world_flat - t_ref) @ R_ref
    z_ref = cam_pts_ref[:, 2]

    valid = (z_ref > 0.1) & np.isfinite(z_ref)

    u = (cam_pts_ref[:, 0] / cam_pts_ref[:, 2]) * intr[0, 0] + intr[0, 2]
    v = (cam_pts_ref[:, 1] / cam_pts_ref[:, 2]) * intr[1, 1] + intr[1, 2]
    pix_valid = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    pix_valid = pix_valid & (z_ref > 0)

    mask_flat = np.zeros(H * W, dtype=bool)
    if pix_valid.any():
        u_int = u.astype(int)
        v_int = v.astype(int)
        ref_z = np.zeros(H * W)
        idx = np.arange(H * W)
        ref_z[idx[pix_valid]] = depth_ref[v_int[pix_valid], u_int[pix_valid]]
        ref_z_valid = ref_z[idx[pix_valid]]

        depth_err = np.abs(z_ref[pix_valid] - ref_z_valid)
        consistent = depth_err < thresh
        mask_flat[idx[pix_valid]] = consistent
    else:
        mask_flat[:] = valid

    return mask_flat


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
        scene.export(path)
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

def start_viewer(host="0.0.0.0", port=8080, max_frames=300, camera_downsample=1):
    state = {
        "frames": [],
        "running": True,
        "stop_requested": False,
        "conf_pct": 1.5,
        "downsample_factor": 10,
        "point_size": 0.005,
        "max_viewer_frames": max_frames,
        "show_all_frames": True,
        "current_timestep": 0,
        "show_trajectory": True,
        "show_camera_feed": True,
        "camera_downsample": camera_downsample,
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
        # Raised to 2.5 for better quality during fast motion
        server.gui.add_slider("Confidence Threshold", 0.1, 5.0, step=0.01,
                              initial_value=1.5).on_update(
            lambda v: state.update({"conf_pct": float(v.target.value)}))

        # Downsample
        server.gui.add_slider("Downsample", 1, 32, step=1,
                              initial_value=10).on_update(
            lambda v: state.update({"downsample_factor": int(v.target.value)}))

        # Point size
        server.gui.add_slider("Point Size", 0.000001, 0.05, step=0.000001,
                              initial_value=0.005).on_update(
            lambda v: state.update({"point_size": float(v.target.value)}))

        # Camera frustum downsample (only every Nth camera is rendered)
        server.gui.add_slider("Camera Stride", 1, 50, step=1,
                              initial_value=state["camera_downsample"]).on_update(
            lambda v: state.update({"camera_downsample": int(v.target.value)}))

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
                    if pc_handle[0] is not None:
                        pc_handle[0].remove()
                    pc_handle[0] = server.scene.add_point_cloud(
                        "_pc",
                        points=g_s,
                        colors=g_cols,
                        point_size=ps,
                        point_shading="flat",
                    )
                except Exception as e:
                    print(f"[Viewer] PC update error: {e}")

                cam_ds = max(1, state.get("camera_downsample", 1))
                vis_keys = {i for i in range(len(visible)) if i % max(ds, cam_ds) == 0}
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
                                scale=scale * 0.015,
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
                        if traj_handle[0] is not None:
                            traj_handle[0].remove()
                        n = len(cam_c)
                        pts = np.array([[cam_c[i], cam_c[i + 1]] for i in range(n - 1)], dtype=np.float32)
                        t_vals = np.linspace(0, 1, max(n - 1, 1))
                        cmap = np.zeros((max(n - 1, 1), 2, 3), dtype=np.float32)
                        cmap[:, 0, 0] = np.clip(np.abs(t_vals * 6 - 3) - 1, 0, 1)
                        cmap[:, 0, 1] = np.clip(2 - np.abs(t_vals * 6 - 2), 0, 1)
                        cmap[:, 0, 2] = np.clip(2 - np.abs(t_vals * 6 - 4), 0, 1)
                        cmap[:, 1] = cmap[:, 0]
                        traj_handle[0] = server.scene.add_line_segments(
                            "_traj",
                            points=pts,
                            colors=cmap,
                            line_width=2.0,
                        )
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
    frame_numbers = []  # shared with cap_loop; frame_numbers[i] = video frame number for buf[i]
    stop_cap = threading.Event()
    cap_err = [None]

    if args.video_file:
        def cap_loop():
            nonlocal cap_finished
            try:
                cap = VideoFileCapture(args.video_file, loop=args.loop_video,
                                       frame_skip=args.frame_skip,
                                       start_frame=getattr(args, "video_start_frame", 0),
                                       throttle=True,
                                       throttle_fps=getattr(args, "cap_read_fps", None))
                n_read = 0
                last_dbg = time.time()
                last_pos_report = -1
                # Hard cap: keep at most cap_buf_high * 2 frames in buf.
                # This allows cap_loop to race ahead and build a large buffer
                # of video frames so the inference thread always has fresh
                # frames with good temporal diversity for scale estimation.
                hard_cap = args.cap_buf_high * 2
                while not stop_cap.is_set():
                    ret, f = cap.read_throttled()
                    if ret:
                        n_read += 1
                        vid_pos = int(cap._cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                        with buf_lock:
                            buf.append(f.copy())
                            frame_numbers.append(vid_pos)
                            while len(buf) > hard_cap:
                                buf.pop(0)
                                frame_numbers.pop(0)
                        now = time.time()
                        if now - last_dbg >= 2.0:
                            pos = int(cap._cap.get(cv2.CAP_PROP_POS_FRAMES))
                            tot = int(cap._cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            print(f"[CapLoop] alive: n_read={n_read} buf={len(buf)} "
                                  f"frame_pos={pos}/{tot} ({100.0*pos/max(1,tot):.1f}%)")
                            last_dbg = now
                            last_pos_report = pos
                    else:
                        pos = int(cap._cap.get(cv2.CAP_PROP_POS_FRAMES))
                        tot = int(cap._cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        print(f"[CapLoop] read_throttled returned False: n_read={n_read} "
                              f"frame_pos={pos}/{tot} loop={args.loop_video} "
                              f"stop={stop_cap.is_set()}")
                        if n_read > 0:
                            print(f"[VideoFile] Finished reading {n_read} frames")
                        break
                cap.release()
            except Exception as e:
                print(f"[VideoFile] cap_loop exception: {e}")
                import traceback; traceback.print_exc()
                cap_err[0] = e
            cap_finished = True
            print(f"[CapLoop] EXITED: cap_finished=True, n_read={n_read}")
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
            max_frames=args.max_viewer_frames,
            camera_downsample=args.camera_downsample)
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
            warmup_frames=args.loop_warmup_frames,
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
        print(f"  Loop detect: enabled (threshold={args.loop_threshold}, "
              f"min_interval={args.loop_min_interval}, "
              f"warmup={args.loop_warmup_frames} frames)")
    if args.enable_keyframe_gate:
        print(f"  Keyframe   : enabled (trans>{args.keyframe_trans_thresh}, rot>{args.keyframe_rot_thresh_deg}deg)")
    if getattr(args, "diag_verbose", False):
        print("  Diag verbose: ON (per-frame pose/keyframe/cap_loop logs)")
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
    bootstrap_done = False
    last_keyframe_c2w = None
    last_valid_depth = None
    last_valid_c2w = None
    ref_depth_for_consistency = None
    ref_c2w_for_consistency = None
    max_t_norm = 0.0  # track max |t| to detect catastrophic pose collapse
    # When bootstrap produces a very small scale (|t|<0.01), the camera head's
    # scale context is effectively invalid. We skip tracking initialization and
    # let the first streaming frame with a real-scale pose establish it.
    _bootstrap_scale_ok = True
    _streaming_pose_init = False  # becomes True after first valid streaming pose

    cap_finished = False
    kv_warmed_up = True  # bootstrap already warms the KV cache
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

            # Wait for next frame (30s timeout to avoid false early-stop)
            buf_wait_start = time.time()
            last_wait_log = buf_wait_start
            while True:
                with buf_lock:
                    if buf:
                        frame_rgb = buf.pop(0)
                        break
                    if cap_finished:
                        stop_reason = "video_end"
                        frame_rgb = None
                        break
                now = time.time()
                # Log every 2s while waiting for the next frame (helps diagnose stalls)
                if now - last_wait_log >= 2.0:
                    print(f"[BufWait] f={frame_idx} waiting for buf "
                          f"(waited {now-buf_wait_start:.1f}s, cap_finished={cap_finished}, "
                          f"buf={len(buf) if buf else 0})")
                    last_wait_log = now
                if now - buf_wait_start > 30.0:
                    stop_reason = "video_end"
                    frame_rgb = None
                    print(f"[Inference] Timeout waiting for frames (30s), "
                          f"cap_finished={cap_finished} "
                          f"buf_size={len(buf) if buf else 0}, stopping")
                    break
                time.sleep(0.1)
            if frame_rgb is None:
                break

            img_t = preprocess_frame(frame_rgb, args.image_size, args.patch_size)
            img_t = img_t.to(device)

            t0 = time.perf_counter()

            # ---- Bootstrap phase ----
            if frame_idx < scale_n:
                # Collect `scale_n` consecutive frames.
                #
                # IMPORTANT: we deliberately grab the MOST RECENT frames from the
                # buffer, not the oldest.  When cap_loop is racing ahead (e.g.
                # cap_read_fps=30, reading at real-time), the buffer at bootstrap
                # time holds the latest frames from the video — the ones closest
                # to the "live" camera position.  Grabbing the oldest frames
                # (frame 400, 401, ...) means we bootstrap on the very first
                # frames after video_start_frame, which are often static.
                #
                # The buffer is a FIFO queue of up to 400 frames.  We take the
                # `scale_n` most-recent entries (tail) so the model sees the
                # most motion-rich portion of what cap_loop has read so far.
                if len(buf) + 1 < scale_n:
                    with buf_lock:
                        buf.insert(0, frame_rgb.copy())
                    time.sleep(0.01)
                    continue

                with buf_lock:
                    # Take up to `scale_n` most-recent frames from buffer tail.
                    # boot_frames[-1] = most-recent frame (highest frame number).
                    num_from_buf = min(scale_n - 1, len(buf))
                    boot_frames = list(buf[-num_from_buf:])  # slice → newest last
                    # Prepend the current frame so chronological order is preserved.
                    boot_frames.insert(0, frame_rgb)

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

                # ---- Extract pose from the last bootstrap frame ----
                pe_boot = pe
                if pe_boot is not None:
                    pe_t_boot = torch.from_numpy(pe_boot).float().unsqueeze(0).unsqueeze(0)
                    ext_boot, intr_boot = pose_encoding_to_extri_intri(pe_t_boot, (canvas_h, canvas_w))
                    ext_4x4_boot = torch.zeros(ext_boot.shape[:-2] + (4, 4), dtype=ext_boot.dtype)
                    ext_4x4_boot[..., :3, :4] = ext_boot
                    ext_4x4_boot[..., 3, 3] = 1.0
                    ext_4x4_boot = closed_form_inverse_se3_general(ext_4x4_boot)
                    boot_c2w = ext_4x4_boot.squeeze().numpy()
                else:
                    boot_c2w = None

                # Initialize tracking state from the final bootstrap frame ONLY
                # if scale was estimated successfully. When |t|<0.01 the scale
                # context is garbage; skip initialization so the streaming phase
                # establishes scale from the first real-motion frame.
                _t_norm_b = float(np.linalg.norm(boot_c2w[:3,3])) if boot_c2w is not None else 0.0
                if _t_norm_b < 0.01:
                    _bootstrap_scale_ok = False
                    print(f"[Bootstrap] Done at frame {frame_idx}, last c2w initialized |t|={_t_norm_b:.4f}")
                    print(f"[Bootstrap][WARN] Scale is suspiciously small (|t|<0.01) — the bootstrap "
                          f"frames had very little motion, which usually means the camera was "
                          f"stationary at the start of the video. Try --video_start_frame N to skip "
                          f"the static intro, e.g. --video_start_frame 400")
                    # The bootstrap pose is essentially the origin; any subsequent
                    # motion will appear as a big jump and pollute the loop-closure
                    # history. Reset the loop detector so it starts clean.
                    if loop_detector is not None:
                        loop_detector.reset()
                        print(f"[LoopDetect] Reset after low-scale bootstrap")
                    # DO NOT initialize last_keyframe_c2w / last_valid_c2w / etc.
                    # Leave them as None so the streaming phase establishes scale.
                else:
                    _bootstrap_scale_ok = True
                    if boot_c2w is not None and not (np.any(np.isnan(boot_c2w)) or np.any(np.isinf(boot_c2w))):
                        last_keyframe_c2w = boot_c2w.copy()
                        last_valid_c2w = boot_c2w.copy()
                        last_valid_depth = depth_np.copy() if depth_np is not None else None
                        ref_depth_for_consistency = depth_np.copy() if depth_np is not None else None
                        ref_c2w_for_consistency = boot_c2w.copy() if boot_c2w is not None else None
                    print(f"[Bootstrap] Done at frame {frame_idx}, last c2w initialized |t|={_t_norm_b:.4f}")
                # Skip past bootstrap phase so subsequent frames enter streaming mode
                bootstrap_done = True
                frame_idx = scale_n
                continue

            else:
                # ---- Streaming phase ----
                fb = img_t.unsqueeze(0).unsqueeze(0)

                if kv_warmed_up and (frame_idx - scale_n) % kv_reset == 0 and frame_idx > scale_n:
                    print(f"[KVReset] Reset + warmup at frame {frame_idx}")
                    model.clean_kv_cache()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                        preds_warm = model.forward(fb, num_frame_for_scale=scale_n,
                                          num_frame_per_block=1, causal_inference=True)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    preds_cpu_warm = {}
                    for k, v in preds_warm.items():
                        if isinstance(v, torch.Tensor):
                            preds_cpu_warm[k] = v.detach().cpu()
                        else:
                            preds_cpu_warm[k] = v
                    depth_warm = preds_cpu_warm.get("depth")
                    depth_conf_warm = preds_cpu_warm.get("depth_conf")
                    pose_enc_warm = preds_cpu_warm.get("pose_enc")
                    depth_np_w = depth_warm[-1, -1, :, :, 0].numpy() if depth_warm is not None else None
                    pe_w = pose_enc_warm[-1, -1].numpy() if pose_enc_warm is not None else None
                    if pe_w is not None:
                        pe_t_w = torch.from_numpy(pe_w).float().unsqueeze(0).unsqueeze(0)
                        ext_w, _ = pose_encoding_to_extri_intri(pe_t_w, (canvas_h, canvas_w))
                        ext_4x4_w = torch.zeros(ext_w.shape[:-2] + (4, 4), dtype=ext_w.dtype)
                        ext_4x4_w[..., :3, :4] = ext_w
                        ext_4x4_w[..., 3, 3] = 1.0
                        ext_4x4_w = closed_form_inverse_se3_general(ext_4x4_w)
                        c2w_warm = ext_4x4_w.squeeze().numpy()
                        if not (np.any(np.isnan(c2w_warm)) or np.any(np.isinf(c2w_warm))):
                            last_valid_c2w = c2w_warm.copy()
                            if depth_np_w is not None:
                                last_valid_depth = depth_np_w.copy()
                    # Warmup done; advance frame and go to viewer update (don't add warmup frame)
                    kv_warmed_up = False
                    frame_idx += 1
                    goto_viewer_update = True
                else:
                    goto_viewer_update = False

                if goto_viewer_update:
                    # Warmup-only frame: skip inference and viewer update.
                    # Tracking state (last_valid_c2w etc.) was already updated above.
                    frame_idx += 1
                    if frame_idx % gc_interval == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
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

            # ---- Bootstrap failed: establish scale from the first real-motion
            #      streaming frame. When bootstrap produced |t|<0.01, the scale
            #      context is garbage. We skip tracking initialization and wait
            #      for the first streaming frame with genuine scale (|t|>0.1),
            #      then use its pose as the tracking reference. This lets the
            #      streaming inference recover scale naturally. ----
            if (not _bootstrap_scale_ok and not _streaming_pose_init
                    and c2w_np is not None and not has_nan):
                t_norm = float(np.linalg.norm(c2w_np[:3, 3]))
                if t_norm > 0.1:
                    last_keyframe_c2w = c2w_np.copy()
                    last_valid_c2w = c2w_np.copy()
                    last_valid_depth = depth_np.copy() if depth_np is not None else None
                    ref_depth_for_consistency = depth_np.copy() if depth_np is not None else None
                    ref_c2w_for_consistency = c2w_np.copy()
                    _streaming_pose_init = True
                    print(f"[ScaleInit] Bootstrap scale was invalid; established tracking from "
                          f"first valid streaming frame {frame_idx} with |t|={t_norm:.3f}")

            # ---- Motion quality checks (run before any point cloud work) ----
            is_pose_jump = False
            jump_t, jump_r = 0.0, 0.0
            if c2w_np is not None and last_valid_c2w is not None:
                is_pose_jump, jump_t, jump_r = _detect_pose_jump(last_valid_c2w, c2w_np)

            # ---- Catastrophic pose jump: reset KV cache to prevent
            #      poisoning the cache with a wildly wrong pose. The next
            #      frame is re-warmed using the last good c2w. ----
            kv_reset_for_jump = False
            if is_pose_jump and (jump_t > 1.0 or jump_r > 30.0):
                print(f"[KVReset] Catastrophic pose jump: dt={jump_t:.3f}, dr={jump_r:.1f}° "
                      f"at frame {frame_idx}, resetting KV cache")
                model.clean_kv_cache()
                if loop_detector is not None:
                    loop_detector.reset()
                    print(f"[LoopDetect] Reset after catastrophic pose jump")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Re-warm with current frame (still uses fb already prepared)
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                    _ = model.forward(fb, num_frame_for_scale=scale_n,
                                      num_frame_per_block=1, causal_inference=True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                kv_reset_for_jump = True

            # ---- Catastrophic pose recovery: detect tracking failure and re-bootstrap ----
            rebootstrap = False
            if c2w_np is not None and last_keyframe_c2w is not None and not bootstrap_done:
                t_curr = float(np.linalg.norm(c2w_np[:3, 3]))
                t_prev = float(np.linalg.norm(last_keyframe_c2w[:3, 3]))
                # Detect catastrophic scale collapse (|t| dropped >80%) or
                # absolute trajectory extent regressed significantly
                if t_prev > 1.0 and t_curr < t_prev * 0.2:
                    print(f"[Recovery] Detected pose collapse: |t| {t_prev:.3f} -> {t_curr:.3f}, re-bootstrapping")
                    rebootstrap = True

            is_depth_degraded = False
            depth_info = {}
            if depth_np is not None:
                is_depth_degraded, depth_info = detect_depth_degradation(
                    depth_np, conf_np, last_valid_depth
                )

            # ---- Catastrophic pose: log but DON'T skip -- just log warning ----
            # The catastrophic detection keeps triggering once it fires, so we just warn
            # without affecting the pipeline.
            if c2w_np is not None and max_t_norm > 1.0:
                curr_t = float(np.linalg.norm(c2w_np[:3, 3]))
                if curr_t < max_t_norm * 0.2:
                    print(f"[Warning] Pose |t| dropped {max_t_norm:.3f} -> {curr_t:.3f}")

            quality_flags = []
            if is_pose_jump:
                quality_flags.append(f"jump(t={jump_t:.3f},r={jump_r:.1f})")
            if is_depth_degraded:
                quality_flags.append(f"degraded")
            quality_str = " ".join(quality_flags) if quality_flags else ""

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
                    is_depth_degraded = True

            # ---- Color from original camera frame ----
            H, W = depth_np.shape if depth_np is not None else (canvas_h, canvas_w)
            color_np = cv2.resize(frame_rgb.astype(np.float32) / 255.0, (W, H)).reshape(-1, 3).astype(np.float32)

            # ---- Confidence-aware depth filtering (apply to world_pts before flat) ----
            if world_pts is not None and conf_np is not None and len(conf_np) == H * W:
                conf_mask_2d = conf_np.reshape(H, W) >= args.conf_threshold
                world_pts[~conf_mask_2d] = 0.0
                if not conf_mask_2d.any():
                    is_depth_degraded = True

            # ---- Depth consistency check against previous frame ----
            if world_pts is not None and ref_depth_for_consistency is not None and ref_c2w_for_consistency is not None:
                c2w_4x4 = c2w_np if c2w_np.shape == (4, 4) else None
                ref_4x4 = ref_c2w_for_consistency if ref_c2w_for_consistency.shape == (4, 4) else None
                if c2w_4x4 is not None and ref_4x4 is not None and intr_np is not None:
                    try:
                        consistent_mask = depth_consistency_check(
                            world_pts, ref_depth_for_consistency,
                            c2w_4x4, ref_4x4, intr_np, thresh=0.8
                        )
                        world_pts_flat = world_pts.reshape(-1, 3)
                        valid_mask = np.isfinite(world_pts_flat).all(axis=1)
                        full_mask = consistent_mask & valid_mask
                        n_in = full_mask.sum()
                        n_out = (~consistent_mask).sum()
                        world_pts = world_pts_flat[full_mask]
                        color_np = color_np[full_mask]
                        conf_np = conf_np[full_mask] if conf_np is not None and len(conf_np) == H * W else conf_np
                        if n_in == 0:
                            world_pts = None
                            is_depth_degraded = True
                        elif frame_idx % 50 == 0:
                            print(f"[Consistency] frame={frame_idx} in={n_in} out={n_out} ({n_out/(n_in+n_out+1)*100:.1f}%% filtered)")
                    except Exception as e:
                        if frame_idx % 50 == 0:
                            print(f"[Consistency] Check failed: {e}")

            # ---- Update reference for next frame ----
            if depth_np is not None and not has_nan and c2w_np is not None:
                last_valid_depth = depth_np.copy()
                last_valid_c2w = c2w_np.copy()
                if frame_idx % 30 == 0 or is_pose_jump or is_depth_degraded:
                    ref_depth_for_consistency = depth_np.copy()
                    ref_c2w_for_consistency = c2w_np.copy()

            # ---- Assemble frame data ----
            if world_pts is not None:
                if world_pts.ndim == 3:
                    pc = world_pts.reshape(-1, 3).astype(np.float32)
                else:
                    pc = world_pts.astype(np.float32)
            else:
                pc = np.zeros((0, 3), dtype=np.float32)
                color_np = np.zeros((0, 3), dtype=np.float32)
                conf_np = np.zeros(0, dtype=np.float32)

            # ---- Keyframe gating (relax threshold when pose already collapsed) ----
            accepted = False
            t_delta, r_delta = 0.0, 0.0
            if args.enable_keyframe_gate:
                # If the last accepted keyframe is near the origin (pose
                # already collapsed), use a much smaller threshold so we keep
                # accepting frames instead of refusing every frame once
                # tracking has broken down.
                kf_t = (args.keyframe_trans_thresh
                        if (last_keyframe_c2w is None
                            or float(np.linalg.norm(last_keyframe_c2w[:3, 3])) > 0.1)
                        else args.keyframe_trans_thresh * 0.2)
                kf_r = (args.keyframe_rot_thresh_deg
                        if (last_keyframe_c2w is None
                            or float(np.linalg.norm(last_keyframe_c2w[:3, 3])) > 0.1)
                        else args.keyframe_rot_thresh_deg * 0.3)
                accepted, t_delta, r_delta = _should_accept_keyframe(
                    last_keyframe_c2w, c2w_np,
                    translation_thresh=kf_t,
                    rotation_thresh_deg=kf_r,
                )
            else:
                accepted = c2w_np is not None

            # ---- Apply extra downsample for low-quality frames ----
            extra_ds = 1
            if is_pose_jump:
                extra_ds = max(extra_ds, 4)
            if is_depth_degraded:
                extra_ds = max(extra_ds, 3)
            # Catastrophic pose: DON'T skip -- add to viewer even if pose collapsed
            if accepted:
                final_ds = args.downsample_factor * extra_ds
                pc_ds = pc[::final_ds] if len(pc) > 0 and final_ds > 1 else pc
                color_ds = color_np[::final_ds] if len(color_np) > 0 and final_ds > 1 else color_np
                conf_ds = conf_np[::final_ds] if len(conf_np) > 0 and final_ds > 1 else conf_np
                # Debug: log c2w translation magnitude and translation delta.
                # The "last" reference is the most recently appended viewer
                # frame's c2w (or last_valid_c2w if no viewer frame yet).
                t_norm = float(np.linalg.norm(c2w_np[:3, 3])) if c2w_np is not None else 0.0
                _ref = last_valid_c2w  # updated at line ~1429 for this very frame, so this is the same frame
                # Real cross-frame delta: use the c2w from the *previous*
                # viewer frame, which is what `last_keyframe_c2w` tracks.
                _prev = last_keyframe_c2w if last_keyframe_c2w is not None else None
                if _prev is not None and c2w_np is not None:
                    t_delta_dbg = float(np.linalg.norm(c2w_np[:3, 3] - _prev[:3, 3]))
                    r_dbg = _rotation_angle_deg(_prev, c2w_np)
                else:
                    t_delta_dbg = 0.0
                    r_dbg = 0.0
                if getattr(args, "diag_verbose", False) or frame_idx <= 50 or frame_idx % 5 == 0:
                    print(f"[Pose] f={frame_idx} |t|={t_norm:.4f} dt={t_delta_dbg:.4f} "
                          f"dr={r_dbg:.1f} acc={len(viewer_frames)+1} npc={len(pc_ds)} "
                          f"tx={c2w_np[0,3]:.3f} ty={c2w_np[1,3]:.3f} tz={c2w_np[2,3]:.3f}")
                # Early warning: pose starts collapsing (dt < 0.001 while |t| > 0.5)
                if t_norm > 0.5 and t_delta_dbg < 0.001:
                    print(f"[DIAG-COLLAPSE] f={frame_idx} |t|={t_norm:.4f} dt={t_delta_dbg:.6f} "
                          f"dr={r_dbg:.2f} — pose collapsing! last_keyframe may be stale.")
                frame_data = {
                    "pc": pc_ds,
                    "color": color_ds,
                    "conf": conf_ds,
                    "c2w": c2w_np,
                    "image": frame_rgb,
                }
                viewer_frames.append(frame_data)
                last_keyframe_c2w = c2w_np.copy() if c2w_np is not None else last_keyframe_c2w
                if len(viewer_frames) > args.max_viewer_frames:
                    viewer_frames.pop(0)
                if quality_str and (frame_idx % 20 == 0 or getattr(args, "diag_verbose", False)):
                    print(f"[Quality] frame={frame_idx} {quality_str} extra_ds={extra_ds}x")
            elif args.diag_verbose:
                print(f"[Viewer] skip frame {frame_idx}: "
                      f"(dt={t_delta:.6f}, dr={r_delta:.2f}deg, "
                      f"|t|={float(np.linalg.norm(c2w_np[:3,3])):.4f})")

            # ---- Loop detection ----
            # Only check for loop closure when pose has meaningful scale.
            # When |t| < 0.1 the pose has collapsed to near the origin and
            # any distance is meaningless — skip to avoid false positives.
            if (loop_detector is not None and c2w_np is not None
                    and float(np.linalg.norm(c2w_np[:3, 3])) > 0.1):
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
    g.add_argument("--frame_skip", type=int, default=1,
                   help="Skip every N frames from video file input (default: 1, no skip)")
    g.add_argument("--video_start_frame", type=int, default=0,
                   help="Skip the first N frames of the video. For camera_20260614_034116.mp4 "
                        "the robot starts moving around frame ~420; use e.g. --video_start_frame 400 "
                        "to skip the static intro. Bootstrap now takes the MOST RECENT frames from "
                        "the buffer so it works even if cap_loop has raced ahead. Default: 0")
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
    g.add_argument("--camera_downsample", type=int, default=1,
                   help="Render only 1 in every N camera frustums (reduces clutter). Default 1.")

    g = p.add_argument_group("Stop")
    g.add_argument("--max_frames", type=int, default=None)
    g.add_argument("--enable_loop_detection", action="store_true")
    g.add_argument("--loop_threshold", type=float, default=0.5,
                   help="Distance (meters) for loop closure. Default 0.5. "
                        "Lower = stricter, fewer false positives.")
    g.add_argument("--loop_min_interval", type=int, default=90,
                   help="Min frame distance between two frames considered for loop matching. "
                        "Default 90 (was 30) — at ~1 FPS pipeline, 30 frames only drifts ~0.3m, "
                        "below the 0.5m threshold, causing constant false positives.")
    g.add_argument("--loop_min_history", type=int, default=90,
                   help="Min frames of history before loop detection activates. Default 90.")
    g.add_argument("--loop_warmup_frames", type=int, default=50,
                   help="Skip loop detection for the first N frames after bootstrap. "
                        "Default 50 — gives the trajectory time to stabilize before "
                        "we start looking for loop closures.")
    g.add_argument("--enable_keyframe_gate", action="store_true",
                   help="Only show frames when camera moves significantly")
    g.add_argument("--keyframe_trans_thresh", type=float, default=0.0025,
                   help="Min translation for accepting frame (default: 0.0025)")
    g.add_argument("--keyframe_rot_thresh_deg", type=float, default=6.0,
                   help="Min rotation deg for accepting frame (default: 6.0)")
    g.add_argument("--cap_buf_high", type=int, default=200,
                   help="Hard cap for buf size: cap_loop discards oldest when buf > 2x this (default: 200, total video ~19635 frames)")
    g.add_argument("--cap_read_fps", type=float, default=None,
                   help="Override cap_loop read rate (frames/sec). Default: video FPS (~30). "
                        "Use a lower value (e.g. 5) to slow cap_loop and let inference keep up "
                        "more naturally; use a higher value (e.g. 60) for faster-than-realtime "
                        "processing of a video file.")

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

    g.add_argument("--diag_verbose", action="store_true",
                   help="Print per-frame diagnostic logs (pose/keyframes/cap_loop)")

    args = p.parse_args()

    if args.enable_loop_detection and args.loop_min_history < args.loop_min_interval:
        args.loop_min_history = args.loop_min_interval

    # Clamp loop_warmup_frames to a sensible range
    if args.loop_warmup_frames < 0:
        args.loop_warmup_frames = 0

    if args.video_file and args.video_device != "/dev/video0":
        print("[Warn] Both --video_file and --video_device specified; using --video_file")

    run(args)


if __name__ == "__main__":
    main()

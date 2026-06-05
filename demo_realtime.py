"""LingBot-MAP Real-Time 3D Reconstruction from UVC Camera.

Usage:
    python demo_realtime.py --model_path /path/to/checkpoint.pt --video_device /dev/video0

    # Higher resolution for better quality
    python demo_realtime.py --model_path /path/to/checkpoint.pt --video_device /dev/video0 \
        --image_size 518 --patch_size 14 --num_scale_frames 8 --fps 15

    # Jetson with hardware-accelerated capture (recommended)
    python demo_realtime.py --model_path /path/to/checkpoint.pt --video_device /dev/video0 \
        --use_gstreamer --image_width 640 --image_height 384 --fps 20

    # Specify server IP (for LAN access from another machine)
    python demo_realtime.py --model_path /path/to/checkpoint.pt --video_device /dev/video0 \
        --server_ip 192.168.1.100 --port 8080

Stop recording:
    1. Ctrl+C (manual stop)
    2. GUI: click "Stop & Export" in the viewer
    3. Auto-stop: --max_frames (auto-stop after N frames)
    4. Loop detection: auto-detect loop closure and stop
       (--loop_threshold distance, --loop_skip_frames N)
       When camera returns near a previously visited position,
       loop is detected and reconstruction is stopped + exported.

Export:
    - GLB file: 3D point cloud + camera trajectory (for MeshLab, Blender, etc.)
    - NPZ file: raw predictions (depth, poses, images) for reprocessing
    - MP4 video: rendered flythrough with EDL shading
"""

import argparse
import glob
import math
import os
import socket
import sys
import tempfile
import threading
import time
import signal

# Must be set before `import torch` / any CUDA init.
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

from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import closed_form_inverse_se3_general


# =============================================================================
# Image Preprocessing (mirrors load_and_preprocess_images from demo.py)
# =============================================================================

def _preprocess_frame_for_model(frame_rgb, image_size, patch_size=14):
    """Preprocess a single camera frame for model input.

    Mirrors the crop-mode preprocessing in load_and_preprocess_images():
      1. Resize so width = image_size, height scaled to preserve aspect ratio
         and rounded to nearest multiple of patch_size.
      2. Center-crop the result to image_size pixels tall.

    Args:
        frame_rgb: numpy array [H, W, 3] in [0, 255] BGR (from cv2) or RGB.
        image_size: Target width (and the "canonical" model width).
        patch_size: Patch size for height alignment (must divide height evenly).
    Returns:
        torch.Tensor [3, H, W] in range [0, 1] matching what ToTensor() produces.
    """
    from torchvision import transforms as TF

    # Convert BGR (cv2) to RGB
    if frame_rgb.ndim == 3 and frame_rgb.shape[2] == 3:
        img = Image.fromarray(frame_rgb)
    else:
        img = Image.fromarray(frame_rgb)

    w_orig, h_orig = img.size  # PIL: (width, height)

    # Crop mode (same as demo.py):
    # width = image_size, height scaled preserving aspect ratio, rounded to patch_size
    new_width = image_size
    new_height = round(h_orig * (new_width / w_orig) / patch_size) * patch_size

    img_resized = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
    to_tensor = TF.ToTensor()
    img_t = to_tensor(img_resized)  # [3, H, W], range [0, 1]

    # Center-crop to canonical height
    if new_height > image_size:
        start_y = (new_height - image_size) // 2
        img_t = img_t[:, start_y: start_y + image_size]
    # Note: we do NOT pad here (unlike load_fn which pads in "pad" mode).
    # For realtime camera, cropping is fine; we maintain the exact same H as demo.py.

    return img_t  # [3, H, W], range [0, 1]


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
# Export
# =============================================================================

def export_reconstruction(frames, output_dir, conf_threshold=50.0, downsample_factor=4,
                          export_glb=True, export_npz=True, export_video=False):
    """Export the reconstructed scene to GLB, NPZ, and optionally MP4."""
    os.makedirs(output_dir, exist_ok=True)
    result = {}

    if not frames:
        print("[Export] No frames to export")
        return result

    print(f"[Export] Processing {len(frames)} frames...")

    world_points = []
    all_colors = []
    all_conf = []
    all_c2w = []
    images = []

    for i, (pc, color, conf, c2w, intrinsic, image) in enumerate(frames):
        if pc is None or len(pc) == 0:
            continue
        conf_arr = np.asarray(conf).flatten() if conf is not None and len(conf) > 0 else np.ones(len(pc))
        pts_arr = np.asarray(pc).reshape(-1, 3)
        cols_arr = np.asarray(color).reshape(-1, 3) if color is not None and len(color) > 0 else np.zeros((len(pts_arr), 3))

        if conf_threshold < 100 and len(conf_arr) > 0:
            threshold_val = np.percentile(conf_arr, conf_threshold)
            mask = (conf_arr >= threshold_val) & (conf_arr > 1e-5)
        else:
            mask = conf_arr > 1e-5 if len(conf_arr) > 0 else np.ones(len(pts_arr), dtype=bool)

        pts = pts_arr[mask]
        cols = cols_arr[mask] if len(cols_arr) == len(pts_arr) else cols_arr
        cols = cols[mask] if len(cols) == len(pts_arr) else cols
        if len(pts) == 0:
            continue
        if downsample_factor > 1:
            idx = np.arange(0, len(pts), downsample_factor)
            pts = pts[idx]
            cols = cols[idx] if len(cols) >= len(idx) else cols
        world_points.append(pts)
        all_colors.append(cols)
        all_conf.append(conf_arr[mask] if len(conf_arr) == len(pts_arr) else None)
        all_c2w.append(c2w)
        images.append(image)

    if not world_points:
        print("[Export] No valid points after filtering")
        return result

    all_points = np.concatenate(world_points)
    try:
        all_colors = np.concatenate(all_colors)
    except ValueError:
        all_colors = np.zeros((len(all_points), 3), dtype=np.uint8)

    scene_center = all_points.mean(axis=0)
    all_points_centered = all_points - scene_center
    scale = np.percentile(np.linalg.norm(all_points_centered, axis=1), 95) + 1e-6

    if export_glb:
        print("[Export] Building GLB...")
        t_color = (all_colors.astype(np.float32) / 255.0) if all_colors.max() > 1 else all_colors.astype(np.float32)
        pc_mesh = trimesh.PointCloud(vertices=all_points_centered / scale, colors=t_color)
        scene = trimesh.Scene()
        scene.add_geometry(pc_mesh, geom_name="point_cloud")

        for i, c2w in enumerate(all_c2w):
            if c2w is None:
                continue
            c2w_vis = c2w.copy()
            c2w_vis[:3, 3] = (c2w[:3, 3] - scene_center) / scale
            frustum = trimesh.creation.camera_frustum(c2w_vis, 0.5, 0.3)
            scene.add_geometry(frustum, geom_name=f"camera_{i}", transform=None)

        cam_positions = [c2w[:3, 3] for c2w in all_c2w if c2w is not None]
        if len(cam_positions) >= 2:
            import matplotlib.cm as cm
            traj_pts = np.array(cam_positions)
            traj_pts_centered = (traj_pts - scene_center) / scale
            traj_pts_centered = traj_pts_centered.astype(np.float64)
            colormap = cm.get_cmap('gist_rainbow')
            num_cam = len(cam_positions)
            for i in range(len(traj_pts_centered) - 1):
                p0 = traj_pts_centered[i]
                p1 = traj_pts_centered[i + 1]
                seg_len = float(np.linalg.norm(p1 - p0))
                if seg_len < 1e-8:
                    continue
                direction = (p1 - p0) / seg_len
                mid = (p0 + p1) * 0.5
                cyl = trimesh.creation.cylinder(radius=0.003, height=seg_len, sections=6)
                z_axis = np.array([0.0, 0.0, 1.0])
                v = np.cross(z_axis, direction)
                c = float(np.dot(z_axis, direction))
                if np.linalg.norm(v) < 1e-8:
                    rot = np.eye(3) if c > 0 else np.diag([1, -1, -1])
                else:
                    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                    rot = np.eye(3) + vx + vx @ vx / (1.0 + c)
                transform = np.eye(4)
                transform[:3, :3] = rot
                transform[:3, 3] = mid
                cyl.apply_transform(transform)
                t_color = (i + 0.5) / max(num_cam - 1, 1)
                rgba = colormap(t_color)
                color_rgb = tuple(int(255 * x) for x in rgba[:3])
                cyl.visual.face_colors[:, :3] = color_rgb
                scene.add_geometry(cyl)

        glb_path = os.path.join(output_dir, "scene.glb")
        scene.glb(glb_path)
        result["glb"] = glb_path
        print(f"[Export] GLB saved: {glb_path}")

    if export_npz:
        print("[Export] Saving NPZ...")
        npz_path = os.path.join(output_dir, "reconstruction.npz")
        np.savez(npz_path,
                 points=all_points,
                 colors=all_colors,
                 c2w_arr=np.stack(all_c2w) if all_c2w else np.zeros((0, 4, 4)),
                 images=images)
        result["npz"] = npz_path
        print(f"[Export] NPZ saved: {npz_path}")

    if export_video:
        print("[Export] Rendering MP4 (requires open3d)...")
        mp4_path = os.path.join(output_dir, "reconstruction.mp4")
        try:
            _export_video_fallback(all_points, all_colors, all_c2w, mp4_path)
        except Exception as e:
            print(f"[Export] MP4 export failed: {e}")
        else:
            result["mp4"] = mp4_path

    print(f"[Export] Done. Output directory: {output_dir}")
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
# Camera Capture
# =============================================================================

def _detect_jetson():
    try:
        with open("/proc/device-tree/model", "r") as f:
            return "nvidia" in f.read().lower() or "jetson" in f.read().lower()
    except Exception:
        return False


class UVCCapture:
    """High-performance UVC camera capture with GStreamer or V4L2 support."""

    def __init__(self, device="/dev/video0", width=640, height=384, fps=20,
                 pixel_format="MJPG", use_gstreamer=False, capture_fps=None, verbose=True):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.pixel_format = pixel_format
        self.use_gstreamer = use_gstreamer
        self.capture_fps = capture_fps if capture_fps is not None else fps
        self.verbose = verbose
        self._capture = None
        self.is_jetson = _detect_jetson()
        self._last_return_time = 0
        self._interval_sec = 1.0 / self.capture_fps

        if verbose:
            if self.is_jetson:
                print(f"[Camera] Jetson detected")
            print(f"[Camera] Opening: {device} @ {width}x{height} {fps}fps ({pixel_format})")

        self._open()

    def _open(self):
        opened = False

        if self.use_gstreamer:
            if self.is_jetson:
                pipeline = (
                    f"v4l2src device={self.device} ! "
                    f"image/jpeg,framerate={self.capture_fps}/1 ! "
                    f"jpegdec ! videoconvert ! "
                    f"video/x-raw,format=RGB ! "
                    f"appsink emit-signals=true drop=true sync=false"
                )
                self._capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if self._capture.isOpened():
                    print("[Camera] GStreamer opened successfully")
                    opened = True
                else:
                    print("[Camera] GStreamer failed, falling back to V4L2")

        if not opened:
            self._capture = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
            if not self._capture.isOpened():
                raise RuntimeError(f"Cannot open camera device: {self.device}")

            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if self.pixel_format == "MJPG":
                self._capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            elif self.pixel_format == "YUYV":
                self._capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
            self._capture.set(cv2.CAP_PROP_FPS, self.capture_fps)
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            actual_w = self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self._capture.get(cv2.CAP_PROP_FPS)
            if self.verbose:
                print(f"[Camera] V4L2 opened: {int(actual_w)}x{int(actual_h)} @ {actual_fps:.1f}fps")

    def read_throttled(self):
        """Read a frame, throttled to target_fps."""
        now = time.time()
        if now - self._last_return_time < self._interval_sec:
            return False, None
        ret, frame = self._capture.read()
        if not ret:
            return False, None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._last_return_time = now
        return True, frame_rgb

    def release(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None


# =============================================================================
# Model Loading (mirrors demo.py's load_model)
# =============================================================================

def load_model(args, device):
    """Load GCTStream model from checkpoint. Mirrors demo.py's load_model exactly."""
    from lingbot_map.models.gct_stream import GCTStream

    print("Building model...")
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
        print(f"Loading checkpoint: {args.model_path}")
        ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
        print("  Checkpoint loaded.")

    return model.to(device).eval()


# =============================================================================
# Prediction Processing (mirrors demo.py's postprocess + prepare_for_visualization)
# =============================================================================

_BATCHED_NDIMS = {
    "pose_enc": 3,
    "depth": 5,
    "depth_conf": 4,
    "world_points": 5,
    "world_points_conf": 4,
    "extrinsic": 4,
    "intrinsic": 4,
    "images": 5,
}


def _squeeze_single_batch(key, value):
    """Drop the leading batch dimension for single-sequence demo outputs."""
    batched_ndim = _BATCHED_NDIMS.get(key)
    if batched_ndim is None or not hasattr(value, "ndim"):
        return value
    if value.ndim == batched_ndim and value.shape[0] == 1:
        return value[0]
    return value


def _select_last_frame(key, value):
    """Take the last frame from sequence-like prediction tensors/arrays."""
    batched_ndim = _BATCHED_NDIMS.get(key)
    if batched_ndim is None or not hasattr(value, "ndim"):
        return value

    value = _squeeze_single_batch(key, value)
    if not hasattr(value, "ndim"):
        return value

    # Sequence-like outputs keep time on the leading dimension after the
    # single-batch squeeze, e.g. [S, H, W, 3] or [S, 9].
    if value.ndim == batched_ndim - 1 and value.shape[0] > 1:
        return value[-1]
    return value


def _select_frame_index(key, value, index):
    """Take a specific frame from sequence-like prediction tensors/arrays."""
    batched_ndim = _BATCHED_NDIMS.get(key)
    if batched_ndim is None or not hasattr(value, "ndim"):
        return value

    value = _squeeze_single_batch(key, value)
    if not hasattr(value, "ndim"):
        return value

    if value.ndim == batched_ndim - 1:
        if value.shape[0] == 0:
            return value
        index = max(0, min(int(index), value.shape[0] - 1))
        return value[index]
    return value


def postprocess(predictions, images):
    """Convert pose encoding to extrinsics (c2w) and move to CPU."""
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])

    extrinsic_4x4 = torch.zeros((*extrinsic.shape[:-2], 4, 4), device=extrinsic.device, dtype=extrinsic.dtype)
    extrinsic_4x4[..., :3, :4] = extrinsic
    extrinsic_4x4[..., 3, 3] = 1.0
    extrinsic_4x4 = closed_form_inverse_se3_general(extrinsic_4x4)
    extrinsic = extrinsic_4x4[..., :3, :4]

    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    predictions.pop("pose_enc_list", None)

    print("Moving results to CPU...")
    for k in list(predictions.keys()):
        if isinstance(predictions[k], torch.Tensor):
            predictions[k] = _squeeze_single_batch(
                k, predictions[k].to("cpu", non_blocking=True)
            )
    images_cpu = images.to("cpu", non_blocking=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return predictions, images_cpu


def prepare_for_visualization(predictions, images=None):
    """Convert predictions to the unbatched NumPy format used by vis code."""
    vis_predictions = {}
    for k, v in predictions.items():
        if isinstance(v, torch.Tensor):
            v = _squeeze_single_batch(k, v.detach().cpu())
            vis_predictions[k] = v.numpy()
        elif isinstance(v, np.ndarray):
            vis_predictions[k] = _squeeze_single_batch(k, v)
        else:
            vis_predictions[k] = v

    if images is None:
        images = predictions.get("images")

    if isinstance(images, torch.Tensor):
        images = images.detach().cpu()
    if isinstance(images, np.ndarray):
        images = _squeeze_single_batch("images", images)
    elif isinstance(images, torch.Tensor):
        images = _squeeze_single_batch("images", images).numpy()

    if isinstance(images, torch.Tensor):
        images = images.numpy()

    if images is not None:
        vis_predictions["images"] = images

    return vis_predictions


# =============================================================================
# Viser Viewer (mirrors demo.py's PointCloudViewer + realtime front-end)
# =============================================================================

def start_viewer(host="0.0.0.0", port=8080, max_viewer_frames=300,
                 initial_conf_threshold=10.0, initial_downsample_factor=1,
                 initial_point_size=0.003):
    """Start the real-time viser viewer in a background thread.

    Shows:
      - 3D point cloud with cumulative reconstruction
      - Camera frustums for each frame
      - Camera trajectory line
      - Live camera feed thumbnail
      - Confidence/Downsample/PointSize sliders
    """
    state = {
        "frames": [],
        "running": True,
        "stop_requested": False,
        "conf_threshold": float(initial_conf_threshold),
        "downsample_factor": int(initial_downsample_factor),
        "point_size": float(initial_point_size),
        "max_viewer_frames": max_viewer_frames,
        "show_trajectory": True,
        "show_camera_feed": True,
    }

    global_pc_handle = [None]
    MAX_PREALLOC = 50
    cam_handles = []
    for i in range(MAX_PREALLOC):
        cam_handles.append([None])
    trajectory_line_handle = [None]
    actual_port = [None]

    def viewer_thread_fn():
        import viser
        import numpy as np

        # Find an available port if the requested one is busy
        import socket
        actual_port_val = port
        for candidate in [port, port + 1, port + 2]:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind((host if host != "0.0.0.0" else "", candidate))
                actual_port_val = candidate
                break
            except OSError:
                continue
        actual_port[0] = actual_port_val

        print(f"[Viewer] Starting viser server on {host}:{actual_port_val}...")
        server = viser.ViserServer(host=host, port=actual_port_val, _local_only=False)
        print(f"[Viewer] Viser server started successfully")

        print(f"[Viewer] Open in browser:")
        print(f"  Local:   http://127.0.0.1:{actual_port_val}")
        if host == "0.0.0.0":
            try:
                for info in socket.getaddrinfo(socket.gethostname(), actual_port_val, socket.AF_INET):
                    lan_ip = info[4][0]
                    if not lan_ip.startswith("127."):
                        print(f"  LAN:     http://{lan_ip}:{actual_port_val}")
                        break
            except Exception:
                pass

        try:
            server.scene.configure_default_lights(enabled=True, cast_shadow=False)
        except Exception:
            pass
        try:
            server.scene.add_frame("/world_axes", show_axes=True, axes_length=0.5)
        except Exception:
            pass
        try:
            server.scene.add_grid("/ground_grid", width=10.0, height=10.0, cell_color=(80, 80, 80))
        except Exception:
            pass

        try:
            server.gui.configure_theme(control_layout="collapsible")
        except Exception:
            pass

        # Confidence slider (percentile, 0-100)
        conf_slider = server.gui.add_slider("Confidence Percent", 0.0, 100.0, 0.5, float(initial_conf_threshold))
        conf_slider.on_update(lambda val: state.update({"conf_threshold": float(val.value)}))

        # Downsample slider
        ds_slider = server.gui.add_slider("Downsample", 1, 32, 1, int(initial_downsample_factor))
        ds_slider.on_update(lambda val: state.update({"downsample_factor": int(val.target.value)}))

        # Point size slider
        pt_slider = server.gui.add_slider("Point Size", 0.000001, 0.1, 0.000001, float(initial_point_size))
        pt_slider.on_update(lambda val: state.update({"point_size": float(val.target.value)}))

        show_cam_toggle = server.gui.add_checkbox("Show Camera Feed", True)
        show_cam_toggle.on_update(lambda val: state.update({"show_camera_feed": val.target.value}))

        show_trajectory_toggle = server.gui.add_checkbox("Show Trajectory", True)
        show_trajectory_toggle.on_update(lambda val: state.update({"show_trajectory": val.target.value}))

        server.gui.add_button("Stop & Export").on_click(
            lambda _: state.update({"stop_requested": True}))

        cam_feed_handle = [None]

        @server.on_client_disconnect
        def _on_disconnect(_client):
            pass

        def _compute_center_scale(frames):
            if not frames:
                return np.zeros(3), 1.0
            all_pts = []
            for pc, _, _, _, _, _ in frames:
                if pc is not None and len(pc) > 0:
                    all_pts.append(pc)
            if not all_pts:
                return np.zeros(3), 1.0
            pts = np.concatenate(all_pts)
            center = pts.mean(axis=0)
            scale = np.percentile(np.linalg.norm(pts - center, axis=1), 95) + 1e-6
            return center, scale

        def _update_scene():
            frames = state["frames"]
            if not frames:
                return

            center, scale = _compute_center_scale(frames)
            conf_percentile = state["conf_threshold"]
            downsample = state["downsample_factor"]
            pt_size = state["point_size"]
            show_cam = state.get("show_camera_feed", True)

            all_pts_list = []
            all_cols_list = []
            all_conf_list = []
            cam_positions = []

            for pc, color, conf, c2w, intrinsic, image in frames:
                if pc is None or len(pc) == 0:
                    continue
                all_pts_list.append(pc)
                all_cols_list.append(color)
                if conf is not None and len(conf) > 0:
                    all_conf_list.append(conf.flatten())
                else:
                    all_conf_list.append(np.ones(len(pc)))

                if c2w is not None:
                    t = c2w[:3, 3]
                    if isinstance(t, np.ndarray):
                        cam_positions.append(t.copy())

            if not all_pts_list:
                return

            all_conf_flat = np.concatenate(all_conf_list)
            threshold_val = np.percentile(all_conf_flat, conf_percentile)

            final_pts = []
            final_cols = []
            for pts, cols, conf_arr in zip(all_pts_list, all_cols_list, all_conf_list):
                if pts.ndim == 1:
                    pts = pts.reshape(-1, 3)
                pts_n = len(pts)
                if pts_n == 0:
                    continue

                if conf_arr.shape[0] != pts_n:
                    conf_arr = np.resize(np.asarray(conf_arr).flatten(), pts_n)
                mask = (conf_arr >= threshold_val) & (conf_arr > 1e-5)

                pts_f = pts[mask]

                if cols is None or len(cols) == 0:
                    cols_f = np.zeros((len(pts_f), 3), dtype=np.float32)
                elif len(cols) != pts_n:
                    cols_arr = np.asarray(cols).reshape(-1, 3)
                    if len(cols_arr) != pts_n:
                        cols_arr = np.resize(cols_arr, (pts_n, 3))
                    cols_f = cols_arr[mask]
                else:
                    cols_f = np.asarray(cols).reshape(-1, 3)[mask]

                if downsample > 1:
                    idx = np.arange(0, len(pts_f), downsample)
                    pts_f = pts_f[idx]
                    if len(cols_f) >= len(idx):
                        cols_f = cols_f[idx]
                if len(pts_f) > 0:
                    final_pts.append(pts_f)
                    final_cols.append(cols_f)

            global_pts = np.concatenate(final_pts) if final_pts else np.zeros((0, 3), dtype=np.float32)
            global_cols = np.concatenate(final_cols) if final_cols else np.zeros((0, 3), dtype=np.float32)

            if len(global_pts) == 0:
                return

            global_pts_scaled = (global_pts - center) / scale
            if global_pts_scaled.dtype != np.float32:
                global_pts_scaled = global_pts_scaled.astype(np.float32)
            if global_cols.max() > 1:
                global_cols = global_cols.astype(np.float32) / 255.0
            else:
                global_cols = global_cols.astype(np.float32)

            with server.atomic():
                try:
                    if global_pc_handle[0] is None:
                        global_pc_handle[0] = server.scene.add_point_cloud(
                            "_global_pc",
                            points=global_pts_scaled,
                            colors=global_cols,
                            point_size=pt_size,
                            point_shading="flat",
                            visible=True,
                        )
                    else:
                        global_pc_handle[0].points = global_pts_scaled
                        global_pc_handle[0].colors = global_cols
                        global_pc_handle[0].point_size = pt_size
                        global_pc_handle[0].visible = True
                except Exception as e:
                    print(f"[Viewer] point cloud update error: {e}")

                visible_frames = frames[-MAX_PREALLOC:]
                n_visible = len(visible_frames)
                frustum_scale = 0.05

                for i, (pc, color, conf, c2w, intrinsic, image) in enumerate(visible_frames):
                    cam_handle = cam_handles[i][0]
                    if c2w is None:
                        if cam_handle is not None:
                            try:
                                cam_handle.visible = False
                            except Exception:
                                pass
                        continue

                    try:
                        c2w_vis = c2w.copy()
                        t_vis = (c2w_vis[:3, 3] - center) / scale
                        wxyz = viser.transforms.SO3.from_matrix(c2w_vis[:3, :3]).wxyz
                        position = tuple(float(x) for x in t_vis)
                        if cam_handle is None:
                            cam_handles[i][0] = server.scene.add_camera_frustum(
                                f"_cam_{i}",
                                fov=0.8,
                                aspect=1.4,
                                scale=frustum_scale,
                                color=(0, 1, 0),
                                wxyz=wxyz,
                                position=position,
                                variant="wireframe",
                                visible=True,
                            )
                        else:
                            cam_handles[i][0].wxyz = wxyz
                            cam_handles[i][0].position = position
                            cam_handles[i][0].scale = frustum_scale
                            cam_handles[i][0].visible = True
                    except Exception:
                        pass

                for i in range(n_visible, MAX_PREALLOC):
                    if cam_handles[i][0] is not None:
                        try:
                            cam_handles[i][0].visible = False
                        except Exception:
                            pass

                show_traj = state.get("show_trajectory", True)
                if show_traj and len(cam_positions) >= 2:
                    traj_pts = np.array(cam_positions)
                    traj_pts_scaled = (traj_pts - center) / scale
                    traj_pts_scaled = traj_pts_scaled.astype(np.float32)
                    try:
                        n_pts = len(traj_pts_scaled)
                        t = np.linspace(0.0, 1.0, max(n_pts - 1, 1))
                        cmap = np.zeros((max(n_pts - 1, 1), 3), dtype=np.float32)
                        cmap[:, 0] = np.clip(np.abs(t * 6.0 - 3.0) - 1.0, 0.0, 1.0)
                        cmap[:, 1] = np.clip(2.0 - np.abs(t * 6.0 - 2.0), 0.0, 1.0)
                        cmap[:, 2] = np.clip(2.0 - np.abs(t * 6.0 - 4.0), 0.0, 1.0)
                        segs = np.array([[i, i + 1] for i in range(n_pts - 1)], dtype=np.int32)
                        if trajectory_line_handle[0] is None:
                            trajectory_line_handle[0] = server.scene.add_line_set(
                                "_trajectory",
                                points=traj_pts_scaled,
                                segments=segs,
                                colors=cmap,
                                line_width=2.0,
                                visible=True,
                            )
                        else:
                            trajectory_line_handle[0].points = traj_pts_scaled
                            trajectory_line_handle[0].segments = segs
                            trajectory_line_handle[0].colors = cmap
                            trajectory_line_handle[0].line_width = 2.0
                            trajectory_line_handle[0].visible = True
                    except Exception:
                        pass
                elif trajectory_line_handle[0] is not None:
                    try:
                        trajectory_line_handle[0].visible = False
                    except Exception:
                        pass

                # Update camera feed thumbnail
                latest_frame = None
                for _, _, _, _, _, image in reversed(frames):
                    if image is not None:
                        latest_frame = image
                        break

                if show_cam and latest_frame is not None:
                    try:
                        if hasattr(latest_frame, 'cpu'):
                            img_np = latest_frame.cpu().numpy()
                        elif hasattr(latest_frame, 'numpy'):
                            img_np = latest_frame.numpy()
                        else:
                            img_np = np.asarray(latest_frame)

                        if img_np.dtype == np.float32 or img_np.dtype == np.float64:
                            if img_np.max() <= 1.0:
                                img_disp = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
                            else:
                                img_disp = np.clip(img_np, 0, 255).astype(np.uint8)
                        elif img_np.dtype == np.uint8:
                            img_disp = img_np
                        else:
                            img_disp = img_disp.astype(np.uint8)

                        if img_disp.ndim == 3 and img_disp.shape[-1] == 3:
                            img_rgb = img_disp
                        else:
                            img_rgb = img_disp

                        if cam_feed_handle[0] is None:
                            cam_feed_handle[0] = server.gui.add_image(
                                image=img_rgb,
                                label="Camera Feed",
                                visible=True,
                            )
                        else:
                            try:
                                cam_feed_handle[0].image = img_rgb
                            except Exception:
                                cam_feed_handle[0] = None
                    except Exception as e:
                        if cam_feed_handle[0] is None:
                            print(f"[Viewer] camera feed error: {e}")

        last_report = [0]

        while state["running"]:
            try:
                _update_scene()
            except Exception as e:
                print(f"[Viewer] _update_scene error: {e}")
                import traceback
                traceback.print_exc()

            frames = state["frames"]
            n = len(frames)
            now = time.time()
            if n != last_report[0] and now - last_report[0] > 5.0:
                print(f"[Viewer] {n} frames accumulated")
                last_report[0] = now

            time.sleep(0.25)

    thread = threading.Thread(target=viewer_thread_fn, daemon=True)
    thread.start()
    return thread, state, actual_port


# =============================================================================
# Main Real-Time Inference Loop
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
    # Defensive: ensure both are valid [4, 4] matrices
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
            return m[-1]  # take last frame
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


def run_realtime(args):
    device = torch.device(args.device if hasattr(args, "device") else
                          "cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"[Device] Using {device}")
        print(f"[Device] cuDNN benchmark enabled")
        if torch.backends.cuda.matmul.allow_tf32:
            print(f"[Device] TF32 tensor cores enabled")

    # ---- Load Model (same as demo.py) ----
    model = load_model(args, device)

    # Determine dtype (same as demo.py)
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    # Cast aggregator to dtype (same as demo.py)
    if dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
        print(f"Casting aggregator to {dtype} (heads kept in fp32)")
        model.aggregator = model.aggregator.to(dtype=dtype)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Compute canvas height (matches demo.py crop mode) ----
    raw_h = round(args.image_size * 360 / 640)
    canvas_h = round(raw_h / args.patch_size) * args.patch_size
    canvas_w = args.image_size
    print(f"[Preprocess] Canvas: {canvas_w}x{canvas_h} (image_size={args.image_size})")

    # ---- Model warmup (same as demo.py's streaming warmup) ----
    scale_frames = max(1, args.num_scale_frames)
    print(f"[Model] Warming up with {canvas_w}x{canvas_h}, num_scale_frames={scale_frames}...")
    dummy_scale = torch.zeros((1, scale_frames, 3, canvas_h, canvas_w),
                               dtype=torch.float32, device=device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        model.inference_streaming(dummy_scale, num_scale_frames=scale_frames)
    model.clean_kv_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Camera capture ----
    frame_buffer = []
    frame_buffer_lock = threading.Lock()
    stop_capture = threading.Event()
    capture_error = [None]

    def capture_loop():
        try:
            cap = UVCCapture(
                device=args.video_device,
                width=args.image_width,
                height=args.image_height,
                fps=args.fps,
                pixel_format=args.pixel_format,
                use_gstreamer=args.use_gstreamer,
                capture_fps=args.capture_fps,
            )

            while not stop_capture.is_set():
                ret, frame_rgb = cap.read_throttled()
                if ret:
                    with frame_buffer_lock:
                        frame_buffer.append(frame_rgb.copy())
                        # Drop oldest if too many pending
                        while len(frame_buffer) > 10:
                            frame_buffer.pop(0)
                else:
                    time.sleep(0.005)
            cap.release()
        except Exception as e:
            capture_error[0] = e

    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()

    # ---- Loop detection ----
    loop_detector = LoopDetector(
        threshold=args.loop_threshold,
        min_interval=args.loop_min_interval,
        min_history=args.loop_min_history,
    ) if args.enable_loop_detection else None

    # ---- Viewer ----
    viewer_thread = None
    viewer_state = None
    viewer_actual_port = [None]
    try:
        viewer_thread, viewer_state, viewer_actual_port[0] = start_viewer(
            host=args.server_ip,
            port=args.port,
            max_viewer_frames=args.max_viewer_frames,
            initial_conf_threshold=args.conf_threshold,
            initial_downsample_factor=args.downsample_factor,
            initial_point_size=args.point_size,
        )
    except Exception as e:
        import traceback
        print(f"[Viewer] Viser viewer failed: {e}")
        traceback.print_exc()

    # ---- Config summary ----
    print("=" * 60)
    print("LingBot-MAP Real-Time 3D Reconstruction")
    print("=" * 60)
    print(f"  Camera device : {args.video_device}")
    actual_port_str = str(viewer_actual_port[0]) if viewer_actual_port[0] is not None else str(args.port)
    print(f"  Viewer port  : {actual_port_str}")
    print(f"  Image size    : {args.image_width}x{args.image_height}")
    print(f"  Canvas        : {canvas_w}x{canvas_h}")
    print(f"  Target FPS   : {args.fps}")
    print(f"  Pixel format : {args.pixel_format}")
    print(f"  Model        : {args.model_path}")
    print(f"  Camera iters : {args.camera_num_iterations}")
    print(f"  Scale frames : {args.num_scale_frames}")
    if args.enable_keyframe_gate:
        print(f"  Keyframe gate: enabled (trans>{args.keyframe_translation_thresh}, rot>{args.keyframe_rotation_thresh_deg}deg)")
    else:
        print(f"  Keyframe gate: disabled (every frame shown)")
    if args.enable_loop_detection:
        print(f"  Loop detect  : enabled (threshold={args.loop_threshold})")
    if args.max_frames:
        print(f"  Auto-stop    : after {args.max_frames} frames")
    print("=" * 60)

    print(f"[Inference] Starting real-time loop. Press Ctrl+C to stop.")

    viewer_frames = []
    frame_idx = 0
    last_infer_time = time.perf_counter()
    fps_window = []
    overall_fps = 0.0
    stop_reason = "user"
    total_elapsed = 0.0
    last_keyframe_c2w = None
    stream_initialized = False
    bootstrap_tensors = []
    fps_report_time = time.time()
    fps_frame_count = 0
    frames_since_reset = 0  # frames since last KV cache reset
    kv_reset_interval = getattr(args, 'kv_reset_interval', 200)  # reset before hitting max_frame_num

    try:
        while True:
            # Check stop conditions
            if args.max_frames and frame_idx >= args.max_frames:
                stop_reason = "max_frames"
                break
            if viewer_state and viewer_state.get("stop_requested", False):
                stop_reason = "gui"
                break
            if capture_error[0]:
                raise capture_error[0]

            # Get latest frame from buffer
            with frame_buffer_lock:
                if not frame_buffer:
                    time.sleep(0.001)
                    continue
                frame_rgb = frame_buffer[-1]
                frame_buffer.clear()

            # Preprocess: convert to model input format (matches demo.py crop mode)
            img_tensor = _preprocess_frame_for_model(frame_rgb, args.image_size, args.patch_size)
            img_tensor = img_tensor.to(device)  # [3, H, W], range [0, 1]

            # Inference
            t0 = time.perf_counter()

            if not stream_initialized:
                bootstrap_tensors.append(img_tensor)
                if len(bootstrap_tensors) < scale_frames:
                    print(f"[StreamInit] Buffering scale frames: {len(bootstrap_tensors)}/{scale_frames}")
                    continue

                # Bootstrap KV cache with scale frames (same as demo.py Phase 1)
                init_batch = torch.stack(bootstrap_tensors, dim=0).unsqueeze(0)  # [1, S, 3, H, W]
                model.clean_kv_cache()
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                    predictions = model.forward(
                        init_batch,
                        num_frame_for_scale=scale_frames,
                        num_frame_per_block=scale_frames,
                        causal_inference=True,
                    )
                stream_initialized = True
                bootstrap_tensors.clear()
                print(f"[StreamInit] Streaming KV cache initialized with {scale_frames} scale frames")
            else:
                # Streaming forward (same as demo.py Phase 2)
                frame_batch = img_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, H, W]

                # Periodic KV cache reset to prevent 3D RoPE position index overflow.
                # Each reset keeps the already-accumulated viewer point cloud intact
                # (it's stored separately in viewer_frames). We only reset the model's
                # internal KV cache so 3D RoPE frame indices stay in bounds.
                if frames_since_reset >= kv_reset_interval:
                    print(f"[KVReset] Resetting KV cache at frame {frame_idx} "
                          f"(frames_since_reset={frames_since_reset})")
                    model.clean_kv_cache()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # Re-bootstrap: feed current frame as scale frame to rebuild KV context
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                        predictions = model.forward(
                            frame_batch,
                            num_frame_for_scale=1,
                            num_frame_per_block=1,
                            causal_inference=True,
                        )
                    frames_since_reset = 0
                    print(f"[KVReset] KV cache re-initialized")
                else:
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                        predictions = model.forward(
                            frame_batch,
                            num_frame_for_scale=scale_frames,
                            num_frame_per_block=1,
                            causal_inference=True,
                        )
                    frames_since_reset += 1

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_time = time.perf_counter() - t0

            # Move predictions to CPU immediately
            predictions_cpu = {}
            for k, v in predictions.items():
                if isinstance(v, torch.Tensor):
                    predictions_cpu[k] = v.detach().cpu()
                else:
                    predictions_cpu[k] = v

            # ---- DIAGNOSTICS: print model outputs for first few frames ----
            if frame_idx < 5:
                pose_enc = predictions_cpu.get("pose_enc")
                world_pts = predictions_cpu.get("world_points")
                depth = predictions_cpu.get("depth")
                depth_conf = predictions_cpu.get("depth_conf")
                wp_conf = predictions_cpu.get("world_points_conf")
                print(f"[DEBUG frame {frame_idx}] infer={infer_time*1000:.1f}ms | "
                      f"pose_enc={pose_enc.shape if pose_enc is not None else None} | "
                      f"world_points={world_pts.shape if world_pts is not None else None} | "
                      f"world_points_conf={wp_conf.shape if wp_conf is not None else None} | "
                      f"depth={depth.shape if depth is not None else None} | "
                      f"depth_conf={depth_conf.shape if depth_conf is not None else None} | "
                      f"images={predictions_cpu.get('images').shape if predictions_cpu.get('images') is not None else None}")

            def _build_viewer_frame(frame_index, frame_rgb_for_color):
                per_frame = {}
                for key, value in predictions_cpu.items():
                    per_frame[key] = _select_frame_index(key, value, frame_index)

                pose_enc = per_frame.get("pose_enc")
                c2w = None
                intrinsic = None
                if pose_enc is not None:
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, (canvas_h, canvas_w))
                    extrinsic_4x4 = torch.zeros((*extrinsic.shape[:-2], 4, 4), dtype=extrinsic.dtype)
                    extrinsic_4x4[..., :3, :4] = extrinsic
                    extrinsic_4x4[..., 3, 3] = 1.0
                    extrinsic_4x4 = closed_form_inverse_se3_general(extrinsic_4x4)
                    c2w = extrinsic_4x4.squeeze().numpy()

                world_points = per_frame.get("world_points")
                world_points_conf = per_frame.get("world_points_conf")
                depth = per_frame.get("depth")
                depth_conf = per_frame.get("depth_conf")

                pc = None
                conf = None

                if world_points is not None:
                    wp = world_points.squeeze().numpy()
                    if wp.ndim == 3:
                        H_wp, W_wp, _ = wp.shape
                        pc = wp.reshape(-1, 3).astype(np.float32)
                    else:
                        H_wp, W_wp = 0, 0
                        pc = np.zeros((0, 3), dtype=np.float32)

                    if world_points_conf is not None:
                        conf_src = world_points_conf.squeeze().numpy()
                        if conf_src.ndim == 2:
                            conf = conf_src.flatten().astype(np.float32)
                        else:
                            conf = np.ones(len(pc), dtype=np.float32)
                    else:
                        conf = np.ones(len(pc), dtype=np.float32)

                    if H_wp > 0 and W_wp > 0 and pc is not None and len(pc) > 0:
                        frame_resized = cv2.resize(frame_rgb_for_color.astype(np.float32), (W_wp, H_wp)) / 255.0
                        color = frame_resized.reshape(-1, 3).astype(np.float32)
                    else:
                        color = np.zeros((len(pc), 3), dtype=np.float32) if pc is not None else np.zeros((0, 3), dtype=np.float32)
                elif depth is not None:
                    d = depth.squeeze().numpy()
                    if d.ndim == 3:
                        d = d[:, :, 0]
                    H_d, W_d = d.shape

                    if intrinsic is not None:
                        intr_np = intrinsic.squeeze().numpy()
                    else:
                        intr_np = np.array([[W_d * 0.8, 0, W_d * 0.5],
                                            [0, H_d * 0.8, H_d * 0.5],
                                            [0, 0, 1]], dtype=np.float32)

                    fx, fy = intr_np[0, 0], intr_np[1, 1]
                    cx, cy = intr_np[0, 2], intr_np[1, 2]
                    u_coords = np.arange(W_d, dtype=np.float32)
                    v_coords = np.arange(H_d, dtype=np.float32)
                    vv, uu = np.meshgrid(v_coords, u_coords, indexing='ij')
                    cam_coords = np.stack([
                        (uu - cx) / fx * d,
                        (vv - cy) / fy * d,
                        d,
                    ], axis=-1)

                    if c2w is not None:
                        cam_flat = cam_coords.reshape(-1, 3)
                        ones = np.ones((cam_flat.shape[0], 1))
                        cam_h = np.concatenate([cam_flat, ones], axis=1)
                        world_h = cam_h @ c2w.T
                        pc = world_h[:, :3].astype(np.float32)
                    else:
                        pc = np.zeros((H_d * W_d, 3), dtype=np.float32)

                    frame_resized = cv2.resize(frame_rgb_for_color.astype(np.float32), (W_d, H_d)) / 255.0
                    color = frame_resized.reshape(-1, 3).astype(np.float32)

                    if depth_conf is not None:
                        dc = depth_conf.squeeze().numpy()
                        if dc.ndim == 2:
                            conf = dc.flatten().astype(np.float32)
                        else:
                            conf = np.ones(len(pc), dtype=np.float32)
                    else:
                        conf = (d.flatten() > 1e-4).astype(np.float32)
                else:
                    pc = np.zeros((0, 3), dtype=np.float32)
                    color = np.zeros((0, 3), dtype=np.float32)
                    conf = np.zeros(0, dtype=np.float32)

                ds_factor = args.downsample_factor
                if ds_factor > 1 and pc is not None and len(pc) > 0:
                    idx = np.arange(0, len(pc), ds_factor)
                    pc = pc[idx]
                    color = color[idx] if len(color) >= len(idx) else color
                    conf = conf[idx] if len(conf) >= len(idx) else conf

                if len(color) != len(pc):
                    color = np.zeros((len(pc), 3), dtype=np.float32)

                if c2w is not None:
                    c2w = np.asarray(c2w)
                    if c2w.shape == (4, 4):
                        pass
                    elif c2w.shape == (3, 4):
                        c2w_4x4 = np.eye(4, dtype=c2w.dtype)
                        c2w_4x4[:3] = c2w
                        c2w = c2w_4x4
                    elif c2w.ndim == 3 and c2w.shape[-2:] == (4, 4):
                        c2w = c2w[-1]
                    else:
                        print(f"[WARN frame {frame_idx}] c2w shape {c2w.shape} unexpected, skipping")
                        c2w = None

                return pc, color, conf, c2w, intrinsic

            if not stream_initialized:
                # unreachable: stream_initialized is flipped above
                pass

            latest_rgb = frame_rgb
            latest_index = max(0, int(predictions_cpu.get("pose_enc").shape[1] - 1)) if isinstance(predictions_cpu.get("pose_enc"), torch.Tensor) else 0
            if isinstance(predictions_cpu.get("pose_enc"), torch.Tensor):
                sequence_len = predictions_cpu["pose_enc"].shape[1]
            else:
                sequence_len = 1

            new_viewer_entries = []
            if frame_idx == 0 and sequence_len > 1:
                # Bootstrap path: expose every scale frame so the map starts with
                # a small continuous bundle instead of a single frame.
                for seq_idx in range(sequence_len):
                    pc, color, conf, c2w, intrinsic = _build_viewer_frame(seq_idx, latest_rgb)
                    new_viewer_entries.append((pc, color, conf, c2w, intrinsic, latest_rgb))
                pc, color, conf, c2w, intrinsic, _ = new_viewer_entries[-1]
            else:
                pc, color, conf, c2w, intrinsic = _build_viewer_frame(sequence_len - 1, latest_rgb)
                new_viewer_entries.append((pc, color, conf, c2w, intrinsic, latest_rgb))

            # ---- DIAGNOSTICS: print processed results ----
            if frame_idx < 5:
                if pc is not None and len(pc) > 0:
                    pc_center = pc.mean(axis=0)
                    pc_norm_95 = np.percentile(np.linalg.norm(pc - pc_center, axis=1), 95) if len(pc) > 0 else 0
                    print(f"[DEBUG frame {frame_idx}] pc={len(pc)} pts, "
                          f"c2w_t={c2w[:3, 3] if c2w is not None and c2w.shape == (4, 4) else 'N/A'}, "
                          f"color_range=[{color.min():.3f},{color.max():.3f}], "
                          f"conf_range=[{conf.min():.4f},{conf.max():.4f}] "
                          f"(mean={conf.mean():.4f}), "
                          f"pc_scale={pc_norm_95:.4f}")
                else:
                    print(f"[DEBUG frame {frame_idx}] pc=EMPTY, c2w={c2w.shape if c2w is not None else None}")

            # ---- Loop detection ----
            if loop_detector is not None and c2w is not None:
                if loop_detector.update(c2w):
                    stop_reason = "loop_closure"
                    print(f"[Inference] Loop closure detected at frame {frame_idx}")
                    break

            # ---- Keyframe gating for viewer ----
            if args.enable_keyframe_gate:
                # Gating enabled: only show frames with significant camera motion
                accepted, t_delta, r_delta = _should_accept_keyframe(
                    last_keyframe_c2w,
                    c2w,
                    translation_thresh=args.keyframe_translation_thresh,
                    rotation_thresh_deg=args.keyframe_rotation_thresh_deg,
                )
            else:
                # No gating: show every valid frame
                accepted = c2w is not None
                t_delta, r_delta = 0.0, 180.0 if accepted else 0.0

            if accepted:
                for entry in new_viewer_entries:
                    viewer_frames.append(entry)
                last_keyframe_c2w = c2w.copy() if c2w is not None else last_keyframe_c2w
                while len(viewer_frames) > args.max_viewer_frames:
                    viewer_frames.pop(0)
            elif frame_idx % 20 == 0:
                print(f"[Viewer] skip frame {frame_idx}: "
                      f"(dt={t_delta:.6f}, dr={r_delta:.2f}deg)")

            # Update viewer state
            if viewer_state is not None:
                viewer_state["frames"] = list(viewer_frames)

            # FPS tracking
            fps_frame_count += 1
            total_time = time.perf_counter() - last_infer_time
            last_infer_time = time.perf_counter()
            total_elapsed += total_time
            fps_window.append(1.0 / total_time if total_time > 0 else 0)
            if len(fps_window) > 30:
                fps_window.pop(0)

            now = time.time()
            if now - fps_report_time >= 5.0:
                avg_fps = sum(fps_window) / len(fps_window) if fps_window else 0
                overall_fps = fps_frame_count / total_elapsed if total_elapsed > 0 else 0
                print(f"  FPS: {avg_fps:.1f} (infer={infer_time*1000:.0f}ms, "
                      f"overall={overall_fps:.1f})  frames={frame_idx}")
                fps_report_time = now

            frame_idx += 1

    except KeyboardInterrupt:
        stop_reason = "keyboard"
        print("[Inference] Interrupted by user")

    # ---- Cleanup ----
    print(f"[Inference] Stop requested: {stop_reason}")
    stop_capture.set()
    capture_thread.join(timeout=3)
    if viewer_state:
        viewer_state["running"] = False
    if viewer_thread:
        viewer_thread.join(timeout=3)

    # ---- Export ----
    output_dir = args.output_dir or f"realtime_output_{stop_reason}"
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.getcwd(), output_dir)

    print(f"[Export] Exporting {len(viewer_frames)} frames to {output_dir}...")
    export_reconstruction(
        viewer_frames,
        output_dir,
        conf_threshold=args.conf_threshold,
        downsample_factor=args.downsample_factor,
        export_glb=args.export_glb,
        export_npz=args.export_npz,
        export_video=args.export_video,
    )

    print(f"\n  FPS Overall: {overall_fps:.1f}")


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LingBot-MAP Real-Time 3D Reconstruction from UVC Camera.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Camera
    g_cam = parser.add_argument_group("Camera")
    g_cam.add_argument("--device", default="cuda", help="Device for inference (cuda/cpu)")
    g_cam.add_argument("--video_device", default="/dev/video0",
                       help="V4L2 device path (e.g. /dev/video0)")
    g_cam.add_argument("--image_width", type=int, default=640, help="Camera capture width")
    g_cam.add_argument("--image_height", type=int, default=384, help="Camera capture height")
    g_cam.add_argument("--fps", type=int, default=20, help="Camera framerate")
    g_cam.add_argument("--capture_fps", type=int, default=None,
                       help="Capture framerate cap (default: same as --fps)")
    g_cam.add_argument("--pixel_format", default="MJPG",
                       help="Camera pixel format: MJPG, YUYV, NV12")
    g_cam.add_argument("--use_gstreamer", action="store_true",
                       help="Force GStreamer pipeline (recommended for Jetson)")

    # Model
    g_model = parser.add_argument_group("Model")
    g_model.add_argument("--model_path", required=True, help="Path to model checkpoint (.pt file)")
    g_model.add_argument("--image_size", type=int, default=518, help="Model input image width (default: 518)")
    g_model.add_argument("--patch_size", type=int, default=14, help="Patch size (default: 14)")
    g_model.add_argument("--num_scale_frames", type=int, default=8,
                         help="Number of initial frames for scale estimation (default: 8)")
    g_model.add_argument("--enable_3d_rope", action=argparse.BooleanOptionalAction, default=True,
                         help="Enable 3D rotary position encoding (default: enabled)")
    g_model.add_argument("--max_frame_num", type=int, default=256,
                        help="Max number of frames for 3D RoPE (default: 256). "
                             "KV cache resets automatically before hitting this limit.")
    g_model.add_argument("--kv_reset_interval", type=int, default=200,
                        help="Reset KV cache every N frames to prevent 3D RoPE index overflow (default: 200). "
                             "Should be less than --max_frame_num.")
    g_model.add_argument("--kv_cache_sliding_window", type=int, default=64,
                         help="KV cache sliding window size")
    g_model.add_argument("--use_sdpa", action="store_true",
                         help="Use SDPA backend instead of FlashInfer (slower, no flashinfer needed)")
    g_model.add_argument("--camera_num_iterations", type=int, default=2,
                         help="Camera head iterations (1=faster, 4=better accuracy, default: 2)")

    # Viewer
    g_view = parser.add_argument_group("Viewer")
    g_view.add_argument("--server_ip", default="0.0.0.0",
                        help="Viewer server bind IP (default: 0.0.0.0 for LAN access)")
    g_view.add_argument("--port", type=int, default=8080, help="HTTP/WebSocket port")
    g_view.add_argument("--conf_threshold", type=float, default=0.0,
                        help="Confidence percentile (0-100) for filtering points (default: 0.0 = keep all)")
    g_view.add_argument("--downsample_factor", type=int, default=1,
                        help="Point cloud downsample factor for viewer (default: 1)")
    g_view.add_argument("--point_size", type=float, default=0.012,
                        help="Viewer point size (default: 0.012 for denser realtime display)")
    g_view.add_argument("--max_viewer_frames", type=int, default=300,
                        help="Maximum frames kept in viewer history (default: 300)")
    g_view.add_argument("--enable_keyframe_gate", action="store_true",
                        help="Enable pose-change gating (only show frames when camera moves significantly). "
                             "By default, every valid frame is shown in the viewer.")
    g_view.add_argument("--keyframe_translation_thresh", type=float, default=0.0025,
                        help="Min translation delta for accepting frame (default: 0.0025)")
    g_view.add_argument("--keyframe_rotation_thresh_deg", type=float, default=6.0,
                        help="Min rotation delta in degrees for accepting frame (default: 6.0)")

    # Stop conditions
    g_stop = parser.add_argument_group("Stop")
    g_stop.add_argument("--max_frames", type=int, default=None,
                        help="Auto-stop after processing N frames")
    g_stop.add_argument("--enable_loop_detection", action="store_true",
                        help="Enable loop closure detection to auto-stop")
    g_stop.add_argument("--loop_threshold", type=float, default=0.5,
                        help="Distance threshold for loop closure (default: 0.5)")
    g_stop.add_argument("--loop_min_interval", type=int, default=30,
                        help="Min frames between loop detection matches (default: 30)")
    g_stop.add_argument("--loop_min_history", type=int, default=30,
                        help="Min frames observed before enabling loop detection (default: 30)")

    # Export
    g_exp = parser.add_argument_group("Export")
    g_exp.add_argument("--output_dir", default=None,
                       help="Output directory (default: realtime_output_<stop_reason>/)")
    g_exp.add_argument("--export_glb", action="store_true",
                       help="Export GLB point cloud with camera trajectory")
    g_exp.add_argument("--export_npz", action="store_true",
                       help="Export NPZ with raw predictions")
    g_exp.add_argument("--export_video", action="store_true",
                       help="Export MP4 flythrough video (requires open3d)")

    args = parser.parse_args()

    if args.enable_loop_detection and args.loop_min_history < args.loop_min_interval:
        args.loop_min_history = args.loop_min_interval

    run_realtime(args)


if __name__ == "__main__":
    main()

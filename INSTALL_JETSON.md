# LingBot-Map Jetson Install Notes

This note records the working installation path for this project on a Jetson
device like this machine:

- JetPack `6.2.1`
- L4T `36.4.4`
- Ubuntu `22.04`
- Python `3.10`

It avoids the common failure where `pip install torch torchvision` pulls a
generic CUDA wheel and `torch.cuda.is_available()` becomes `False`.

## 1. System prerequisites

Verify the Jetson release:

```bash
cat /etc/nv_tegra_release
dpkg -l | grep nvidia-jetpack
python3 --version
```

Expected on this machine:

- `nvidia-jetpack 6.2.1`
- Python `3.10.x`

Install base system packages:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip libopenblas-dev libjpeg-dev zlib1g-dev
```

For JetPack 6, cuSPARSELt must be present for recent NVIDIA PyTorch wheels:

```bash
dpkg -l | grep libcusparselt
```

If missing:

```bash
wget https://developer.download.nvidia.com/compute/cusparselt/0.7.1/local_installers/cusparselt-local-tegra-repo-ubuntu2204-0.7.1_1.0-1_arm64.deb
sudo dpkg -i cusparselt-local-tegra-repo-ubuntu2204-0.7.1_1.0-1_arm64.deb
sudo cp /var/cusparselt-local-tegra-repo-ubuntu2204-0.7.1/cusparselt-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install -y libcusparselt0 libcusparselt-dev
```

## 2. Create project environment

```bash
cd /home/seeed/Downloads/bak/lingbot-map
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 3. Install the project itself

This repo needs a tiny `setup.py` shim so editable install works correctly with
the current `pyproject.toml`:

```python
from setuptools import setup

setup()
```

Then install the package:

```bash
pip install -e .
```

## 4. Install viewer and project dependencies

Install the packages used by `demo.py` and `demo_realtime.py`:

```bash
pip install \
  "numpy==1.26.4" \
  "opencv-python==4.10.0.84" \
  "viser>=0.2.23" \
  aiohttp \
  onnxruntime \
  trimesh \
  matplotlib \
  requests
```

Why pin these:

- `numpy 2.x` breaks the NVIDIA Jetson PyTorch wheel bridge
- newer `opencv-python` wheels may require `numpy>=2`

## 5. Install Jetson GPU PyTorch

Do not run:

```bash
pip install torch torchvision
```

That can pull a generic wheel like `torch 2.12.0+cu130`, which imported but had:

```python
torch.cuda.is_available() == False
```

On this machine, the final working pair was installed from these local wheels:

- `/home/seeed/Downloads/torch-2.8.0a0+gitba56102-cp310-cp310-linux_aarch64.whl`
- `/home/seeed/Downloads/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl`

Install them together:

```bash
pip uninstall -y torchvision torch
pip install \
  /home/seeed/Downloads/torch-2.8.0a0+gitba56102-cp310-cp310-linux_aarch64.whl \
  /home/seeed/Downloads/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl
```

Verify:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
if torch.cuda.device_count():
    print(torch.cuda.get_device_name(0))
PY
```

Expected:

- `torch 2.8.0a0+gitba56102`
- `True`
- device name like `Orin`

## 6. Install TorchVision for Jetson

This project imports:

```python
from torchvision import transforms as TF
```

The final verified working result on this machine is:

- `torch 2.8.0a0+gitba56102`
- `torchvision 0.23.0`
- `torch.cuda.is_available() == True`

### 6A. Preferred path: matched local wheel pair

If you already have the local wheel pair from section 5, use that first. This
was the only path fully verified end-to-end on this machine.

Verify:

```bash
python - <<'PY'
import torch
import torchvision
from torchvision import transforms as TF
print(torch.__version__)
print(torchvision.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(TF.ToTensor)
PY
```

Expected:

- `torch 2.8.0a0+gitba56102`
- `torchvision 0.23.0`
- `True`

### 6B. Tutorial links and alternate wheel sources

The Seeed Jetson tutorial points to SharePoint-hosted wheel downloads and also
mentions matching JP6/CUDA 12.6 pairs such as:

- `PyTorch 2.5 + torchvision 0.20`
- `PyTorch 2.7 + torchvision 0.22.0`

In CLI-only environments, those SharePoint links may download an HTML page
instead of a raw wheel. If that happens, download the real `.whl` files in a
browser first, then install from local files.

A secondary Jetson wheel index that can be useful for experiments is:

```bash
https://pypi.jetson-ai-lab.io/jp6/cu126/+simple
```

If a mismatched wheel is installed, you may see:

```text
RuntimeError: operator torchvision::nms does not exist
```

That means the wheel does not match the installed `torch`.

### 6C. Fallback: build TorchVision from source

If the wheel path above fails, build TorchVision against the installed NVIDIA
PyTorch.

Recommended example:

```bash
cd /tmp
rm -rf torchvision-jetson
git clone --branch v0.20.0 --depth 1 https://github.com/pytorch/vision torchvision-jetson
cd torchvision-jetson
python setup.py install
```

Notes:

- Build is slow on Jetson
- It may warn about missing WEBP or NVJPEG support; this is acceptable for this
  project
- `ninja` is optional; without it the build falls back to distutils and is
  slower

Optional speed-up:

```bash
pip install ninja
```

Verify:

```bash
python - <<'PY'
import torch
import torchvision
from torchvision import transforms as TF
print(torch.__version__)
print(torch.cuda.is_available())
print(torchvision.__version__)
print(TF.ToTensor)
PY
```

If import fails with:

```text
RuntimeError: operator torchvision::nms does not exist
```

the wheel or build does not match the installed `torch`, and you should retry
with a closer TorchVision version or rebuild from source against the current
environment.

## 7. Project-specific fixes in this bak copy

This `bak` copy also needed two local fixes:

### `setup.py`

Added to make editable install work and avoid package name becoming
`UNKNOWN-0.0.0`.

### `lingbot_map/utils/pose_enc.py`

There was an indentation error around the intrinsic matrix block. The fixed
section keeps the epsilon-based focal length calculation inside:

```python
if build_intrinsics:
    H, W = image_size_hw
    eps = 1e-6
    fy = (H / 2.0) / torch.tan(fov_h / 2.0 + eps)
    fx = (W / 2.0) / torch.tan(fov_w / 2.0 + eps)
```

Without this fix, both `demo.py` and `demo_realtime.py` fail at import time
with `IndentationError`.

## 8. Runtime checks

Check the interactive demo:

```bash
cd /home/seeed/Downloads/bak/lingbot-map
source .venv/bin/activate
python demo.py --help
```

Check the real-time demo:

```bash
python demo_realtime.py --help
```

If both print help normally, imports are healthy.

## 9. Real-time viewer modes

`demo_realtime.py` supports:

- `--render_mode viser`
  - browser-side WebGL point-cloud rendering
- `--render_mode stream`
  - server-side Open3D render, browser shows JPEG stream

`stream` mode additionally needs `open3d` installed.

## 10. Real camera bring-up without code changes

Important:

- Always run `demo_realtime.py` from this repo's `.venv`
- Do not use the system `python` / conda `python`
- If you forget this, the common failure is:

```text
ModuleNotFoundError: No module named 'torch'
```

Use:

```bash
cd /home/seeed/Downloads/bak/lingbot-map
source .venv/bin/activate
python demo_realtime.py --help
```

If `demo_realtime.py` opens the camera but then fails during warmup or forward
with a shape error like:

```text
RuntimeError: shape ... is invalid for input of size ...
```

you can often avoid code changes by reducing scale frames to `1`.

Why:

- the realtime script feeds one camera frame at a time
- with some parameter combinations, `--num_scale_frames 4` can mismatch the
  actual single-frame streaming path during warmup

For the USB camera on this machine, the stable no-code command is:

```bash
cd /home/seeed/Downloads/bak/lingbot-map
source .venv/bin/activate
python -u demo_realtime.py \
  --model_path ./lingbot-map.pt \
  --video_device /dev/video0 \
  --image_width 640 \
  --image_height 360 \
  --fps 10 \
  --pixel_format MJPG \
  --num_scale_frames 1 \
  --use_sdpa \
  --server_ip 0.0.0.0 \
  --port 18087
```

Notes:

- do not use `--host` here; `demo_realtime.py` uses `--server_ip`
- `/dev/video0` is the actual UVC video stream on this machine
- `/dev/video1` is metadata, not the camera image stream
- `640x360 MJPG` is a confirmed working camera mode here

Viewer URLs after startup:

- local: `http://127.0.0.1:18087`
- LAN: `http://192.168.137.137:18087`

Tradeoff:

- `--num_scale_frames 1` is the easiest way to get realtime working without
  touching code
- scale estimation may be less stable than multi-frame startup

## 11. Known failure modes

### `torch.cuda.is_available()` is `False`

Cause:

- wrong generic PyTorch wheel

Fix:

- uninstall generic `torch` and `torchvision`
- reinstall the NVIDIA Jetson wheel from section 5

### NumPy warning or torch import complains about NumPy ABI

Cause:

- `numpy 2.x`

Fix:

```bash
pip install "numpy==1.26.4"
```

### OpenCV conflicts after pinning NumPy

Cause:

- newer OpenCV wheel expects `numpy>=2`

Fix:

```bash
pip install "opencv-python==4.10.0.84"
```

### `demo.py` or `demo_realtime.py` fails on `torchvision`

Cause:

- TorchVision missing, or TorchVision does not match the installed Jetson
  PyTorch wheel

Fix:

- first try a matching wheel from the Jetson wheel index in section 6A
- if import still fails, build TorchVision from source as in section 6B

### `RuntimeError: operator torchvision::nms does not exist`

Cause:

- TorchVision wheel installed, but ABI/operator set does not match the local
  Jetson PyTorch build

Fix:

- uninstall TorchVision
- try a closer Jetson wheel version
- if that still fails, build TorchVision from source against the installed
  Jetson `torch`

## 12. Validated FlashInfer + GPU launch

The machine also supports a FlashInfer / GPU path for `demo_realtime.py`. The
camera modes confirmed on `/dev/video0` were:

- `MJPG`: `640x360@30`, `1280x720@30`, `1920x1080@30`
- `NV12`: `640x360@30`, `1280x720@15`

Check them again if needed:

```bash
v4l2-ctl --device /dev/video0 --list-formats-ext
```

### Recommended command

Keep FlashInfer and Torch JIT caches inside the repo so they stay writable:

```bash
cd /home/seeed/Downloads/bak/lingbot-map
source .venv/bin/activate
export FLASHINFER_WORKSPACE_BASE=/home/seeed/Downloads/bak/lingbot-map/.cache
export TORCH_EXTENSIONS_DIR=/home/seeed/Downloads/bak/lingbot-map/.cache/torch_extensions

python demo_realtime.py \
  --model_path /home/seeed/Downloads/bak/lingbot-map/lingbot-map.pt \
  --video_device /dev/video0 \
  --server_ip 0.0.0.0 \
  --port 18087 \
  --pixel_format MJPG \
  --image_width 640 \
  --image_height 360 \
  --fps 10 \
  --capture_fps 10 \
  --camera_num_iterations 4 \
  --num_scale_frames 4 \
  --conf_threshold 10 \
  --downsample_factor 2 \
  --export_glb \
  --export_npz
```

Notes:

- `--server_ip 0.0.0.0` exposes the viewer on the LAN
- use `.venv`, not system Python
- `MJPG 640x360@10` is the currently verified realtime path
- `camera_num_iterations 4` is preferred when pose stability matters

### LAN URL

Find the device IP:

```bash
hostname -I
```

One active LAN IP on this machine was:

```text
192.168.137.137
```

Open the viewer from another device on the same network:

```text
http://192.168.137.137:18087
```

### Stable background service and logs

The most reliable way on this machine is to use the helper scripts in
`scripts/` instead of ad-hoc `nohup`:

```bash
cd /home/seeed/Downloads/bak/lingbot-map
bash scripts/start_realtime_service.sh
```

Check status:

```bash
bash scripts/status_realtime_service.sh
```

Stop it:

```bash
bash scripts/stop_realtime_service.sh
```

Default service settings baked into the script:

- port `18087`
- `MJPG`
- `640x360`
- `fps 10`
- `capture_fps 10`
- `camera_num_iterations 4`
- `num_scale_frames 4`

The script writes:

- PID file: `/tmp/lingbot_realtime.pid`
- log file: `/tmp/lingbot_realtime.log`

### Manual background launch and logs

```bash
cd /home/seeed/Downloads/bak/lingbot-map
source .venv/bin/activate
export FLASHINFER_WORKSPACE_BASE=/home/seeed/Downloads/bak/lingbot-map/.cache
export TORCH_EXTENSIONS_DIR=/home/seeed/Downloads/bak/lingbot-map/.cache/torch_extensions

setsid python -u demo_realtime.py \
  --model_path /home/seeed/Downloads/bak/lingbot-map/lingbot-map.pt \
  --video_device /dev/video0 \
  --server_ip 0.0.0.0 \
  --port 18087 \
  --pixel_format MJPG \
  --image_width 640 \
  --image_height 360 \
  --fps 10 \
  --capture_fps 10 \
  --camera_num_iterations 4 \
  --num_scale_frames 4 \
  --conf_threshold 10 \
  --downsample_factor 2 \
  --export_glb \
  --export_npz \
  > /tmp/lingbot_realtime.log 2>&1 < /dev/null &
```

Useful checks:

```bash
tail -f /tmp/lingbot_realtime.log
```

```bash
ss -ltnp | grep 18087
```

```bash
pkill -f demo_realtime.py
```

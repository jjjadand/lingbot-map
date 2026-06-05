"""Quick profiling to understand where time is spent in streaming inference."""
import os
import sys
os.environ['PATH'] = '/home/seeed/miniconda3/envs/lingbot-map/bin:' + os.environ.get('PATH', '')
sys.path.insert(0, '/home/seeed/Downloads/bak/lingbot-map')

import torch
import time
from lingbot_map.models.gct_stream import GCTStream

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA arch: {torch.cuda.get_device_capability()}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

print("\nBuilding model...")
model = GCTStream(
    img_size=518,
    patch_size=14,
    embed_dim=1024,
    enable_3d_rope=False,
    kv_cache_sliding_window=64,
    kv_cache_scale_frames=4,
    kv_cache_cross_frame_special=True,
    kv_cache_include_scale_frames=True,
    use_sdpa=False,
    camera_num_iterations=4,
)
print("Model built.")

print(f"use_sdpa: {model.aggregator.use_sdpa}")
print(f"use_flashinfer: {model.aggregator.use_flashinfer}")
print(f"Global block type: {type(model.aggregator.global_blocks[0]).__name__}")
print(f"Frame block type: {type(model.aggregator.frame_blocks[0]).__name__}")

print("\nLoading weights...")
ckpt = torch.load('/home/seeed/Downloads/bak/lingbot-map/lingbot-map.pt', map_location='cpu', weights_only=False)
state_dict = ckpt.get("model", ckpt)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

dtype = torch.bfloat16
model = model.to(device).eval()
if dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
    model.aggregator = model.aggregator.to(dtype=dtype)
print(f"Model on {device}, aggregator dtype={dtype}, rest=float32")

dummy = torch.randn(1, 1, 3, 518, 378, device=device, dtype=dtype)
print(f"Input: {dummy.shape}")

print("\nWarmup...")
with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
    _ = model.forward(dummy, num_frame_for_scale=4, num_frame_per_block=1, causal_inference=True)
torch.cuda.synchronize() if torch.cuda.is_available() else None
print("Warmup done.")

print("\nTiming 10 forward passes...")
times = []
for i in range(10):
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        out = model.forward(dummy, num_frame_for_scale=4, num_frame_per_block=1, causal_inference=True)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.perf_counter()
    times.append(t1 - t0)
    print(f"  Frame {i}: {t1-t0:.3f}s")

avg = sum(times) / len(times)
print(f"\nAverage: {avg:.3f}s/frame = {1/avg:.1f} FPS")
if torch.cuda.is_available():
    print(f"Peak GPU mem: {torch.cuda.max_memory_allocated()/1024**2:.0f} MB")

mgr = model.aggregator.kv_cache_manager
if mgr:
    stats = mgr.get_cache_stats(0)
    print(f"KV cache stats: {stats}")
print("\nDone.")

"""Detailed PyTorch profiler analysis to identify exact bottlenecks."""
import os
import sys
os.environ['PATH'] = '/home/seeed/miniconda3/envs/lingbot-map/bin:' + os.environ.get('PATH', '')
sys.path.insert(0, '/home/seeed/Downloads/bak/lingbot-map')

import torch
import time
from lingbot_map.models.gct_stream import GCTStream

device = torch.device('cuda')
dtype = torch.bfloat16
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("Building model with FlashInfer backend...")
model = GCTStream(
    img_size=518, patch_size=14, embed_dim=1024,
    enable_3d_rope=False,
    kv_cache_sliding_window=64,
    kv_cache_scale_frames=4,
    use_sdpa=False,
    camera_num_iterations=4,
)

ckpt = torch.load('/home/seeed/Downloads/bak/lingbot-map/lingbot-map.pt',
                  map_location='cpu', weights_only=False)
state_dict = ckpt.get("model", ckpt)
model.load_state_dict(state_dict, strict=False)

model = model.to(device).eval()
model.aggregator = model.aggregator.to(dtype=dtype)

dummy = torch.randn(1, 1, 3, 518, 378, device=device, dtype=dtype)

# Warmup
print("Warming up...")
with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
    for _ in range(3):
        _ = model.forward(dummy, num_frame_for_scale=4, num_frame_per_block=1, causal_inference=True)
torch.cuda.synchronize()
print("Warmup done.\n")

# Profile with autocast
print("Running profiler...")
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True,
) as prof:
    for i in range(5):
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            out = model.forward(dummy, num_frame_for_scale=4, num_frame_per_block=1, causal_inference=True)
        torch.cuda.synchronize()

torch.cuda.synchronize()

# Print CUDA time breakdown (top 30 kernel by CUDA time)
print("\n" + "="*80)
print("TOP 30 CUDA KERNELS BY CUDA TIME")
print("="*80)
print(prof.key_averages().table(
    sort_by="cuda_time_total", row_limit=30,
    header=None
))

# Print CUDA time for operator categories
print("\n" + "="*80)
print("OPERATOR CATEGORY BREAKDOWN (by CUDA time)")
print("="*80)
key_avgs = prof.key_averages()

categories = {}
for evt in key_avgs:
    name = evt.key
    if not name or evt.cuda_time_total <= 0:
        continue
    if "flashinfer" in name.lower() or "batchprefill" in name.lower():
        cat = "FlashInfer"
    elif "attn" in name.lower() or "scaled_dot" in name.lower() or "sdpa" in name.lower():
        cat = "Attention (SDPA)"
    elif "conv" in name.lower() or "conv2d" in name.lower() or "im2col" in name.lower():
        cat = "Convolution"
    elif "gemm" in name.lower() or "matmul" in name.lower() or "linear" in name.lower():
        cat = "Linear/GEMM"
    elif "layernorm" in name.lower() or "layer_norm" in name.lower() or "norm" in name.lower():
        cat = "LayerNorm"
    elif "rope" in name.lower() or "rotary" in name.lower():
        cat = "RoPE"
    elif "add" in name.lower() or "mul" in name.lower() or "silu" in name.lower() or "gelu" in name.lower():
        cat = "Activation/Add"
    elif "softmax" in name.lower():
        cat = "Softmax"
    elif "permute" in name.lower() or "reshape" in name.lower() or "view" in name.lower() or "contiguous" in name.lower():
        cat = "Tensor reshape/permute"
    elif "copy" in name.lower() or "clone" in name.lower():
        cat = "Copy/Clone"
    elif "cat" in name.lower() or "concat" in name.lower():
        cat = "Concat"
    elif "chunk" in name.lower() or "split" in name.lower():
        cat = "Split"
    else:
        cat = "Other"

    if cat not in categories:
        categories[cat] = {"cuda_time": 0, "count": 0, "self_cuda_time": 0}
    categories[cat]["cuda_time"] += evt.cuda_time_total
    categories[cat]["count"] += 1
    categories[cat]["self_cuda_time"] += evt.self_cuda_time_total

sorted_cats = sorted(categories.items(), key=lambda x: x[1]["cuda_time"], reverse=True)
total_cuda = sum(c[1]["cuda_time"] for c in sorted_cats)
for cat, stats in sorted_cats:
    pct = stats['cuda_time'] / total_cuda * 100
    self_pct = stats['self_cuda_time'] / total_cuda * 100
    print(f"  {cat:30s}  {stats['cuda_time']/1000:8.1f}ms ({pct:5.1f}%)  self={stats['self_cuda_time']/1000:7.1f}ms ({self_pct:4.1f}%)  count={stats['count']}")

fi_time = categories.get("FlashInfer", {"cuda_time": 0})["cuda_time"]
print(f"\n  FlashInfer fraction of total CUDA time: {fi_time/total_cuda*100:.1f}%")
print(f"  Non-FlashInfer CUDA time: {(total_cuda-fi_time)/1000:.1f}ms")

print("\n" + "="*80)
print("TOP 30 SELF-CUDA TIME (excluding child kernels)")
print("="*80)
print(prof.key_averages().table(
    sort_by="self_cuda_time_total", row_limit=30,
    header=None
))

print("\nProfile complete.")

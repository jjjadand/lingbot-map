"""Time individual sections of the forward pass to identify bottlenecks."""
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

print("Building model...")
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
model.load_state_dict(ckpt.get("model", ckpt), strict=False)
model = model.to(device).eval()
model.aggregator = model.aggregator.to(dtype=dtype)

dummy = torch.randn(1, 1, 3, 518, 378, device=device, dtype=dtype)

# Warmup
print("Warming up...")
with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
    for _ in range(3):
        _ = model.forward(dummy, num_frame_for_scale=4, num_frame_per_block=1, causal_inference=True)
torch.cuda.synchronize()
print("Done.\n")

N = 10

# Time whole forward
t_start = time.perf_counter()
for _ in range(N):
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        out = model.forward(dummy, num_frame_for_scale=4, num_frame_per_block=1, causal_inference=True)
    torch.cuda.synchronize()
total = (time.perf_counter() - t_start) / N * 1000
print(f"Total forward: {total:.1f}ms\n")

# ---- Aggregator timing ----
agg = model.aggregator
cam = model.camera_head

# Get agg output for cam/dpt timing
agg_out_tuple = agg(dummy, selected_idx=[4, 11, 17, 23], num_frame_for_scale=4, num_frame_per_block=1)
agg_out_list = agg_out_tuple[0]
torch.cuda.synchronize()

# Time aggregator
t = time.perf_counter()
for _ in range(N):
    _ = agg(dummy, selected_idx=[4, 11, 17, 23], num_frame_for_scale=4, num_frame_per_block=1)
    torch.cuda.synchronize()
t_agg = (time.perf_counter() - t) / N * 1000
print(f"Aggregator:    {t_agg:.1f}ms ({t_agg/total*100:.0f}%)\n")

# Time camera head
t = time.perf_counter()
for _ in range(N):
    _ = cam(agg_out_list, causal_inference=True, num_frame_per_block=1, num_frame_for_scale=4)
    torch.cuda.synchronize()
t_cam = (time.perf_counter() - t) / N * 1000
print(f"Camera head:   {t_cam:.1f}ms ({t_cam/total*100:.0f}%)\n")

# Time DPT head
from lingbot_map.heads.dpt_head import DPTHead
dpt = model.depth_head
t = time.perf_counter()
for _ in range(N):
    _ = dpt(agg_out_list, dummy, patch_start_idx=6)
    torch.cuda.synchronize()
t_dpt = (time.perf_counter() - t) / N * 1000
print(f"DPT head:      {t_dpt:.1f}ms ({t_dpt/total*100:.0f}%)\n")

# Check aggregator components
print("="*60)
print("BREAKDOWN SUMMARY")
print("="*60)
print(f"  Aggregator:   {t_agg:.1f}ms ({t_agg/total*100:.0f}%)")
print(f"  Camera head: {t_cam:.1f}ms ({t_cam/total*100:.0f}%)")
print(f"  DPT head:     {t_dpt:.1f}ms ({t_dpt/total*100:.0f}%)")
print(f"  Other:        {max(0,total-t_agg-t_cam-t_dpt):.1f}ms ({max(0,total-t_agg-t_cam-t_dpt)/total*100:.0f}%)")
print(f"  Total:        {total:.1f}ms")
print()

# --- Deeper: aggregator sub-components ---
print("="*60)
print("AGGREGATOR SUB-COMPONENT TIMING")
print("="*60)

# Get embed output for sub-component timing
from lingbot_map.aggregator.base import AggregatorBase
images_norm = dummy  # [B=1, S=1, C, H, W]

# Time patch embedding
# Simulate what _embed_images does
def get_embed(images):
    tokens = cam  # placeholder
    return agg._embed_images(images, num_frame_for_scale=4)

t = time.perf_counter()
for _ in range(N):
    tokens_out = agg._embed_images(dummy, num_frame_for_scale=4)
    torch.cuda.synchronize()
t_embed = (time.perf_counter() - t) / N * 1000
tokens_tuple = tokens_out
B2, S_in, _, H2, W2 = dummy.shape
tokens, B_local, S_local, S_global, P, C = tokens_tuple

print(f"  Patch embed: {t_embed:.1f}ms ({t_embed/total*100:.0f}%)\n")

# Time positions
t = time.perf_counter()
for _ in range(N):
    pos_local = agg._get_positions(1, S_local, H2, W2, device=device)
    pos_global = agg._get_positions(1, S_global, H2, W2, device=device)
    torch.cuda.synchronize()
t_pos = (time.perf_counter() - t) / N * 1000
print(f"  Positions:    {t_pos:.1f}ms ({t_pos/total*100:.0f}%)\n")

# Time frame attention (only 1 block group of frame attention, repeated 6 times)
# The frame blocks are called in each of 6 block groups
# Each group: 1 frame block pass + 1 global block pass
# We need to time just the frame blocks

# Simulate one frame block forward
t = time.perf_counter()
for _ in range(N):
    tokens_reshaped = tokens.view(B_local, S_local, P, C)
    pos_l = agg._get_positions(1, S_local, H2, W2, device=device)
    for fb in agg.frame_blocks:
        tokens_flat = tokens_reshaped.view(B_local, S_local * P, C)
        _ = fb(tokens_flat, pos=pos_l)
    torch.cuda.synchronize()
t_frame = (time.perf_counter() - t) / N * 1000
print(f"  Frame blocks (all 24): {t_frame:.1f}ms ({t_frame/total*100:.0f}%)\n")

# Time global blocks (all 24 FlashInfer blocks)
t = time.perf_counter()
from lingbot_map.layers.flashinfer_cache import FlashInferKVCacheManager
# Need to create a manager for this
mgr = agg.kv_cache_manager  # Should exist after the warmup agg call
# But if we call again, we need to reset
if mgr:
    mgr.reset()
for _ in range(N):
    tokens_flat = tokens.view(B_local, S_local * P, C)
    pos_g = agg._get_positions(1, S_global, H2, W2, device=device)
    global_idx = 0
    for gb in agg.global_blocks:
        _ = gb(tokens_flat, pos=pos_g, num_patches=P-6, num_special=6,
                num_frames=1, enable_3d_rope=False, kv_cache=mgr,
                global_idx=global_idx, num_frame_per_block=1,
                num_frame_for_scale=4, num_register_tokens=4)
        global_idx += 1
    torch.cuda.synchronize()
t_global = (time.perf_counter() - t) / N * 1000
print(f"  Global blocks (all 24): {t_global:.1f}ms ({t_global/total*100:.0f}%)\n")

t_overhead = t_agg - t_embed - t_pos - t_frame - t_global
print(f"  Overhead:    {t_overhead:.1f}ms ({t_overhead/total*100:.0f}%)")
print(f"  Aggregator sum: {t_embed+t_pos+t_frame+t_global+t_overhead:.1f}ms")

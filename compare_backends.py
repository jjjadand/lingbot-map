"""Compare FlashInfer vs SDPA performance for streaming inference."""
import os
import sys
os.environ['PATH'] = '/home/seeed/miniconda3/envs/lingbot-map/bin:' + os.environ.get('PATH', '')
sys.path.insert(0, '/home/seeed/Downloads/bak/lingbot-map')

import torch
import time
from lingbot_map.models.gct_stream import GCTStream

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.bfloat16
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def build_and_profile(use_sdpa: bool, label: str):
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {label}  (use_sdpa={use_sdpa})")
    print('='*60)

    model = GCTStream(
        img_size=518, patch_size=14, embed_dim=1024,
        enable_3d_rope=False,
        kv_cache_sliding_window=64,
        kv_cache_scale_frames=4,
        use_sdpa=use_sdpa,
        camera_num_iterations=4,
    )
    print(f"  Global block: {type(model.aggregator.global_blocks[0]).__name__}")
    print(f"  Backend: {'SDPA' if use_sdpa else 'FlashInfer'}")

    ckpt = torch.load('/home/seeed/Downloads/bak/lingbot-map/lingbot-map.pt',
                      map_location='cpu', weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=False)

    model = model.to(device).eval()
    if getattr(model, "aggregator", None) is not None:
        model.aggregator = model.aggregator.to(dtype=dtype)
    model = model.to(device)

    # Single-frame input
    dummy = torch.randn(1, 1, 3, 518, 378, device=device, dtype=dtype)

    # Warmup
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        _ = model.forward(dummy, num_frame_for_scale=4, num_frame_per_block=1, causal_inference=True)
    torch.cuda.synchronize()
    print("  Warmup done.")

    # Time 10 forward passes
    times = []
    for i in range(10):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            out = model.forward(dummy, num_frame_for_scale=4, num_frame_per_block=1, causal_inference=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        print(f"  Frame {i}: {t1-t0:.3f}s")

    avg = sum(times) / len(times)
    print(f"\n  Average: {avg:.3f}s/frame = {1/avg:.1f} FPS")
    print(f"  Peak GPU mem: {torch.cuda.max_memory_allocated()/1024**2:.0f} MB")

    # KV cache info
    agg = model.aggregator
    if use_sdpa:
        k0 = agg.kv_cache.get("k_0")
        print(f"  SDPA k_0 shape: {tuple(k0.shape) if k0 is not None else None}")
    else:
        mgr = agg.kv_cache_manager
        if mgr:
            print(f"  FlashInfer cache stats: {mgr.get_cache_stats(0)}")

    return avg

# Run both
t_sdpa = build_and_profile(use_sdpa=True, label="SDPA Backend")
t_fi  = build_and_profile(use_sdpa=False, label="FlashInfer Backend")

print(f"\n{'='*60}")
print("SUMMARY")
print('='*60)
print(f"  SDPA:       {t_sdpa:.3f}s/frame = {1/t_sdpa:.1f} FPS")
print(f"  FlashInfer: {t_fi:.3f}s/frame = {1/t_fi:.1f} FPS")
print(f"  Speedup:    {t_sdpa/t_fi:.2f}x")
print(f"  Improvement: {(t_sdpa-t_fi)/t_sdpa*100:.1f}% faster with FlashInfer")

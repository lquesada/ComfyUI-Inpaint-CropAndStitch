"""
Benchmark and correctness tests for fillholes_iterative_hipass_fill_m.

Compares three implementations:
  1. CPU  – scipy binary_closing + binary_fill_holes (the reference)
  2. Old GPU – single hardcoded >0.5 threshold flood-fill (the buggy version)
  3. New GPU – iterates all 14 thresholds with skip optimization (the fix)

Run:
    python test_fillholes_benchmark.py

Why CPU and GPU results can still differ
----------------------------------------
Even though the new GPU implementation iterates the same 14 thresholds as
the CPU version, minor numerical differences can occur because:

  * **Binary closing implementation differs.**
    CPU uses ``scipy.ndimage.binary_closing`` with ``border_value=1``, which
    treats out-of-bounds pixels as 1 during the erosion step (so border
    pixels are never eroded away).  The GPU approximation uses
    ``max_pool2d`` for dilation followed by ``-max_pool2d(-x)`` for erosion,
    both with zero-padding.  Zero-padding during erosion can erode mask
    pixels that touch the image border, producing slightly different closed
    masks.

  * **Flood-fill connectivity differs.**
    CPU ``binary_fill_holes`` uses 4-connectivity (cross-shaped structuring
    element) by default.  The GPU flood-fill uses ``max_pool2d`` with a 3×3
    kernel, which propagates in all 8 directions (8-connectivity).  This
    means the GPU may classify a thin diagonal gap as "sealed" while the CPU
    considers it open, or vice versa.

  * **Floating-point threshold comparison.**
    The CPU converts the mask to float64 (via numpy) before comparing with
    the threshold, while the GPU stays in float32 throughout.  Values like
    0.9 have slightly different representations in float32 vs float64, so
    ``>= 0.9`` can give different answers for border-case pixels.

For purely white (1.0) masks, these differences vanish because there is only
one threshold level (1.0) and the mask fills the entire image.
"""

import time
import numpy as np
import torch
import torch.nn.functional as TF
from scipy.ndimage import binary_closing, binary_fill_holes


# ---------------------------------------------------------------------------
# Implementation: CPU (reference – identical to CPUProcessorLogic)
# ---------------------------------------------------------------------------
def cpu_fillholes(samples):
    """CPU reference: scipy binary_closing + binary_fill_holes per threshold."""
    thresholds = [1, 0.99, 0.97, 0.95, 0.93, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    results = []
    for i in range(samples.shape[0]):
        mask_np = samples[i].cpu().numpy()
        for threshold in thresholds:
            thresholded_mask = mask_np >= threshold
            closed_mask = binary_closing(thresholded_mask, structure=np.ones((3, 3)), border_value=1)
            filled_mask = binary_fill_holes(closed_mask)
            mask_np = np.maximum(mask_np, np.where(filled_mask != 0, threshold, 0))
        results.append(torch.from_numpy(mask_np.astype(np.float32)))
    return torch.stack(results, dim=0)


# ---------------------------------------------------------------------------
# Implementation: Old GPU (single hardcoded >0.5 threshold – the bug)
# ---------------------------------------------------------------------------
def old_gpu_fillholes(samples):
    """Old GPU: single >0.5 threshold, misses values <= 0.5."""
    B, H, W = samples.shape
    device = samples.device
    inv_mask = 1.0 - (samples > 0.5).float()
    padded_inv = torch.zeros((B, H + 2, W + 2), device=device)
    padded_inv[:, 1:-1, 1:-1] = inv_mask
    curr_outside = torch.zeros((B, H + 2, W + 2), device=device)
    curr_outside[:, 0, :] = 1
    curr_outside[:, -1, :] = 1
    curr_outside[:, :, 0] = 1
    curr_outside[:, :, -1] = 1
    for _ in range(max(H, W)):
        next_outside = TF.max_pool2d(
            curr_outside.unsqueeze(1), kernel_size=3, stride=1, padding=1
        ).squeeze(1)
        next_outside = next_outside * padded_inv
        next_outside[:, 0, :] = 1
        next_outside[:, -1, :] = 1
        next_outside[:, :, 0] = 1
        next_outside[:, :, -1] = 1
        if torch.all(next_outside == curr_outside):
            break
        curr_outside = next_outside
    filled = 1.0 - curr_outside[:, 1:-1, 1:-1]
    return torch.max(samples, filled)


# ---------------------------------------------------------------------------
# Implementation: New GPU (14 thresholds with skip – the fix)
# ---------------------------------------------------------------------------
def new_gpu_fillholes(samples):
    """New GPU: iterates 14 thresholds, matching CPU threshold list."""
    B, H, W = samples.shape
    device = samples.device
    thresholds = [1, 0.99, 0.97, 0.95, 0.93, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    result = samples.clone()
    for threshold in thresholds:
        thresholded = (result >= threshold).float()
        if thresholded.sum() == 0:
            continue
        closed = TF.max_pool2d(thresholded.unsqueeze(1), kernel_size=3, stride=1, padding=1)
        closed = -TF.max_pool2d(-closed, kernel_size=3, stride=1, padding=1)
        closed = closed.squeeze(1)
        inv_mask = 1.0 - closed
        padded_inv = torch.zeros((B, H + 2, W + 2), device=device)
        padded_inv[:, 1:-1, 1:-1] = inv_mask
        curr_outside = torch.zeros((B, H + 2, W + 2), device=device)
        curr_outside[:, 0, :] = 1
        curr_outside[:, -1, :] = 1
        curr_outside[:, :, 0] = 1
        curr_outside[:, :, -1] = 1
        for _ in range(max(H, W)):
            next_outside = TF.max_pool2d(
                curr_outside.unsqueeze(1), kernel_size=3, stride=1, padding=1
            ).squeeze(1)
            next_outside = next_outside * padded_inv
            next_outside[:, 0, :] = 1
            next_outside[:, -1, :] = 1
            next_outside[:, :, 0] = 1
            next_outside[:, :, -1] = 1
            if torch.all(next_outside == curr_outside):
                break
            curr_outside = next_outside
        filled = 1.0 - curr_outside[:, 1:-1, 1:-1]
        result = torch.max(result, filled * threshold)
    return result


# ===================================================================
# Test helpers
# ===================================================================
ATOL = 1e-6  # absolute tolerance for floating-point comparisons


def make_pure_white_mask(size=512):
    """Pure-white mask (all 1.0) – only threshold=1.0 is active."""
    return torch.ones(1, size, size)


def make_ring_mask(size=64, value=0.3):
    """Ring of *value* with a hole inside – tests sub-0.5 fill."""
    m = torch.zeros(1, size, size)
    m[0, 10:54, 10:54] = value
    m[0, 20:44, 20:44] = 0.0
    return m


def make_multi_threshold_mask(size=64):
    """Concentric rings at several threshold levels."""
    m = torch.zeros(1, size, size)
    m[0, 5:59, 5:59] = 0.1
    m[0, 10:54, 10:54] = 0.3
    m[0, 15:49, 15:49] = 0.5
    m[0, 20:44, 20:44] = 0.7
    m[0, 25:39, 25:39] = 0.9
    m[0, 28:36, 28:36] = 0.0  # central hole
    return m


# ===================================================================
# Benchmark runner
# ===================================================================
def bench(fn, mask, label, warmup=1, repeats=5):
    """Time *fn* on *mask* and return (label, mean_ms)."""
    for _ in range(warmup):
        fn(mask.clone())
    times = []
    for _ in range(repeats):
        m = mask.clone()
        t0 = time.perf_counter()
        fn(m)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    mean = sum(times) / len(times)
    return label, mean


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    torch.set_grad_enabled(False)

    # ------------------------------------------------------------------
    # 1. SPEED COMPARISON
    # ------------------------------------------------------------------
    print("=" * 72)
    has_cuda = torch.cuda.is_available()
    env = "CUDA GPU" if has_cuda else "CPU-only (no GPU)"
    print(f"SPEED COMPARISON (environment: {env}, times in ms, 5 runs averaged)")
    print("=" * 72)

    for mask_name, mask_fn in [
        ("pure-white 512×512", lambda: make_pure_white_mask(512)),
        ("multi-threshold 64×64", lambda: make_multi_threshold_mask(64)),
    ]:
        mask = mask_fn()
        print(f"\nMask: {mask_name}")
        print("-" * 50)
        results = []
        for label, fn in [
            ("CPU (reference)", cpu_fillholes),
            ("Old GPU (single >0.5)", old_gpu_fillholes),
            ("New GPU (14 thresholds)", new_gpu_fillholes),
        ]:
            _, ms = bench(fn, mask, label)
            results.append((label, ms))
            print(f"  {label:30s}  {ms:8.2f} ms")

        # Ratios
        cpu_ms = results[0][1]
        old_ms = results[1][1]
        new_ms = results[2][1]
        print(f"\n  New GPU / Old GPU ratio:  {new_ms / old_ms:.2f}×")
        print(f"  New GPU / CPU ratio:      {new_ms / cpu_ms:.2f}×")

    # ------------------------------------------------------------------
    # 2. CORRECTNESS – pure-white mask (all implementations must agree)
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("CORRECTNESS: pure-white mask (should be identical for all)")
    print("=" * 72)

    mask = make_pure_white_mask(64)
    cpu_out = cpu_fillholes(mask.clone())
    old_out = old_gpu_fillholes(mask.clone())
    new_out = new_gpu_fillholes(mask.clone())

    diff_old = (cpu_out - old_out).abs().max().item()
    diff_new = (cpu_out - new_out).abs().max().item()
    print(f"  max|CPU − Old GPU|  = {diff_old:.6f}  {'✓ PASS' if diff_old < ATOL else '✗ FAIL'}")
    print(f"  max|CPU − New GPU|  = {diff_new:.6f}  {'✓ PASS' if diff_new < ATOL else '✗ FAIL'}")

    # ------------------------------------------------------------------
    # 3. CORRECTNESS – sub-0.5 ring (old GPU bug, new GPU must fix)
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("CORRECTNESS: ring with value 0.3 and inner hole (sub-0.5 bug test)")
    print("=" * 72)

    mask = make_ring_mask(64, value=0.3)
    cpu_out = cpu_fillholes(mask.clone())
    old_out = old_gpu_fillholes(mask.clone())
    new_out = new_gpu_fillholes(mask.clone())

    hole_y, hole_x = 32, 32  # centre of hole
    cpu_val = cpu_out[0, hole_y, hole_x].item()
    old_val = old_out[0, hole_y, hole_x].item()
    new_val = new_out[0, hole_y, hole_x].item()

    print(f"  Hole centre value:")
    print(f"    CPU (reference)        = {cpu_val:.4f}  (should be ≥ 0.3)")
    print(f"    Old GPU (buggy)        = {old_val:.4f}  (expected 0.0 – bug)")
    print(f"    New GPU (fixed)        = {new_val:.4f}  (should be ≥ 0.3)")
    print()
    old_bug = old_val < 0.1  # old GPU ignores <0.5
    new_fix = new_val >= 0.29
    cpu_ok = cpu_val >= 0.29
    print(f"  Old GPU missed sub-0.5?  {'✓ YES (expected bug)' if old_bug else '✗ NO (unexpected)'}")
    print(f"  New GPU fills sub-0.5?   {'✓ PASS' if new_fix else '✗ FAIL'}")
    print(f"  CPU fills sub-0.5?       {'✓ PASS' if cpu_ok else '✗ FAIL'}")

    # ------------------------------------------------------------------
    # 4. CORRECTNESS – multi-threshold mask
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("CORRECTNESS: multi-threshold concentric-ring mask")
    print("=" * 72)
    print("(Minor differences between CPU and GPU are expected – see docstring)")

    mask = make_multi_threshold_mask(64)
    cpu_out = cpu_fillholes(mask.clone())
    old_out = old_gpu_fillholes(mask.clone())
    new_out = new_gpu_fillholes(mask.clone())

    diff_old = (cpu_out - old_out).abs().max().item()
    diff_new = (cpu_out - new_out).abs().max().item()
    print(f"  max|CPU − Old GPU|  = {diff_old:.6f}")
    print(f"  max|CPU − New GPU|  = {diff_new:.6f}")
    print()

    # The new GPU should be much closer to CPU than the old GPU
    # (or at worst equal for this particular mask)
    print(f"  New GPU closer to CPU than Old GPU?  ", end="")
    if diff_new <= diff_old:
        print("✓ YES" + (f"  (both {diff_new:.6f})" if diff_new == diff_old else ""))
    else:
        print(f"✗ NO  (new={diff_new:.6f} > old={diff_old:.6f})")

    # Check that the centre hole is filled by both CPU and new GPU
    hole_y, hole_x = 32, 32
    cpu_val = cpu_out[0, hole_y, hole_x].item()
    new_val = new_out[0, hole_y, hole_x].item()
    old_val = old_out[0, hole_y, hole_x].item()
    print(f"\n  Centre-hole values:  CPU={cpu_val:.4f}  Old GPU={old_val:.4f}  New GPU={new_val:.4f}")

    # Detailed per-threshold correctness
    print(f"\n  Pixel-level differences (CPU vs New GPU):")
    diff_map = (cpu_out - new_out).abs()
    n_diff = (diff_map > ATOL).sum().item()
    total = cpu_out.numel()
    print(f"    Differing pixels: {n_diff}/{total}  ({100*n_diff/total:.1f}%)")
    if n_diff > 0:
        print(f"    Max absolute diff: {diff_map.max().item():.6f}")
        print(f"    Mean absolute diff (over differing pixels): "
              f"{diff_map[diff_map > ATOL].mean().item():.6f}")

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)

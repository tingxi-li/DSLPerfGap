import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from test_utils import get_tol, compare_tensors, RESULTS_DIR
import torch
import json, time, traceback

from pytorch_impl import cross_entropy_fwd as pytorch_fwd
from triton_impl import cross_entropy_fwd as triton_fwd
from tilelang_impl import cross_entropy_fwd as tilelang_fwd


torch.manual_seed(42)

n_rows, n_cols = 4, 8
logits = torch.randn((n_rows, n_cols), dtype=torch.float32, device='cuda')
labels = torch.randint(0, n_cols, (n_rows,), dtype=torch.int32, device='cuda')


def run_fwd(impl, logits, labels, smoothing, logit_scale, lse_square_scale,
            ignored_index, total_classes, class_start_idx, BLOCK_SIZE,
            HAS_SMOOTHING, SPLIT):
    return impl(logits, labels, smoothing, logit_scale, lse_square_scale,
                ignored_index, total_classes, class_start_idx,
                BLOCK_SIZE, HAS_SMOOTHING, SPLIT)


def compare_valid(ref_tuple, test_tuple, block_size, nr, nc, is_split, dtype):
    """Compare only valid output positions, skip z_loss if SPLIT."""
    num_blocks = (nc + block_size - 1) // block_size
    valid_indices = []
    for cb in range(num_blocks):
        for r in range(nr):
            flat_idx = cb * nr + r
            valid_indices.append(flat_idx)

    overall_pass = True
    overall_max_err = 0.0

    # Compare loss and lse (indices 0 and 1), conditionally z_loss (index 2)
    for idx, (ref, test) in enumerate(zip(ref_tuple, test_tuple)):
        if idx == 2 and is_split:
            # z_loss not stored when SPLIT=True, skip
            continue
        if not (isinstance(ref, torch.Tensor) and isinstance(test, torch.Tensor)):
            continue

        ref_f = ref.float().flatten()
        test_f = test.float().flatten()
        vi = torch.tensor([i for i in valid_indices if i < ref_f.numel()],
                          device=ref.device, dtype=torch.long)
        ref_v = ref_f[vi]
        test_v = test_f[vi]
        p, e = compare_tensors(ref_v, test_v, dtype, True)
        overall_pass = overall_pass and p
        overall_max_err = max(overall_max_err, e)

    return overall_pass, overall_max_err


test_configs = [
    {
        "name": "no_smooth_no_split_1blk",
        "params": dict(smoothing=0.0, logit_scale=1.0, lse_square_scale=0.1,
                        ignored_index=-1, total_classes=10, class_start_idx=0,
                        BLOCK_SIZE=8, HAS_SMOOTHING=False, SPLIT=False),
    },
    {
        "name": "smooth_no_split_1blk",
        "params": dict(smoothing=0.1, logit_scale=1.0, lse_square_scale=0.1,
                        ignored_index=-1, total_classes=10, class_start_idx=0,
                        BLOCK_SIZE=8, HAS_SMOOTHING=True, SPLIT=False),
    },
    {
        "name": "smooth_split_1blk",
        "params": dict(smoothing=0.1, logit_scale=1.0, lse_square_scale=0.1,
                        ignored_index=-1, total_classes=10, class_start_idx=0,
                        BLOCK_SIZE=8, HAS_SMOOTHING=True, SPLIT=True),
    },
    {
        "name": "no_smooth_no_split_2blk",
        "params": dict(smoothing=0.0, logit_scale=1.0, lse_square_scale=0.1,
                        ignored_index=-1, total_classes=10, class_start_idx=0,
                        BLOCK_SIZE=4, HAS_SMOOTHING=False, SPLIT=False),
    },
    {
        "name": "logit_scale_2blk",
        "params": dict(smoothing=0.0, logit_scale=0.5, lse_square_scale=0.05,
                        ignored_index=-1, total_classes=10, class_start_idx=0,
                        BLOCK_SIZE=4, HAS_SMOOTHING=False, SPLIT=False),
    },
]


def main():
    kernel_name = "cross_entropy"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_pass = True
    records = []

    # --- TileLang vs PyTorch ---
    print(f"\n{'='*60}")
    print(f"  Kernel (TileLang): {kernel_name}")
    print(f"{'='*60}")

    for cfg in test_configs:
        label = cfg["name"]
        params = cfg["params"]
        bs = params["BLOCK_SIZE"]
        is_split = params["SPLIT"]
        dtype = torch.float32

        try:
            ref_out = run_fwd(pytorch_fwd, logits, labels, **params)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            tl_out = run_fwd(tilelang_fwd, logits, labels, **params)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) * 1000

            passed, max_err = compare_valid(ref_out, tl_out, bs, n_rows, n_cols, is_split, dtype)

            if not passed:
                print(f"  FAIL  {label}  |  max_err={max_err:.2e}  tol={get_tol(dtype, True)}")
                all_pass = False
                records.append({"name": label, "status": "FAIL", "max_err": max_err})
            else:
                print(f"  PASS  {label}  |  max_err={max_err:.2e}  time={elapsed:.2f}ms")
                records.append({"name": label, "status": "PASS", "max_err": max_err, "time_ms": elapsed})
        except Exception:
            tb = traceback.format_exc()
            print(f"  ERROR {label}\n{tb}")
            all_pass = False
            records.append({"name": label, "status": "ERROR", "traceback": tb})

    result = {"kernel": kernel_name + "_tilelang", "overall": "PASS" if all_pass else "FAIL", "test_cases": records}
    out_path = RESULTS_DIR / f"{kernel_name}_tilelang.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nResult: {'PASS' if all_pass else 'FAIL'}  -> {out_path}")
    print(f"{'='*60}\n")
    sys.exit(0 if all_pass else 1)


main()

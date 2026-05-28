import torch
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor

try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
    from tuning.cache import get_best_config as _get_best_config
    _TUNED = _get_best_config("layer_norm", "triton") or {}
except Exception:
    _TUNED = {}


@triton.autotune(
    configs=[
        triton.Config({"XBLOCK": 1, "RBLOCK": 1024}, num_stages=1, num_warps=8),
        triton.Config({"XBLOCK": 1, "RBLOCK": 2048}, num_stages=1, num_warps=8),
    ],
    key=["xnumel", "rnumel"],
)
@triton.jit
def triton_red_fused_native_layer_norm_0(
    in_out_ptr0,
    in_ptr0,
    in_ptr1,
    in_ptr2,
    out_ptr0,
    out_ptr1,
    xnumel,
    rnumel,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp3_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(
            in_ptr0 + (r1 + (rnumel * x0)), rmask, eviction_policy="evict_last"
        ).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(rmask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(rmask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(rmask, tmp3_weight_next, tmp3_weight)
    tmp3_tmp, tmp4_tmp, tmp5_tmp = triton_helpers.welford(
        tmp3_mean, tmp3_m2, tmp3_weight, 1
    )
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp3, None)
    tmp6 = rnumel
    tmp7 = tmp4 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, None)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(
            in_ptr0 + (r1 + (rnumel * x0)), rmask, eviction_policy="evict_first"
        ).to(tl.float32)
        tmp15 = tl.load(in_ptr1 + (r1), rmask, eviction_policy="evict_last").to(
            tl.float32
        )
        tmp18 = tl.load(in_ptr2 + (r1), rmask, eviction_policy="evict_last").to(
            tl.float32
        )
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp12 - tmp3
        tmp14 = tmp13 * tmp10
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp14 * tmp16
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp17 + tmp19
        tmp21 = tmp20.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (rnumel * x0)), tmp21, rmask)


def _fused_native_layer_norm(weight, bias, x):
    """Internal: calls the Triton kernel with original argument order.
    Note: eps is hardcoded to 1e-5 in the kernel.
    """
    S, D = x.shape
    device = x.device
    device_idx = device.index if device.index is not None else 0
    with torch.cuda._DeviceGuard(device_idx):
        torch.cuda.set_device(device_idx)
        buf0 = torch.empty((S, 1), dtype=torch.float32, device=device)
        buf1 = torch.empty((S, 1), dtype=torch.float32, device=device)
        buf3 = reinterpret_tensor(buf1, (S, 1), (1, 1), 0)
        del buf1
        # Output dtype matches input dtype
        buf4 = torch.empty((S, D), dtype=x.dtype, device=device)
        stream0 = get_raw_stream(device_idx)
        grid = lambda META: (triton.cdiv(S, META["XBLOCK"]),)
        triton_red_fused_native_layer_norm_0[grid](
            buf3, x, weight, bias, buf0, buf4, S, D
        )
    return buf4, x, buf0, buf3


def layer_norm(x, weight, bias, eps=1e-5):
    """Unified API: layer_norm(x, weight, bias, eps) -> Tensor
    Note: eps is fixed at 1e-5 in the Triton kernel.
    Supports ND input: normalizes over the last dimension.
    """
    if eps != 1e-5:
        raise ValueError(f"Only eps=1e-5 is supported (got {eps}). "
                         "The Triton kernel hardcodes eps=1e-5.")
    orig_shape = x.shape
    D = orig_shape[-1]
    # Flatten to 2D for the kernel
    x_2d = x.reshape(-1, D).contiguous()
    result_tuple = _fused_native_layer_norm(weight, bias, x_2d)
    return result_tuple[0].reshape(orig_shape)

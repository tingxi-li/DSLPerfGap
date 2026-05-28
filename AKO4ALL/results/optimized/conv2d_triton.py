import torch
import torch.nn as nn
import triton
import triton.language as tl


# Best approach: padded input + flat K implicit GEMM + pre-transposed weight
# Additional: more varied autotune configs, 3D grid

@triton.autotune(
    configs=[
        # Vary BLOCK_HW
        triton.Config({'BLOCK_HW': 64, 'BLOCK_OC': 64, 'BLOCK_K': 32, 'GROUP_SIZE_HW': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 64, 'BLOCK_OC': 64, 'BLOCK_K': 64, 'GROUP_SIZE_HW': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 64, 'BLOCK_OC': 128, 'BLOCK_K': 32, 'GROUP_SIZE_HW': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_OC': 64, 'BLOCK_K': 32, 'GROUP_SIZE_HW': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_OC': 128, 'BLOCK_K': 32, 'GROUP_SIZE_HW': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_OC': 128, 'BLOCK_K': 32, 'GROUP_SIZE_HW': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_OC': 128, 'BLOCK_K': 32, 'GROUP_SIZE_HW': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_OC': 128, 'BLOCK_K': 64, 'GROUP_SIZE_HW': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_OC': 64, 'BLOCK_K': 64, 'GROUP_SIZE_HW': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_OC': 64, 'BLOCK_K': 32, 'GROUP_SIZE_HW': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_OC': 128, 'BLOCK_K': 32, 'GROUP_SIZE_HW': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_OC': 256, 'BLOCK_K': 32, 'GROUP_SIZE_HW': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_OC': 128, 'BLOCK_K': 32, 'GROUP_SIZE_HW': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_OC': 128, 'BLOCK_K': 32, 'GROUP_SIZE_HW': 16}, num_warps=4, num_stages=3),
        # Small K block for K=2304
        triton.Config({'BLOCK_HW': 128, 'BLOCK_OC': 128, 'BLOCK_K': 16, 'GROUP_SIZE_HW': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_OC': 64, 'BLOCK_K': 64, 'GROUP_SIZE_HW': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_OC': 128, 'BLOCK_K': 64, 'GROUP_SIZE_HW': 8}, num_warps=8, num_stages=2),
    ],
    key=['HW', 'OC', 'K'],
)
@triton.jit
def conv2d_padded_kernel(
    input_ptr, weight_ptr, output_ptr,
    HW, OC, K,
    in_channels,
    out_h, out_w,
    kh: tl.constexpr, kw: tl.constexpr,
    stride_h: tl.constexpr, stride_w: tl.constexpr,
    inp_stride_n, inp_stride_c, inp_stride_h, inp_stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
    w_stride_k, w_stride_n,
    BLOCK_HW: tl.constexpr, BLOCK_OC: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_id = tl.program_id(2)

    num_pid_hw = tl.cdiv(HW, BLOCK_HW)
    num_pid_oc = tl.cdiv(OC, BLOCK_OC)

    num_pid_in_group = GROUP_SIZE_HW * num_pid_oc
    group_id = pid // num_pid_in_group
    first_pid_hw = group_id * GROUP_SIZE_HW
    group_size_hw = min(num_pid_hw - first_pid_hw, GROUP_SIZE_HW)
    pid_hw = first_pid_hw + (pid % group_size_hw)
    pid_oc = (pid % num_pid_in_group) // group_size_hw

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)

    oh_idx = offs_hw // out_w
    ow_idx = offs_hw % out_w

    mask_hw = offs_hw < HW
    mask_oc = offs_oc < OC

    acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

    kh_kw: tl.constexpr = kh * kw
    offs_k = tl.arange(0, BLOCK_K)

    inp_base = input_ptr + batch_id * inp_stride_n

    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k

        ic = k_idx // kh_kw
        k_rem = k_idx % kh_kw
        kh_pos = k_rem // kw
        kw_pos = k_rem % kw

        ih_idx = oh_idx[:, None] * stride_h + kh_pos[None, :]
        iw_idx = ow_idx[:, None] * stride_w + kw_pos[None, :]

        mask_k = k_idx < K

        inp_ptrs = (inp_base +
                    ic[None, :] * inp_stride_c +
                    ih_idx * inp_stride_h +
                    iw_idx * inp_stride_w)
        a = tl.load(inp_ptrs, mask=mask_hw[:, None] & mask_k[None, :], other=0.0).to(tl.float16)

        w_ptrs = weight_ptr + k_idx[:, None] * w_stride_k + offs_oc[None, :] * w_stride_n
        b = tl.load(w_ptrs, mask=mask_k[:, None] & mask_oc[None, :], other=0.0).to(tl.float16)

        acc = tl.dot(a, b, acc)

    out_ptrs = (output_ptr +
                batch_id * out_stride_n +
                offs_oc[None, :] * out_stride_c +
                oh_idx[:, None] * out_stride_h +
                ow_idx[:, None] * out_stride_w)
    tl.store(out_ptrs, acc.to(output_ptr.dtype.element_ty), mask=mask_hw[:, None] & mask_oc[None, :])


def conv2d(input, weight, bias=None, stride=1, padding=0, groups=1):
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    assert groups == 1, "Only groups=1 supported"

    input = input.contiguous()
    weight = weight.contiguous()

    batch, in_channels, in_h, in_w = input.shape
    out_channels, _, kh, kw = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding

    out_h = (in_h + 2 * pad_h - kh) // stride_h + 1
    out_w = (in_w + 2 * pad_w - kw) // stride_w + 1

    # Explicitly pad the input to eliminate spatial bounds checks
    if pad_h > 0 or pad_w > 0:
        input_padded = torch.nn.functional.pad(input, (pad_w, pad_w, pad_h, pad_h))
    else:
        input_padded = input

    output = torch.empty((batch, out_channels, out_h, out_w),
                         device=input.device, dtype=input.dtype)

    K = in_channels * kh * kw
    # Pre-transpose weight: (OC, IC*KH*KW) -> (IC*KH*KW, OC)
    weight_reshaped = weight.reshape(out_channels, K).t().contiguous()

    HW = out_h * out_w
    OC = out_channels

    grid = lambda META: (
        triton.cdiv(HW, META['BLOCK_HW']) * triton.cdiv(OC, META['BLOCK_OC']),
        1,
        batch,
    )

    conv2d_padded_kernel[grid](
        input_padded, weight_reshaped, output,
        HW, OC, K,
        in_channels,
        out_h, out_w,
        kh, kw,
        stride_h, stride_w,
        input_padded.stride(0), input_padded.stride(1), input_padded.stride(2), input_padded.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        weight_reshaped.stride(0), weight_reshaped.stride(1),
    )

    if bias is not None:
        output += bias.view(1, -1, 1, 1)

    return output


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        return conv2d(*args)


def get_inputs():
    x = torch.randn(32, 256, 128, 128, device='cuda', dtype=torch.float16)
    w = torch.randn(256, 256, 3, 3, device='cuda', dtype=torch.float16)
    return [x, w]

def get_init_inputs():
    return []

import torch
import triton
import triton.language as tl


@triton.jit
def conv2d_forward_kernel(
    input_pointer, weight_pointer, output_pointer,
    batch_dim, in_feat_dim, in_height, in_width,
    out_feat_dim, out_height, out_width,
    input_batch_stride, input_in_feat_stride, input_height_stride, input_width_stride,
    weight_out_feat_stride, weight_in_feat_stride, weight_height_stride, weight_width_stride,
    output_batch_stride, output_out_feat_stride, output_height_stride, output_width_stride,
    kernel_height: tl.constexpr, kernel_width: tl.constexpr,
    stride_height: tl.constexpr, stride_width: tl.constexpr,
    padding_height: tl.constexpr, padding_width: tl.constexpr,
    groups: tl.constexpr, fp16: tl.constexpr, tf32: tl.constexpr,
    BLOCK_SIZE_BATCH_HEIGHT_WIDTH: tl.constexpr, BLOCK_SIZE_IN_FEAT: tl.constexpr,
    BLOCK_SIZE_OUT_FEAT: tl.constexpr,
    ):
    batch_height_width_pid = tl.program_id(0)
    out_feat_pid = tl.program_id(1)
    group_pid = tl.program_id(2)

    in_group_dim = in_feat_dim // groups
    out_group_dim = out_feat_dim // groups

    batch_height_width_offset = (batch_height_width_pid * BLOCK_SIZE_BATCH_HEIGHT_WIDTH +
                                 tl.arange(0, BLOCK_SIZE_BATCH_HEIGHT_WIDTH))
    batch_height_offset = batch_height_width_offset // out_width
    batch_offset = batch_height_offset // out_height

    output_feat_offset = (out_feat_pid * BLOCK_SIZE_OUT_FEAT +
                          tl.arange(0, BLOCK_SIZE_OUT_FEAT))
    output_height_offset = batch_height_offset % out_height
    output_width_offset = batch_height_width_offset % out_width

    input_pointer += (input_batch_stride * batch_offset +
                      input_in_feat_stride * group_pid * in_group_dim)[:, None]
    weight_pointer += (weight_out_feat_stride * output_feat_offset +
                       weight_out_feat_stride * group_pid * out_group_dim)[None, :]

    accum = tl.zeros((BLOCK_SIZE_BATCH_HEIGHT_WIDTH, BLOCK_SIZE_OUT_FEAT),
                     dtype=tl.float32)

    for h in range(kernel_height):
        for w in range(kernel_width):
            for c in range(0, in_group_dim, BLOCK_SIZE_IN_FEAT):
                input_feat_offset = c + tl.arange(0, BLOCK_SIZE_IN_FEAT)
                input_height_offset = (h - padding_height +
                                       stride_height * output_height_offset)
                input_width_offset = (w - padding_width +
                                      stride_width * output_width_offset)

                curr_input_pointer = (input_pointer +
                                     (input_in_feat_stride * input_feat_offset)[None, :] +
                                     (input_height_stride * input_height_offset)[:, None] +
                                     (input_width_stride * input_width_offset)[:, None])
                curr_weight_pointer = (weight_pointer +
                                      (weight_in_feat_stride * input_feat_offset)[:, None] +
                                      (weight_height_stride * h) +
                                      (weight_width_stride * w))

                input_mask = ((batch_offset < batch_dim)[:, None] &
                              (input_feat_offset < in_group_dim)[None, :] &
                              (0 <= input_height_offset)[:, None] &
                              (input_height_offset < in_height)[:, None] &
                              (0 <= input_width_offset)[:, None] &
                              (input_width_offset < in_width)[:, None])
                weight_mask = ((input_feat_offset < in_group_dim)[:, None] &
                               (output_feat_offset < out_group_dim)[None, :])

                input_block = tl.load(curr_input_pointer, mask=input_mask)
                weight_block = tl.load(curr_weight_pointer, mask=weight_mask)

                if fp16:
                    input_block = input_block.to(tl.float16)
                    weight_block = weight_block.to(tl.float16)

                accum += tl.dot(input_block, weight_block, allow_tf32=tf32)

    output_pointer += ((output_batch_stride * batch_offset)[:, None] +
                       (output_out_feat_stride * (group_pid * out_group_dim + output_feat_offset))[None, :] +
                       (output_height_stride * output_height_offset)[:, None] +
                       (output_width_stride * output_width_offset)[:, None])
    output_mask = ((batch_offset < batch_dim)[:, None] &
                   (output_feat_offset < out_group_dim)[None, :] &
                   (output_height_offset < out_height)[:, None] &
                   (output_width_offset < out_width)[:, None])

    tl.store(output_pointer, accum, mask=output_mask)


def conv2d(input, weight, bias=None, stride=1, padding=0, groups=1):
    """
    2D convolution using Triton. Note: dilation is not supported (always 1).
    """
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    input = input.contiguous()
    weight = weight.contiguous()

    batch_dim, in_feat_dim, in_height, in_width = input.shape
    out_feat_dim = weight.shape[0]
    kernel_height, kernel_width = weight.shape[2], weight.shape[3]
    stride_height, stride_width = stride
    padding_height, padding_width = padding

    out_height = (in_height + 2 * padding_height - kernel_height) // stride_height + 1
    out_width = (in_width + 2 * padding_width - kernel_width) // stride_width + 1

    output = torch.empty((batch_dim, out_feat_dim, out_height, out_width),
                         device=input.device, dtype=input.dtype)

    input_batch_stride, input_in_feat_stride, input_height_stride, input_width_stride = input.stride()
    weight_out_feat_stride, weight_in_feat_stride, weight_height_stride, weight_width_stride = weight.stride()
    output_batch_stride, output_out_feat_stride, output_height_stride, output_width_stride = output.stride()

    BLOCK_SIZE_BATCH_HEIGHT_WIDTH = 128
    BLOCK_SIZE_IN_FEAT = 32
    BLOCK_SIZE_OUT_FEAT = 32

    grid = (triton.cdiv(batch_dim * out_height * out_width, BLOCK_SIZE_BATCH_HEIGHT_WIDTH),
            triton.cdiv(out_feat_dim // groups, BLOCK_SIZE_OUT_FEAT),
            groups)

    conv2d_forward_kernel[grid](
        input, weight, output,
        batch_dim, in_feat_dim, in_height, in_width,
        out_feat_dim, out_height, out_width,
        input_batch_stride, input_in_feat_stride, input_height_stride, input_width_stride,
        weight_out_feat_stride, weight_in_feat_stride, weight_height_stride, weight_width_stride,
        output_batch_stride, output_out_feat_stride, output_height_stride, output_width_stride,
        kernel_height, kernel_width,
        stride_height, stride_width,
        padding_height, padding_width,
        groups, False, False,
        BLOCK_SIZE_BATCH_HEIGHT_WIDTH, BLOCK_SIZE_IN_FEAT, BLOCK_SIZE_OUT_FEAT
    )

    if bias is not None:
        output += bias.view(1, -1, 1, 1)

    return output

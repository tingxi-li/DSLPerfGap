import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from pytorch_impl import index_select as pytorch_index_select
from triton_impl import index_select as triton_index_select
from tilelang_impl import index_select as tilelang_index_select
from test_utils import run_test, run_tilelang_test


def make_test(shape, num_indices, dtype):
    source = torch.randn(shape, device='cuda', dtype=dtype)
    num_rows = shape[0]
    index = torch.randint(0, num_rows, (num_indices,), device='cuda')
    return source, index


def _output_shape(source, index):
    return (index.shape[0],) + source.shape[1:]


def pytorch_fn(source, index):
    out_shape = _output_shape(source, index)
    output = torch.empty(out_shape, device='cuda', dtype=source.dtype)
    return pytorch_index_select(output, source, index)


def triton_fn(source, index):
    out_shape = _output_shape(source, index)
    output = torch.empty(out_shape, device='cuda', dtype=source.dtype)
    return triton_index_select(output, source, index)


if __name__ == "__main__":
    test_cases = [
        {"name": "small_fp32", "inputs": make_test((16, 64), 8, torch.float32), "dtype": torch.float32},
        {"name": "medium_fp32", "inputs": make_test((256, 512), 128, torch.float32), "dtype": torch.float32},
        {"name": "large_fp32", "inputs": make_test((1024, 1024), 512, torch.float32), "dtype": torch.float32},
        {"name": "single_index_fp32", "inputs": make_test((100, 256), 1, torch.float32), "dtype": torch.float32},
        {"name": "small_fp16", "inputs": make_test((16, 64), 8, torch.float16), "dtype": torch.float16},
        {"name": "3d_source_fp32", "inputs": make_test((32, 8, 16), 10, torch.float32), "dtype": torch.float32},
        {"name": "odd_shape_fp32", "inputs": make_test((37, 73), 15, torch.float32), "dtype": torch.float32},
    ]
    def tilelang_fn(source, index):
        out_shape = _output_shape(source, index)
        output = torch.empty(out_shape, device='cuda', dtype=source.dtype)
        return tilelang_index_select(output, source, index)

    run_tilelang_test("index_select", test_cases, pytorch_fn, tilelang_fn)

"""Auto-tuning config grids for all ViperBench kernels."""

# ── Triton configs ─────────────────────────────────────────────

TRITON_CONFIGS = {
    "add": [
        {"BLOCK_SIZE": bs} for bs in [256, 512, 1024, 2048, 4096, 8192]
    ],
    "mul": [
        {"BLOCK_SIZE": bs} for bs in [256, 512, 1024, 2048, 4096, 8192]
    ],
    "relu": [
        {"BLOCK_SIZE": bs} for bs in [256, 512, 1024, 2048, 4096, 8192]
    ],
    "argmax": [
        {"BLOCK_M": bm, "BLOCK_N": bn}
        for bm in [64, 128, 256] for bn in [64, 128, 256]
    ],
    "matmul": [
        {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk}
        for bm in [32, 64, 128] for bn in [32, 64, 128] for bk in [32, 64]
        if bm * bn <= 128 * 128
    ][:12],
    "leaky_relu": [
        {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk, "GROUP_SIZE_M": gm}
        for bm in [32, 64, 128] for bn in [32, 64, 128] for bk in [32, 64]
        for gm in [4, 8]
        if bm * bn <= 128 * 128
    ][:12],
    "batched_matmul": [
        {"block_m": bm, "block_n": bn, "block_k": bk}
        for bm in [8, 16, 32] for bn in [16, 32, 64] for bk in [32, 64, 128]
    ][:12],
    "conv2d": [
        {"BLOCK_SIZE_BATCH_HEIGHT_WIDTH": bw, "BLOCK_SIZE_IN_FEAT": bf, "BLOCK_SIZE_OUT_FEAT": bo}
        for bw in [64, 128, 256] for bf in [16, 32, 64] for bo in [16, 32, 64]
        if bf * bo <= 64 * 64
    ][:12],
    "embedding": [
        {"BLOCK_N": bn, "num_warps": nw}
        for bn in [32, 64, 128] for nw in [1, 2, 4]
    ],
    "index_select": [
        {"BLOCK_SIZE_COL": bc} for bc in [128, 256, 512, 1024, 2048]
    ],
    "matrix_transpose": [
        {"BLOCK_ROWS": br, "BLOCK_COLS": bc}
        for br in [16, 32, 64] for bc in [16, 32, 64]
    ],
    "mean_reduction": [
        {"BLOCK_M": bm, "BLOCK_N": bn}
        for bm in [4, 8, 16, 32] for bn in [4, 8, 16, 32]
    ][:12],
    "softmax": [
        {"num_warps": nw} for nw in [1, 2, 4, 8, 16]
    ],
    "rms_norm": [
        {"num_warps": nw} for nw in [1, 2, 4, 8, 16]
    ],
    "attention": [
        {"BT": bt, "num_warps": nw}
        for bt in [16, 32, 64] for nw in [2, 4, 8]
    ],
    "linear_activation": [
        {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk}
        for bm in [16, 32, 64] for bn in [16, 32, 64] for bk in [32, 64]
        if bm * bn <= 64 * 64
    ][:12],
    "cross_entropy": [
        {"BLOCK_SIZE": bs} for bs in [128, 256, 512, 1024, 2048]
    ],
    "log_softmax": [
        {"BLOCK_M": bm} for bm in [1, 2, 4, 8, 16]
    ],
    "logsumexp": [
        {"num_warps": nw} for nw in [1, 2, 4, 8, 16, 32]
    ],
    "swiglu": [
        {"BLOCK_N": bn} for bn in [32, 64, 128, 256, 512, 1024]
    ],
    "max_reduction": [
        {"BLOCK_M": bm} for bm in [4, 8, 16, 32, 64]
    ],
    "layer_norm": [
        {"RBLOCK": rb} for rb in [512, 1024, 2048, 4096]
    ],
}

# ── TileLang configs ───────────────────────────────────────────

TILELANG_CONFIGS = {
    "add": [
        {"block_N": bn, "threads": th}
        for bn in [256, 512, 1024, 2048] for th in [128, 256]
    ],
    "mul": [
        {"block_N": bn, "threads": th}
        for bn in [256, 512, 1024, 2048] for th in [128, 256]
    ],
    "relu": [
        {"block_N": bn, "threads": th}
        for bn in [256, 512, 1024, 2048] for th in [128, 256]
    ],
    "matmul": [
        {"block_M": bm, "block_N": bn, "block_K": bk, "num_stages": ns}
        for bm in [64, 128] for bn in [64, 128] for bk in [16, 32, 64]
        for ns in [2, 3]
        if bm * bn <= 128 * 128
    ][:16],
    "leaky_relu": [
        {"block_M": bm, "block_N": bn, "block_K": bk, "num_stages": ns}
        for bm in [64, 128] for bn in [64, 128] for bk in [16, 32, 64]
        for ns in [2, 3]
        if bm * bn <= 128 * 128
    ][:16],
    "linear_activation": [
        {"block_M": bm, "block_N": bn, "block_K": bk, "num_stages": ns}
        for bm in [64, 128] for bn in [64, 128] for bk in [16, 32, 64]
        for ns in [2, 3]
        if bm * bn <= 128 * 128
    ][:16],
    "conv2d": [
        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "NUM_STAGES": ns}
        for bm in [32, 64, 128] for bn in [32, 64, 128] for bk in [16, 32]
        for ns in [2, 3]
        if bm * bn <= 128 * 128
    ][:16],
    "batched_matmul": [
        {"block_N": bn, "block_K": bk, "threads": th}
        for bn in [32, 64, 128] for bk in [32, 64, 128] for th in [128, 256]
    ][:12],
    "softmax": [
        {"threads": th} for th in [32, 64, 128, 256]
    ],
    "log_softmax": [
        {"threads": th} for th in [32, 64, 128, 256]
    ],
    "logsumexp": [
        {"block_M": bm, "threads": th}
        for bm in [2, 4, 8, 16, 32] for th in [128, 256]
    ][:10],
    "layer_norm": [
        {"threads": th} for th in [32, 64, 128, 256]
    ],
    "rms_norm": [
        {"threads": th} for th in [32, 64, 128, 256]
    ],
    "argmax": [
        {"block_M": bm, "block_K": bk, "threads": th}
        for bm in [16, 32, 64] for bk in [16, 32, 64] for th in [128, 256]
    ][:12],
    "max_reduction": [
        {"block_M": bm, "block_K": bk, "threads": th}
        for bm in [16, 32, 64] for bk in [16, 32, 64] for th in [128, 256]
    ][:12],
    "mean_reduction": [
        {"block_M": bm, "block_K": bk, "threads": th}
        for bm in [16, 32, 64] for bk in [16, 32, 64] for th in [128, 256]
    ][:12],
    "matrix_transpose": [
        {"block_M": bm, "block_N": bn, "threads": th}
        for bm in [32, 64, 128] for bn in [32, 64, 128] for th in [128, 256]
    ][:12],
    "embedding": [
        {"block_N": bn, "block_D": bd, "threads": th}
        for bn in [2, 4, 8] for bd in [64, 128, 256] for th in [128, 256]
    ][:12],
    "index_select": [
        {"block_N": bn, "threads": th}
        for bn in [16, 32, 64, 128] for th in [128, 256]
    ],
    "swiglu": [
        {"block_M": bm, "threads": th}
        for bm in [8, 16, 32, 64] for th in [128, 256]
    ],
    "cross_entropy": [
        {"threads": th} for th in [32, 64, 128, 256]
    ],
    "attention": [
        {"block_M": bm, "block_N": bn, "block_K": bk, "threads": th}
        for bm in [16, 32, 64] for bn in [16, 32, 64] for bk in [16, 32]
        for th in [128]
        if bm * bn <= 64 * 64
    ][:12],
}

"""
Generate figures/overview_efficiency.pdf — Figure 1 of the paper.

Library efficiency (%) = t_lib / t_DSL * 100 for all benchmark kernels,
grouped by kernel category and DSL (Triton vs. TileLang).

Data hard-coded from evaluation tables in tex/evaluation.tex.
All values below are the NVIDIA A100-SXM4-40GB per-kernel E_lib figures
(re-derived for the A100 revision; see AUTHORITATIVE_NUMBERS.md).
Regenerate the PDF after any table edit with: python figures/gen_overview.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Data (E_lib % values from evaluation tables)
# Attention excluded: non-equivalent baselines (see sec:eval:gemm)
#
# "fair" = apples-to-apples DSL vs. library comparison
# "unfair_outliers" = super-parity point where PyTorch baseline is unfused;
#                     plotted separately so it does not distort the box.
# ---------------------------------------------------------------------------
fair = {
    # GEMM: 16384^2 square, batched 64x128^2, batched 128x2048^2, fused linear+act
    ("GEMM",         "Triton"):   [32.8, 92.7, 92.9, 72.5],
    ("GEMM",         "TileLang"): [80.0, 7.0, 548.6],
    # Conv2d: 8x64x56x56 3x3, 32x256x128x128 3x3
    ("Conv2d",       "Triton"):   [44.87, 19.33],
    ("Conv2d",       "TileLang"): [4.12,  12.62],
    # Normalization: LayerNorm only (RMSNorm 851.8% is an unfused-baseline artifact)
    ("Norm.",        "Triton"):   [88.7],
    ("Norm.",        "TileLang"): [0.1, 0.9],
    # Element-wise: relu, add
    ("Element-wise", "Triton"):   [95.4, 89.5],
    ("Element-wise", "TileLang"): [68.3,  74.5],
}

# Rendered as hollow-diamond markers (unfair-baseline outlier)
unfair_outliers = {
    ("Norm.", "Triton"): 851.8,   # RMSNorm: PyTorch eager path is unfused
}

CATEGORIES = ["GEMM", "Conv2d", "Norm.", "Element-wise"]
DSLS       = ["Triton", "TileLang"]

# Colors: Paul Tol colorblind-safe pair
COLORS = {"Triton": "#4477AA", "TileLang": "#EE7733"}

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         8,
    "axes.titlesize":    8,
    "axes.labelsize":    8,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "legend.fontsize":   7,
    "pdf.fonttype":      42,   # embed TrueType (required by ACM)
    "ps.fonttype":       42,
})

fig, ax = plt.subplots(figsize=(3.3, 2.6))

# ---------------------------------------------------------------------------
# Layout: 4 categories, 2 DSLs each, small gap between DSLs within a group
# ---------------------------------------------------------------------------
n_cat   = len(CATEGORIES)
n_dsl   = len(DSLS)
group_w = 0.7          # total width allocated per category
box_w   = group_w / (n_dsl + 0.5)   # ~0.28
offsets = np.linspace(-group_w / 2 + box_w / 2,
                       group_w / 2 - box_w / 2,
                       n_dsl)         # symmetric offsets for the two DSLs

x_centers = np.arange(n_cat, dtype=float)

# Categories with enough comparable points for a boxplot
BOX_CATS = {"GEMM"}

# Per-category, per-DSL: draw box or scatter
for ci, cat in enumerate(CATEGORIES):
    for di, dsl in enumerate(DSLS):
        pos   = x_centers[ci] + offsets[di]
        vals  = fair[(cat, dsl)]
        color = COLORS[dsl]

        if cat in BOX_CATS:
            bp = ax.boxplot(
                [vals],
                positions=[pos],
                widths=box_w * 0.85,
                patch_artist=True,
                showfliers=True,
                medianprops=dict(color="black", linewidth=1.2),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
                flierprops=dict(marker="o", markersize=3, linestyle="none",
                                markerfacecolor=color, markeredgecolor=color),
            )
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(0.75)
        else:
            # Scatter dots for small-n categories (n=1 or n=2)
            jitter = np.linspace(-0.04, 0.04, len(vals))
            ax.scatter(
                [pos + j for j in jitter],
                vals,
                color=color,
                alpha=0.85,
                s=18,
                zorder=3,
            )

# Unfair-baseline outliers: hollow diamond, visually distinct from normal fliers
for (cat, dsl), val in unfair_outliers.items():
    ci  = CATEGORIES.index(cat)
    di  = DSLS.index(dsl)
    pos = x_centers[ci] + offsets[di]
    ax.scatter(
        [pos], [val],
        marker="D",
        s=22,
        facecolors="none",
        edgecolors=COLORS[dsl],
        linewidths=1.0,
        zorder=4,
    )

# ---------------------------------------------------------------------------
# Log scale y-axis
# ---------------------------------------------------------------------------
ax.set_yscale("log")
ax.set_ylim(0.15, 3000)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda y, _: f"{y:g}%"
))
ax.set_yticks([0.3, 1, 3, 10, 30, 100, 300, 1000])

# ---------------------------------------------------------------------------
# Parity line at 100 %
# ---------------------------------------------------------------------------
ax.axhline(100, color="gray", linestyle="--", linewidth=0.8, zorder=0)
ax.text(n_cat - 0.05, 100 * 1.08, "parity", ha="right", va="bottom",
        fontsize=6, color="gray")

# ---------------------------------------------------------------------------
# Annotate super-parity outliers so readers can read log values
# ---------------------------------------------------------------------------
# Triton RMSNorm = 1099 % (fusion outlier) — annotate to the left to stay clear of legend
norm_idx   = CATEGORIES.index("Norm.")
triton_idx = DSLS.index("Triton")
x_rms = x_centers[norm_idx] + offsets[triton_idx]
ax.annotate(
    "852%\n(unfused\nbaseline)",
    xy=(x_rms, 851.8),
    xytext=(x_rms - 0.9, 550),
    fontsize=5.5,
    color=COLORS["Triton"],
    arrowprops=dict(arrowstyle="->", lw=0.6, color=COLORS["Triton"]),
    ha="center", va="center",
)

# TileLang fused linear+activation = 548.6 % (GEMM category) — annotate below the flier
gemm_idx     = CATEGORIES.index("GEMM")
tilelang_idx = DSLS.index("TileLang")
x_fused = x_centers[gemm_idx] + offsets[tilelang_idx]
ax.annotate(
    "549%\n(kernel\nfusion)",
    xy=(x_fused, 548.6),
    xytext=(x_fused - 0.55, 250),
    fontsize=5.5,
    color=COLORS["TileLang"],
    arrowprops=dict(arrowstyle="->", lw=0.6, color=COLORS["TileLang"]),
    ha="center", va="center",
)

# ---------------------------------------------------------------------------
# Axes labels and ticks
# ---------------------------------------------------------------------------
ax.set_xticks(x_centers)
ax.set_xticklabels(CATEGORIES, rotation=0)
ax.set_xlim(-0.6, n_cat - 0.4)
ax.set_ylabel("Library efficiency (%)")
ax.set_xlabel("Kernel category")

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
legend_patches = [
    mpatches.Patch(facecolor=COLORS[dsl], alpha=0.75, label=dsl)
    for dsl in DSLS
]
ax.legend(handles=legend_patches, loc="lower right", framealpha=0.85,
          borderpad=0.4, handlelength=1.0)

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
fig.tight_layout(pad=0.4)
out_path = os.path.join(os.path.dirname(__file__), "overview_efficiency.pdf")
fig.savefig(out_path, bbox_inches="tight")
print(f"Saved: {out_path}")

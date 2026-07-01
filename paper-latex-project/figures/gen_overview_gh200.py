"""
Generate figures/overview_efficiency_gh200.pdf — GH200 counterpart to Figure 1.

Library efficiency (%) = t_lib / t_DSL * 100 for all benchmark kernels,
grouped by kernel category and DSL (Triton vs. TileLang).

Data sourced from ViperBench/results/profile.GH200-480GB.csv (large-size rows),
E_lib = pytorch_ms / dsl_ms * 100.
Regenerate the PDF after any table edit with: python figures/gen_overview_gh200.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Data — GH200 large-size E_lib (%) from profile.GH200-480GB.csv
# Attention excluded: non-equivalent baselines
#
# "fair" = apples-to-apples DSL vs. library comparison
# "unfair_outliers" = super-parity points where PyTorch baseline is unfused;
#                     plotted as hollow diamonds so they don't distort the box.
# ---------------------------------------------------------------------------
fair = {
    # GEMM: matmul 16384^2, batched_matmul 128x2048^2, fused linear+activation
    ("GEMM",         "Triton"):   [56.4, 65.6, 169.9],
    ("GEMM",         "TileLang"): [59.7, 110.4, 163.0],
    # Conv2d: small 8x64x56x56 3x3, large 32x256x128x128 3x3
    ("Conv2d",       "Triton"):   [76.7, 13.8],
    ("Conv2d",       "TileLang"): [8.3, 59.9],
    # Normalization: LayerNorm only (RMSNorm 1081.5% is an unfused-baseline artifact)
    ("Norm.",        "Triton"):   [74.8],
    ("Norm.",        "TileLang"): [0.3, 3.3],
    # Element-wise: relu, add
    ("Element-wise", "Triton"):   [97.8, 89.2],
    ("Element-wise", "TileLang"): [66.1, 67.8],
}

# Rendered as hollow-diamond markers (unfair-baseline outlier)
unfair_outliers = {
    ("Norm.", "Triton"): 1081.5,   # RMSNorm: PyTorch eager path is unfused
}

CATEGORIES = ["GEMM", "Conv2d", "Norm.", "Element-wise"]
DSLS       = ["Triton", "TileLang"]

# Colors: Paul Tol colorblind-safe pair (same as A100 figure)
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
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

fig, ax = plt.subplots(figsize=(3.3, 2.6))

# ---------------------------------------------------------------------------
# Layout: 4 categories, 2 DSLs each, small gap between DSLs within a group
# ---------------------------------------------------------------------------
n_cat   = len(CATEGORIES)
n_dsl   = len(DSLS)
group_w = 0.7
box_w   = group_w / (n_dsl + 0.5)
offsets = np.linspace(-group_w / 2 + box_w / 2,
                       group_w / 2 - box_w / 2,
                       n_dsl)

x_centers = np.arange(n_cat, dtype=float)

BOX_CATS = {"GEMM"}

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
            jitter = np.linspace(-0.04, 0.04, len(vals))
            ax.scatter(
                [pos + j for j in jitter],
                vals,
                color=color,
                alpha=0.85,
                s=18,
                zorder=3,
            )

# Unfair-baseline outliers: hollow diamond
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
# Annotate super-parity outlier
# ---------------------------------------------------------------------------
norm_idx   = CATEGORIES.index("Norm.")
triton_idx = DSLS.index("Triton")
x_rms = x_centers[norm_idx] + offsets[triton_idx]
ax.annotate(
    "1082%\n(unfused\nbaseline)",
    xy=(x_rms, 1081.5),
    xytext=(x_rms - 0.9, 600),
    fontsize=5.5,
    color=COLORS["Triton"],
    arrowprops=dict(arrowstyle="->", lw=0.6, color=COLORS["Triton"]),
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
out_path = os.path.join(os.path.dirname(__file__), "overview_efficiency_gh200.pdf")
fig.savefig(out_path, bbox_inches="tight")
print(f"Saved: {out_path}")

#!/usr/bin/env python3
"""Consolidate ncu per-(target,set) CSVs into one RC-grounded table.

Reads experiments/results/<gpu_slug>/ncu/*.csv (produced by ncu_counters.sh),
extracts the key metric per target, and writes:
  * ncu_summary.csv   (tidy: target, kernel, metric, value, unit)
  * a human-readable table to stdout grouped by reviewer root-cause family.
Portable: same script consolidates A100/H100 runs (different slug dir).
"""
import csv
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _harness import device_slug  # noqa: E402

NCU_DIR = os.path.join(HERE, "results", device_slug(), "ncu")

# metric -> (short label, root-cause tag)
KEEP = {
    # (A) vectorized / global-load efficiency  -> RC1 / RC0b
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio":
        ("sectors/req (↑=less vectorized)", "RC1"),
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct":
        ("bytes/sector load eff %", "RC1"),
    # (B) registers + spill proxy + occupancy  -> RC3
    "launch__registers_per_thread": ("regs/thread", "RC3"),
    "launch__occupancy_limit_registers": ("occ limit (reg) %", "RC3"),
    "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum": ("local-ld bytes (spill)", "RC3"),
    "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum": ("local-st bytes (spill)", "RC3"),
    "sm__warps_active.avg.pct_of_peak_sustained_active": ("achieved occ %", "RC3"),
    # (C) warp-stall breakdown  -> RC0
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio":
        ("stall long_scoreboard", "RC0"),
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio":
        ("stall short_scoreboard", "RC0"),
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio":
        ("stall barrier (sync)", "RC0"),
    "smsp__average_warps_issue_stalled_membar_per_issue_active.ratio":
        ("stall membar", "RC0"),
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio":
        ("stall mio_throttle", "RC0"),
    # (D) L2 / DRAM  -> RC2b
    "lts__t_sector_hit_rate.pct": ("L2 hit %", "RC2b"),
    "dram__bytes_read.sum": ("DRAM read bytes", "RC2b"),
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": ("DRAM thru %", "RC2b"),
    "gpu__time_duration.sum": ("kernel time (ns)", "ref"),
}


def parse_csv(path):
    """Yield dict rows from an ncu --csv --log-file file (skip ==PROF== noise)."""
    with open(path) as f:
        lines = [ln for ln in f if not ln.startswith("==PROF==")]
    if not lines:
        return []
    rdr = csv.DictReader(lines)
    return list(rdr)


def target_of(fname):
    base = os.path.basename(fname)[:-4]          # drop .csv
    *target, tag = base.rsplit("_", 1)            # tag = loads|regs|stalls|l2dram
    return "_".join(target), tag


def main():
    files = sorted(glob.glob(os.path.join(NCU_DIR, "*.csv")))
    if not files:
        print(f"no CSVs in {NCU_DIR}", file=sys.stderr)
        return 1

    # target -> kernelname -> metric -> (value, unit)
    data = {}
    tidy = []
    for path in files:
        target, tag = target_of(path)
        for row in parse_csv(path):
            metric = row.get("Metric Name", "")
            if metric not in KEEP:
                continue
            kernel = row.get("Kernel Name", "?")
            val = row.get("Metric Value", "")
            unit = row.get("Metric Unit", "")
            data.setdefault(target, {}).setdefault(kernel, {})[metric] = (val, unit)
            tidy.append(dict(target=target, kernel=kernel, set=tag,
                             metric=metric, label=KEEP[metric][0],
                             rc=KEEP[metric][1], value=val, unit=unit))

    # tidy CSV
    out_csv = os.path.join(NCU_DIR, "..", "ncu_summary.csv")
    out_csv = os.path.abspath(out_csv)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target", "kernel", "set", "rc",
                                          "metric", "label", "value", "unit"])
        w.writeheader()
        w.writerows(tidy)
    print(f"-> wrote tidy summary: {out_csv}  ({len(tidy)} rows)\n")

    # readable table grouped by target
    order = ["matmul_pytorch_large", "matmul_triton_large", "conv2d_triton_large",
             "layer_norm_tilelang_large", "argmax_tilelang_large",
             "max_reduction_tilelang_large", "logsumexp_tilelang_opt_large"]
    targets = [t for t in order if t in data] + [t for t in data if t not in order]

    def g(target, metric):
        for kern, mm in data.get(target, {}).items():
            if metric in mm:
                return mm[metric][0]
        return "-"

    rows_def = [
        ("RC1  sectors/req",   "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio"),
        ("RC1  load-eff %",    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct"),
        ("RC3  regs/thread",   "launch__registers_per_thread"),
        ("RC3  achieved occ%", "sm__warps_active.avg.pct_of_peak_sustained_active"),
        ("RC3  spill ld B",    "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum"),
        ("RC3  spill st B",    "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum"),
        ("RC0  stall l_sb",    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"),
        ("RC0  stall barrier", "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio"),
        ("RC0  stall mio",     "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio"),
        ("RC2b L2 hit %",      "lts__t_sector_hit_rate.pct"),
        ("RC2b DRAM thru %",   "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
        ("ref  kern time ns",  "gpu__time_duration.sum"),
    ]

    short = {t: t.replace("_large", "").replace("_", "/") for t in targets}
    w0 = 20
    colw = 16
    print("kernel captured per target:")
    for t in targets:
        kn = ", ".join(data[t].keys())
        print(f"  {short[t]:<26} -> {kn}")
    print()
    header = " " * w0 + "".join(f"{short[t][:colw-1]:>{colw}}" for t in targets)
    print(header)
    print("-" * len(header))
    for label, metric in rows_def:
        line = f"{label:<{w0}}" + "".join(f"{g(t, metric)[:colw-1]:>{colw}}" for t in targets)
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())

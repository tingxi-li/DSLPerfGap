# Revision-Round TODO — ASE 2026 #4134

Action items for the **revision round** (after rebuttal acceptance), surfaced by the
pre-submission consistency audit (`CONSISTENCY_AUDIT.md` §2). These are **author decisions**,
not rebuttal-round fixes — the rebuttal already promises the revision will address them.
All locations were verified against the PDF and the codebase.

Severity: 🔴 reviewer-disprovable / must-fix · 🟠 should-fix · 🟡 minor.
Check off (`[x]`) as each is resolved in the revision.

---

## 🔴 1. Paper text asserts the 3 refuted mechanisms
- [ ] **RC0a** — change "sequential thread synchronization" framing → **memory-latency-bound**.
      Our counters: `barrier ≈ 0`, `long_scoreboard` dominates. Locations: abstract; §6 p5 L527-541;
      **Table 5** counter column "Thread sync count, warp stall"; the "log₂(256/32)=3 barriers" argument.
- [ ] **RC3** — conv K≥5 "register pressure / spill to local memory" is **mis-attributed**.
      Our counters: conv `n_spills = 0`; the 51 GB spill is in **TileLang layer_norm** (254 regs, occ 16.5%).
      Reassign RC3 to the right kernel. Locations: abstract; §6 p6 L688-702; **Table 5** row "Conv2d (K≥5)" / `sm__register_spill`.
- [ ] **RC4** — "remaining gap **primarily** missing Winograd" overstated.
      Our isolation: Winograd ≈ **2–3%**, not "primary." Locations: abstract; §7.4 "remaining 20% deficit attributed to absent Winograd."
- [ ] Sanity pass: re-read Table 5 end-to-end so every mechanism label matches the collected counters
      (`experiments/results/<gpu>/NCU_FINDINGS.md`). The corrected story is *stronger* (counter-grounded).

## 🔴 2. N1 — artifact doesn't reproduce the paper's stated conditions
Paper claims vs. code reality (verified by grep across `ViperBench/`):
- "NHWC layout" (§3.2 L281-282; Table 2 caption L581) — **no** `channels_last`/`memory_format` anywhere; conv2d does `input.contiguous()` → **NCHW** (`conv2d/triton_impl.py:111`).
- "allow_tf32=False" (§3.2 L281-282) — **no** global `torch.backends.cuda.matmul.allow_tf32` set (the `allow_tf32` hits are per-kernel Triton `tl.dot` flags, a different mechanism).
- "cudnn.benchmark=False" (§3.5 L341-342) — **no** `torch.backends.cudnn.benchmark` set anywhere.
- Entangled with **RC1** ("NHWC breaks LDG.128 alignment," p6 L614-617): the measured conv ran NCHW.

Pick one:
- [ ] **Option A (code):** add the flags + `channels_last`, re-validate all results, confirm RC1 still holds.
- [ ] **Option B (text):** soften §3.2/§3.5/Table-2 to the actual NCHW + default-flags setup, and re-examine the RC1 "NHWC" wording.

> ⚠️ Affects measured results — do not change benchmark code without re-running the suite.

## 🟠 3. Timing methodology mismatch
- [ ] Align §3.3 L297-301 text to the code: it says "CUDA events (`cudaEventRecord`/`cudaEventElapsedTime`)…
      **average**," but `benchmark.py:41-48` uses `time.perf_counter()` + `torch.cuda.synchronize()` and reports the
      **median** (`times.sort(); times[len//2]`). Simplest fix = correct the text (perf_counter + median); no re-measurement needed.

## 🟠 4. E_lib defined but never computed by committed code
- [ ] §3.4 defines `E_lib = t_library / t_DSL × 100%`, but no script emits it; table percentages are hand-derived.
      Add a small script that derives E_lib from `profile.csv` (cheap; also de-risks transcription errors).

## 🟡 5. Stray "21 kernels"
- [ ] §2.5 p2 L137-138 says "21 kernels"; everywhere else (and the code: 22 dirs) says 22. Fix the stray.

## 🟠 6. Locked-clock benchmarking standard + re-baseline decision
- [ ] **Adopt the locked-clock config as the benchmark standard** and document it in §3.3/§3.5:
      `sudo nvidia-smi -i 0 -pm 1 && -lmc 9001 && -lgc 1400` (graphics holds flat at 1410 MHz under the
      130 W cap; memory 9001 idle → deterministic 8551 under load). Verified in
      `experiments/results/<gpu>/clock_lock.txt`; run-to-run rel-std drops from the paper's 9% to **0.0–0.9%**.
- [ ] **Decide: re-baseline the paper's tables at locked clocks, or keep `profile.csv` + add the significance study?**
      Some locked E_lib differ from `profile.csv` (boost vs locked) — e.g. softmax 97.6%→95.2%, cross_entropy
      tilelang 86%→60%. `experiments/exp_significance.py` → `significance.csv` is the locked-clock evidence for the
      near-parity set; a full re-baseline would re-run `benchmark.py` under the lock.
- [ ] **Remove the Table 7 "9% clock variation" footnote** once clocks are locked (the variation is eliminated).
- [ ] Nuance to state: at <1% locked dispersion, even ~0.5pp gaps are *statistically* resolved (e.g. mean_reduction
      99.4%); distinguish **statistical** from **practical** significance (99.4% ≈ parity) so the bands aren't over-read.

---

### Done this session (rebuttal-round — for reference, not TODO)
- ✅ R2-Q4 mislabel on the Nsight-counters paragraph (`REBUTTAL_GAME_PLAN.md:74`).
- ✅ cross_entropy efficiency reconciled to `profile.csv` (Triton 1277% / TileLang 86%).
- ✅ 5 stale "regenerate profile.csv (15908ms)" instructions (CSV already patched).
- ✅ **Path A done** — clocks locked + near-parity set re-measured with dispersion (`exp_significance.py` →
  `significance.csv`, `clock_lock.txt`); rebuttal `L72` now literally true (locked clocks, dispersion reported,
  small gaps resolved as real). Numbers are revision material; rebuttal text stays qualitative.

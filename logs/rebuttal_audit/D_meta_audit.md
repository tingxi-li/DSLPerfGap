# Meta-Audit — ASE 2026 #4134 Rebuttal (independent adversarial review)

Auditor D. Read-only on everything except this file. I independently verified all five
high-stakes items against primary sources (reviews.txt; the PDF Tables 1–8 / §3.1–3.5 / §5 / §7.3;
the autotune/significance/ncu/winograd CSVs; the leaky_relu/conv2d/linear_activation/cross_entropy
impls; tuning/configs.py; the AKO4ALL optimized matmul kernel). Verdicts below are mine, not the
subagents'.

**Headline:** Agent A and Agent C are largely reliable. **Agent B's interpretation of the central
"94.6% vs 97.8%" item is WRONG and so is the coordinator's E1 framing if it follows B.** I overturn
both A's and B's readings and supply the verified-correct interpretation (neither agent had it right).
The tuning-shape finding (item 2) is confirmed exactly as Agent A states. The census (item 3) and the
config count (item 4) are confirmed. I would BLOCK E1-as-drafted, MODIFY E4, and MODIFY E2's §1-L63
consistency claim. Details follow.

---

## ITEM 1 — "94.6% vs 97.8%": THE TRUTH (I overturn BOTH A and B)

**Reviewer A's exact line (verbatim, `reviews.txt:50`):**
> "Table 7 notes a 9% variation in baseline performance across profiling runs for the same convolution
> setup, attributed to GPU clock fluctuations. It is not stated whether clocks were locked. Without
> that, small efficiency differences (e.g., 94.6% vs. 97.8%) may not be meaningful."

So A's "94.6% vs 97.8%" is offered as **an illustrative pair of two near-parity efficiencies** that a 9%
clock swing could blur. A does not say what kernels they are; the burden is on us to identify them.

**What the two numbers actually are (from the PDF — decisive):**
- **94.6% = Triton LayerNorm, Table 3 (p.6).** Table 3 "Normalization latency" row LayerNorm
  8192×8192 (BF16): PyTorch 0.870 ms, **Triton 0.920 ms, E_lib = 94.6%**, TileLang 273.3 ms,
  E_lib = 0.32%. Page-5 prose confirms: *"Triton's `layer_norm` achieves 94.6% of PyTorch throughput at
  the large shape (8192 × 8192)."* The TileLang LayerNorm in Table 3 is **0.32%**, not 94.6%.
- **97.8% = TileLang LayerNorm AFTER the RC0 fix, Table 6 (p.8).** Table 6 "Normalization … before and
  after RC0 correction", TileLang LayerNorm (BF16): Before 1090 ms, **After 0.89 ms, E_lib = 97.8%**.

**Therefore the single correct interpretation:** the pair is **Triton LayerNorm (94.6%, Table 3) vs
TileLang-LayerNorm-after-RC0-fix (97.8%, Table 6)** — two *different DSL backends'* LayerNorm numbers,
both in the normalization results, both near parity. It is NOT layer_norm-vs-softmax, and it is NOT "the
same kernel before/after."

**Agent A (report §3 / D3): WRONG.** A maps it to layer_norm(94.5%)/softmax(95.2%). Softmax is 95.2%,
not 97.8% — A correctly *noticed* the mismatch but mis-stated the right answer. The
`exp_significance.py:9` comment ("softmax (97.6%)") that A flags is ALSO wrong and is a stale artifact
comment; 97.8% has nothing to do with softmax.

**Agent B (report Task 2 / Task 4): ALSO WRONG.** B says 94.6% = "LayerNorm PRE-RC0-fix (Table 3)" and
97.8% = "LayerNorm POST-fix (Table 6) — the SAME kernel before/after mitigation." This is incorrect on
two counts: (a) the pre-fix LayerNorm in Table 3 is **TileLang 0.32%**, not 94.6% — 94.6% is the *Triton*
number, which was never broken and never mitigated; (b) they are therefore **not the same kernel before
vs after** — 94.6% is Triton (unmitigated, a different backend) and 97.8% is TileLang-after-fix. B got
the right tables (3 and 6) and the right operator (LayerNorm) but the wrong backends and the wrong
"before/after-same-kernel" story.

**Why this matters for the rebuttal (a problem NOBODY flagged):** the locked-clock artifact
(`significance.csv`) re-measures Triton LayerNorm at **94.46%** (clean match to 94.6%) but it does
**NOT** contain a 97.8% measurement. significance.csv runs the stock ViperBench TileLang LayerNorm
(`load_impl(... "tilelang")`), which is the **unmitigated** 0.32% kernel (row 3: tilelang layer_norm
e_lib = **0.32**), not the mitigated 97.8% one. So the rebuttal literally cannot say "that pair maps to
two kernels in our locked run" — only ONE of the two (the Triton 94.6%→94.46%) is reproduced under
locked clocks; the 97.8% post-fix TileLang number was never re-run with clocks locked.

**Verified correct wording for §3 (≈L227-229)** — re-anchor to the real identities and DO NOT claim both
were re-measured under locked clocks:

> Reviewer A's illustrative pair is the two near-parity normalization numbers — Triton LayerNorm
> (Table 3) and the post-RC0-fix TileLang LayerNorm (Table 6). Under locked clocks our re-measurement
> shows the Triton LayerNorm efficiency is stable to within run-to-run dispersion well under the
> paper's reported 9% boost-clock band, so a gap of this size is resolved as real rather than a clock
> artifact; the revision reports the dispersion alongside both table cells.

(If the authors want to keep it tight and number-free, even simpler: *"the small normalization
efficiencies Reviewer A cites … are now reported with locked-clock dispersion, so each resolves as real
or within noise."* The key fix is to STOP saying "layer_norm and softmax" and STOP implying both cells
were re-measured.)

**Confirmed-correct as-is:** the coordinator's note that §1 To-A #7 (L112) and §2 W8 (L190) "stay
generic" is RIGHT. L112 says only "small efficiency differences are resolved as either real or within
noise" (number-free, accurate). The §2 W8 row carries "94.6% vs 97.8%" *only inside the reviewer-quote
column* and "0.0–0.9% vs the paper's 9%" in the evidence column — both legitimate in a mapping table.
**Only §3's prose is wrong, and it is wrong on FACTS, not on number-leakage.**

---

## ITEM 2 — TUNING SHAPE (Agent A's D1): CONFIRMED exactly; one extra wrinkle for §1-L63

**`autotune_matmul.csv` (verbatim rows):**
- 4096×4096: `triton_plain` e_vs_cublas = **82.98** (median 2.34086 ms); `triton_autotune` = **102.21**
  (1.90054 ms); speedup_vs_plain = **1.232**.
- 16384×16384: `triton_plain` e_vs_cublas = **32.23** (361.90311 ms); `triton_autotune` = **98.49**
  (118.42032 ms); speedup_vs_plain = **3.056**.

So the dramatic **"~32% → 98%" recovery is at 16384²**, where plain is 32.23% and autotune is 98.49%.
At 4096² plain is already **82.98%** and autotune only nudges it to 102.21% (1.232×). **Agent A's D1 is
correct and the §2-L188 parenthetical "at 4096² (recovers ~32%→98%)" is factually reversed.**

**What the PAPER §7.3 actually says (PDF p.8, also Table 1 p.4 + the §7.3 text on p.7/8):** the §7.3
matmul result is *"@triton.autotune with 12 tile configurations and a GROUP_SIZE_M L2 cache swizzle …
reduces latency from 2.72 ms to 1.63 ms (1.66×, E_lib = 108%) on a **4096 × 4096** matmul after 6
iterations"* (Table 8: Matmul Before 2.71 / After 1.63 / E_lib 108%). So **§7.3 IS measured at 4096²**,
and its headline is **1.66× / 108%**, on the expanded search space (GROUP_SIZE_M + num_warps +
num_stages — confirmed in `AKO4ALL/results/optimized/matmul_triton.py:10-26`, which is exactly a 12-entry
`@triton.autotune` list with GROUP_SIZE_M=8 and num_warps∈{4,8}, num_stages∈{3,4}).

**Reconciling the two:**
- REBUTTAL §1 L63: *"Section 7.3 adds GROUP_SIZE_M L2-swizzle, num_warps, and num_stages on the smaller
  4096² shape."* → **CONSISTENT with the paper** (§7.3 is at 4096²) and with the optimized kernel.
  **KEEP L63 as written.** The coordinator's instruction to "keep §1 L63 consistent" is satisfiable
  *without changing L63* — L63 never claims the 32→98 recovery; it correctly puts §7.3 at 4096².
- REBUTTAL §2 L188: *"§7.3 adds … at 4096² (recovers ~32%→98%)."* → the SHAPE is right for §7.3, but the
  **"(recovers ~32%→98%)" parenthetical is the 16384² result wrongly bolted onto the 4096² line.**

**Verified correct wording for §2 L188** (drop the wrong parenthetical; the artifact value at 4096² is a
modest recover, the 32→98 belongs to 16384²):

> `exp_autotune_matmul.py`: §5 = 12-config block-tile grid, Δ≈0 at 16384² (plain already ≈98% of cuBLAS
> after the expanded search there); §7.3 adds GROUP_SIZE_M L2-swizzle / num_warps / num_stages and is
> measured at 4096² (the paper's 1.66× / 108% point). The large 32%→98% plain-vs-autotune recovery the
> artifact shows is at **16384²**. `results/<gpu>/autotune_matmul.csv`

**Subtle but important caveat (raise it):** the artifact's *own* story (script docstring +
`exp_autotune_matmul.py:165` label "Δ~=0pp regime (§5)" for 16384²) is that **at 16384² the plain
heuristic config is "already close to optimal so the expanded search adds ≈0,"** reconciling §5's Δ=0pp.
But the CSV shows plain at 16384² is **32.23%** and autotune is **98.49%** — i.e. the expanded search
adds a **3.056× / +66pp** gain at 16384², which **contradicts the "expanded search adds ≈0 at 16384²"**
narrative the script and the rebuttal lean on. The Δ=0pp in §5 is "heuristic-tuned vs *default*"; the
autotune CSV is "expanded-autotune vs *plain heuristic*" — these are different comparisons, so there is
no formal contradiction, but the rebuttal/script prose that says "at 16384² the heuristic config is
already near-optimal" is **not** what autotune_matmul.csv shows (plain heuristic is only 32% there).
This is a latent inconsistency in the reconciliation story that Agents A and C both under-stated (A's D2
touches it; neither flags that the script's *expected-story* comment is contradicted by its own data).
**Recommend the authors not assert "16384² heuristic config is already near-optimal" anywhere** — the
honest statement is "§5's heuristic-vs-default sweep moved nothing; the *separately* expanded autotune
search helps at both shapes (1.23× at 4096², 3.06× at 16384²) but was scoped to RQ3." This keeps the
rebuttal defensible if a reviewer opens the CSV.

---

## ITEM 3 — CENSUS (Agent C): CONFIRMED. 15:7 stands.

- **leaky_relu = COMPLEX — CONFIRMED.** `pytorch_impl.py`: `c = torch.matmul(a.float(), b.float())` then
  optional `F.leaky_relu`. `triton_impl.py`: a full tiled `matmul_kernel` with `tl.dot` + GROUP_SIZE_M
  swizzle, activation fused at the epilogue. profile.csv large = `a:(8192,8192) b:(8192,8192) fp16`,
  PyTorch **72.04 ms** — an 8K² GEMM. Agent C's classification and "8K² matmul / 72 ms" are exactly
  right. (The *name* is misleading; the op is GEMM+activation.)
- **linear_activation = COMPLEX — CONFIRMED.** `kernel_ff`: RMS-normalize, **two matmuls**
  (`x_scaled @ w1.T`, `x_scaled @ w3.T`), SiLU gating — a fused Llama FFN. Large =
  `x:(1,2048,4096) w1/w3:(16384,4096)`, dual 8M-row GEMM. COMPLEX is correct.
- **cross_entropy = COMPLEX — CONFIRMED.** `cross_entropy_fwd`: blocked per-row LSE + label gather +
  smoothing/z-loss, flash-style multi-stage reduction. COMPLEX is correct, with the documented caveat
  that its PyTorch reference is flash-CE so its E_lib is not an apples-to-apples library efficiency
  (excluded from C's aggregates — correct).

**Census ratio CONFIRMED: SIMPLE 15 : COMPLEX 7 (≈2.14:1, 68%/32%).** No reclassification needed.
profile.csv independently confirms exactly 22 kernels × 2 sizes. C's Task-3 efficiencies all reproduce
(matmul Triton 31.7%, conv2d Triton ~35%, attention Triton ~1900% vs a weak O(n²) reference, batched_mm
Triton 58%). **C's verdict — the data does NOT support "DSLs better on complex kernels" (Triton SIMPLE
median 99% > COMPLEX 55%) — is sound.** E3's instruction to NOT adopt the margin-note claim "DSLs better
on complex kernels" is well-founded.

---

## ITEM 4 — EXACT CONFIG COUNT: it is **two**, and the paper says so itself.

- **Reviewer C, `reviews.txt:173`:** "The benchmark has only 22 kernels, **usually two configurations
  per cell**, one GPU architecture, and one software snapshot."
- **Paper Figure 1 caption (p.4):** "all other categories show individual kernel measurements as dots
  **(n = 2 per cell)**." (The GEMM box is IQR over a different set; the dots are n=2.)
- **Artifact `profile.csv`:** every one of the 22 kernels has exactly **two** sizes (`small`, `large`).
  Verified by counting: all 22 kernels return count 2.

So the precise replacement for "roughly two configurations per cell" (REBUTTAL L41) is simply
**"two configurations per cell"** — this matches the paper's own Figure 1 caption ("n = 2 per cell"),
Reviewer C's wording, and the artifact. The hedge "roughly" is unnecessary; the number is exactly two.

**One caveat to flag (so the edit is bullet-proof):** the *published tables* are not uniformly 2 rows —
Table 1 square-matmul has **1** shape (16384²) and fused linear+activation has **1** shape; Table 2 conv
has 2; Table 3 norm has 2. So "two configurations per cell" is true for the *benchmark sweep* (profile
.csv) and for the Figure-1 dot categories, but a pedantic reviewer could note the GEMM table cells with
one shape. Reviewer C already hedged with "usually," and the paper's caption says "n = 2," so
**"two configurations per cell" is the safe, paper-grounded phrasing.** If the authors want to be
maximally precise they can write "two configurations per cell for most kernels (a small and a large
shape)" — but plain "two" is defensible and matches the Figure-1 caption verbatim.

---

## ITEM 5 — SPOT-CHECK of A's two most load-bearing SUPPORTED verdicts: both hold, no over-credit.

**Claim 5 (conv n_spills=0; the 51 GB spill is TileLang LayerNorm) — SUPPORTED, verified raw.**
`ncu_summary.csv`:
- conv2d_triton: local-ld (spill) = **0**, local-st (spill) = **0**, regs/thread = 128, achieved occ
  33.26%.
- layer_norm_tilelang: local-ld (spill) = **51,538,558,976 B** (51.5 GB), local-st = **34,359,738,368 B**
  (34.4 GB), regs/thread = 254, occ **16.50%**, long_scoreboard = **104.91**, barrier = **0**.
A did not over-credit; the 🚩 "paper blamed conv K≥5, but spill is layer_norm" correction is real.

**Claim 6 (Winograd ≈2-3%, not "primary") — SUPPORTED, verified raw.**
`winograd_isolation.csv`: `DELTA_det_minus_nondet` e_vs_cudnn = **1.0228** (2.28%), note = "upper bound on
Winograd benefit (det − nondet)"; det run 10.86 ms vs nondet (Winograd-allowed) 10.62 ms.
`cudnn_winograd_3x3.log`: 3× `CUDNN_NUMERICAL_NOTE_WINOGRAD … val=true`. The paper (abstract p.1; §7.4
p.8: "the remaining 20% deficit is attributed to the absent Winograd algorithm selection (RC4)") does
call Winograd the *primary* remaining cause, so the 🚩 walk-back to ≈2-3% is a genuine paper correction.
A did not over-credit.

(Both of these are also the two 🚩 claims the rebuttal §3 leans on hardest; they survive scrutiny.)

---

## PLANNED-EDIT AUDIT

| Edit | Verdict | Action |
|---|---|---|
| **E1** (§3 94.6/97.8) | **BLOCK as drafted** | Do NOT replace with B's "LayerNorm pre/post-fix same kernel" — that is also wrong (item 1). Use the verified wording: 94.6% = Triton LayerNorm (Table 3); 97.8% = TileLang LayerNorm after RC0 fix (Table 6). And drop the false implication that both were re-measured under locked clocks — only the Triton 94.6%→94.46% is in significance.csv; the 97.8% post-fix TileLang cell was never re-run locked. |
| **E2** (§2 L188 tuning) | **MODIFY** | Move "32%→98%" to 16384²; at 4096² the recover is modest (≈1.23×). KEEP §1 L63 *unchanged* — it already correctly places §7.3 at 4096² and never claims the 32→98 figure, so "keep L63 consistent" needs no edit to L63. Also: do not let the revision assert "16384² heuristic is already near-optimal" (CSV shows plain=32% there). |
| **E3** (representativeness; 15:7 + per-kernel table; keep "DSLs most competitive on simpler ops"; reject "better on complex") | **APPROVE** | 15:7 confirmed; the per-kernel-table commitment is a *commitment* (revision), number-free in the response — compliant. Rejecting the margin-note "DSLs better on complex" is data-supported (item 3). Caution: the L32 phrase "where DSLs are most competitive" is only *partially* true (Triton near-parity on simple ops, but TileLang's WORST cases are also simple: layer_norm 0.32%, max_reduction 6%). Recommend softening to "where DSL kernels are often competitive with eager PyTorch" rather than a blanket "most competitive," to avoid a reviewer pointing at TileLang layer_norm 0.32%. |
| **E4** (provenance → lean on TritonBench-G published provenance; keep layer_norm=TorchInductor) | **MODIFY — this one is shakier than the coordinator thinks** | (1) The paper cites **"TritonBench [13]"** (Jianling Li et al., *TritonBench: Benchmarking LLM Capabilities for Generating Triton Operators*, Findings of ACL 2025) — it does **NOT** say "TritonBench-G" anywhere. "TritonBench-G" is the GitHub-sourced split of that benchmark; using the "-G" name in the rebuttal introduces a label not in the paper. (2) TritonBench is a benchmark of **LLM-generated** Triton operators; its "provenance statement" documents which GitHub repos the *reference* operators came from — it does **not** certify that THIS paper's specific kernels are hand-written vs. compiler-generated. (3) This directly **conflicts** with keeping "layer_norm = TorchInductor-generated": layer_norm's `triton_impl.py` imports `torch._inductor.runtime` (Agent A's definitive finding), so layer_norm did NOT come from TritonBench at all. A blanket "we adopt TritonBench-G's provenance for the Triton kernels" would be FALSE for layer_norm. **Recommended fix:** cite TritonBench [13] as the *source corpus we drew from* and its selection methodology, but keep the per-kernel exceptions explicit (layer_norm = TorchInductor-generated; the FlagGems-idiom reductions documented separately). Do not state or imply every Triton kernel inherits TritonBench's provenance. Agent A's D5 (FlagGems is naming-only, not a hard attribution) stands — the response already hedges "follow FlagGems idioms," which is fine; just don't *upgrade* it to a citation that doesn't exist. |
| **E5** (delete "roughly" → exact count) | **APPROVE** | Replace with "two configurations per cell" (item 4). Paper-grounded. |
| **E6** (depthwise "state explicitly" → concrete pointer) | **APPROVE with a correction the agents missed** | The depthwise *exclusion location* in the CURRENT paper is **not** clearly stated — §3.1 (p.3) *claims* conv covers "depthwise and strided cases," and Table 2 shows only 3×3 dense; there is no sentence in the paper saying "depthwise is excluded from the mitigation." So "which we state explicitly" is currently **not true of the paper** (it's true of the artifact: `conv2d_triton.py:115 assert groups==1` + the exp skip note). The honest fix is forward-looking: *"which the revision states explicitly (the optimized implicit-GEMM kernel is restricted to groups=1)."* Do NOT point to an existing §/Table that states it — none does. Agent C's suggested "§7 / Table 2 caption" pointer is to *where it WILL go*, not where it is. |
| **E7** ([project-website URL] placeholder) | **APPROVE** | Flag/remove. A live dead-placeholder in a submitted rebuttal is worse than omitting the clause (Agent C is right). If no public URL exists, delete the clause "and we also maintain a project website … : [project-website URL]" entirely. |
| **E8** (add reviews.txt L58/L60 to W3; soften "fused-library = R3's suggestion" → "compiler-fused") | **APPROVE** | L58 (RC3 "reads more like a hypothesis") and L60 (RC4 "disabling Winograd in cuDNN") are the sharpest A statements of the RC3/RC4 asks and are genuinely uncited in W3 (verified). "compiler-fused" matches C's `reviews.txt:182` exact phrase ("compiler-fused baseline efficiency"); "fused-library" is the rebuttal's coinage. Minor, correct. |
| **E9** (remove inline `>` author TODOs after fixes) | **MODIFY — keep, don't delete (see below)** | |

### E9 — keep vs remove the inline `>` comments
**Recommend KEEP them until every fix is verified in the FINAL trimmed §1, then strip in one pass — but
do not strip blindly per-edit.** Reasons:
- Comment L65 ("validate this paragraph with reviews text, paper draft and artifacts") guards exactly the
  paragraph (§1 L61-63 + §2 L188) that this audit just proved contains a shape error (item 2). If that
  comment is deleted before L188 is fixed, the only standing reminder of the unresolved validation is
  gone.
- Comment L36 ("dsl is better in complicated … that's what i think") records the author's *prior* that
  the data **refutes** (item 3). Keeping it until E3 lands prevents the author from re-introducing the
  refuted claim during the trim.
- Comments L34/L35/L37 (representativeness weak / TritonBench-G provenance / per-kernel count) each map to
  an open revision item (Gap 7) that is *committed but not done*; deleting them loses the to-do.
These are notes-to-self with information not captured elsewhere (the typos "representivenaess",
"explicity", lowercase "i" confirm they're private margin notes, per Agent C). Stripping is correct for
the *submitted* artifact, but should be the LAST step after E1–E8 are in and re-checked, not folded into
each edit.

---

## (a) FINDINGS I CONFIRM
- Item 2 (tuning shape) exactly as Agent A's D1: 32→98 is at 16384², not 4096². §2 L188 is reversed.
- Item 3 census 15:7; leaky_relu/linear_activation/cross_entropy all correctly COMPLEX.
- Item 4 config count = "two" (paper Fig-1 "n=2 per cell"; Reviewer C "two configurations per cell";
  profile.csv 2 sizes/kernel).
- Agent A Claims 5 and 6 (the two load-bearing 🚩) verified against raw CSVs; no over-crediting.
- Agent B's reviewer mapping, citation-accuracy check (Task 3, "zero off-by-N"), and the
  "reviews.txt 502-504 is the paper's line numbers" catch are all correct and useful.
- Agent A's N1/N2 (paper says CUDA-events+average / NHWC / allow_tf32=False / cudnn.benchmark=False, but
  benchmark code uses perf_counter+median / NCHW / sets none) — I confirm the *paper* states all four
  (§3.3 p.3 "CUDA events … average"; §3.2 "NHWC … allow_tf32=False"; §3.5 "cudnn.benchmark=False"), so
  these are artifact-vs-paper internal items, correctly NOT raised to reviewers (REBUTTAL §3 N1-N3).
- §1/§2 W8 prose stays generic/number-free (confirmed). Only §3's prose has the factual error.

## (b) FINDINGS I OVERTURN / CORRECT
- **Agent B (and any E1 that follows B): the "94.6% vs 97.8% = same LayerNorm kernel pre/post-fix" claim
  is WRONG.** 94.6% = Triton LayerNorm (Table 3); the pre-fix TileLang LayerNorm is 0.32%. 97.8% =
  TileLang LayerNorm after RC0 fix (Table 6). Different backends, not one kernel before/after. Primary
  evidence: Table 3 (p.6) and §5 prose (p.5) + Table 6 (p.8).
- **Agent A: "94.6/97.8 maps to layer_norm/softmax" is WRONG** (A flagged the softmax mismatch but landed
  on the wrong identity). 97.8% is not softmax-anything; softmax is 95.2%.
- **Coordinator E4's "TritonBench-G published provenance" overshoots:** paper cites "TritonBench" (not
  "-G"); TritonBench documents LLM-generated-operator sources, not this suite's per-kernel provenance;
  and it contradicts the proven layer_norm=TorchInductor fact. Downgrade to "cite TritonBench [13] as the
  source corpus + keep per-kernel exceptions."
- **Coordinator E6 / Agent C:** "which we state explicitly" is currently false of the *paper* (no
  depthwise-exclusion sentence exists; §3.1 in fact *claims* depthwise coverage). Make it forward-looking
  ("the revision states explicitly"), not a pointer to a nonexistent current location.

## (c) VERIFIED CORRECT WORDING — see Item 1 (§3) and Item 2 (§2 L188) boxes above.

## (d) CENSUS RATIO = 15:7 (≈2.14:1). CONFIG PHRASING = "two configurations per cell".

## (e) EDITS TO BLOCK/MODIFY
- **BLOCK E1 as drafted** (do not adopt B's same-kernel framing) → use item-1 wording.
- **MODIFY E2** (32→98 onto 16384²; leave L63 unchanged; ban "16384² heuristic already near-optimal").
- **MODIFY E4** (TritonBench not TritonBench-G; corpus-level cite + per-kernel exceptions; don't
  contradict layer_norm=TorchInductor).
- **MODIFY E6** (forward-looking "revision states explicitly"; no pointer to a current §/Table).
- **MODIFY E9** (keep inline comments until E1-E8 verified in the trimmed §1; strip last, in one pass).
- **MODIFY E3** (soften "most competitive on simpler ops" → "often competitive with eager PyTorch"; the
  blanket "most competitive" is contradicted by TileLang's simple-op failures: layer_norm 0.32%,
  max_reduction 6%).
- E5, E7, E8 APPROVE.

## (f) THINGS MISSED BY THE COORDINATOR AND THE THREE AGENTS
1. **The 97.8% number is NOT in the locked-clock artifact.** All three agents discuss whether 94.6/97.8
   "maps" to measured kernels; none noticed that significance.csv re-measures Triton LayerNorm (94.46%)
   and the *unmitigated* TileLang LayerNorm (0.32%) — the post-fix 97.8% TileLang cell was never re-run
   under locked clocks. Any §3 wording claiming "that pair … in our locked run" overstates the artifact.
   This is the single most important omission; it directly bounds what E1 may claim.
2. **The autotune reconciliation's "16384² heuristic is already near-optimal" story is contradicted by
   its own CSV** (plain=32.23% at 16384²). A's D2 brushes the 1.66× non-reproduction but neither A nor C
   flags that the *reconciliation narrative itself* (script docstring + rebuttal) misdescribes the 16384²
   regime. If a reviewer opens autotune_matmul.csv expecting "expanded search adds ≈0 at 16384²," they
   find a 3.06× gain instead. Recommend the authors reframe §5-vs-§7.3 strictly as
   "heuristic-vs-default (Δ=0)" vs "expanded-autotune-vs-plain (helps at both shapes)."
3. **E4's deeper problem:** "provenance is partly recoverable" + "TritonBench-G provenance" + "layer_norm
   = TorchInductor" cannot all be asserted together without contradiction. The clean story is: source
   corpus = TritonBench [13] + TileLang examples + own impls (per §3.1); individual provenance is
   *partly* recoverable (layer_norm demonstrably TorchInductor; some reductions follow FlagGems-style
   naming); full per-kernel provenance is a revision deliverable. The rebuttal already says roughly this
   — E4 should *not* "upgrade" it to a TritonBench-G citation.
4. **Paper-internal citation oddity (out of scope, but worth the authors knowing):** §3.1 cites
   "TritonBench [13]" and [13] *is* the TritonBench ACL-2025 paper — consistent. But [11] is the
   Lavin-Gray Winograd paper and [12] is "Triton-Forge," so the numbering around the DSL/Triton
   references is dense; not an error, just confirm [13]→TritonBench survives any renumbering in revision.
5. **Reviewer concern closure:** the edits do NOT add any new *experiment*, so the genuinely-open
   reviewer asks remain open exactly as the rebuttal already concedes: A5 kernel-mix-bias (Gap 7,
   addressed only by the 15:7 framing + commitment — fine for a rebuttal), A4 in-DSL fix (framing-only),
   cross-arch A100/H100 (deferred). No edit *fails to close* something it claims to close. E3's 15:7
   framing is the right partial answer to A5; it does not over-claim.

## (g) NUMBER-FREE-RULE CHECK on the planned edits
None of E1–E9 inject a NEW number into the *response prose* (§1) when done correctly:
- E1 lives in §3 (detailed explanation), which the rebuttal's own rule permits to carry numbers; the fix
  is factual, and the recommended wording can be number-free if desired.
- E2 lives in §2 (mapping/evidence table) — numbers permitted there by the rebuttal's stated convention.
- E3/E4/E5/E6/E8 are commitments/definitions/citations, number-free in §1 (the "15:7" and "two configs"
  appear as a *committed per-kernel table* and a count word, not as new measured results).
- **Watch-out:** if the trim pulls the §3/§2 numeric corrections UP into the ≤750-word §1 response, that
  WOULD violate the no-new-results rule. Keep the corrected 94.6/97.8 identities and the 16384²/4096²
  figures in §2/§3 and the artifact, never in the trimmed §1 prose.

---

**Report path:** `/home/lxt230026/ASE-GPUDSL-ARTIFACT/logs/rebuttal_audit/D_meta_audit.md`

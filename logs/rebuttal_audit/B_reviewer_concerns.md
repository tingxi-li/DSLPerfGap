# Reviewer Concern Catalog & Rebuttal Audit — ASE 2026 Paper #4134

**Auditor scope:** authoritative catalog of every reviewer concern in `reviews.txt`; confirmation of
paper context for concerns that reference paper internals (via `ase26-paper4134.pdf`); a line-by-line
citation-accuracy check of every `reviews.txt:<line>` reference in `REBUTTAL.md`; and a coverage/fidelity
cross-check of the rebuttal against this catalog.

**Method:** read `reviews.txt` (187 numbered lines + line 188 with no trailing newline) in full; read the
full 11-page PDF (pp. 1–11); printed every cited line of `reviews.txt` verbatim and compared. Prior
analysis `logs/REVIEWER_WEAKNESS_ANALYSIS.md` was used only as a lead and independently re-verified.

**Reviewer mapping — CONFIRMED from the file's own headers:**

| Rebuttal handle | reviews.txt header | Merit (line) | Role |
|---|---|---|---|
| @Reviewer_A | `Review #4134A` (line 7) | "2. Weak reject" (line 12) | rigor gatekeeper + **sole Artifact reviewer** (line 91 "3. Satisfactory") |
| @Reviewer_B | `Review #4134B` (line 100) | "2. Weak reject" (line 105) | benchmark-construction critic |
| @Reviewer_C | `Review #4134C` (line 154) | "3. Weak accept" (line 159) | external-validity / correctness critic |

The mapping in `REBUTTAL.md:9-10` is **accurate**.

---

## TASK 1 — CONCERN CATALOG (verbatim, with line numbers)

Classification key: RIGOR · REPR (representativeness / benchmark construction) · CORR (correctness) ·
SCOPE (claim-scope / external validity) · CLAR (clarity / definitions) · NOV (novelty) · MINOR.
"Blocking?" = raised by a Weak-Reject reviewer (A or B) **and** plausibly acceptance-critical.

### Reviewer A (#4134A — Weak Reject) — the densest reviewer

A appears twice for most points: once as a "Reasons to not accept" bullet (lines 28–32), again as a
"Detailed comment" (lines 46–75), and a third time as a numbered "Question" (lines 79–87). I treat each
distinct *topic* as one concern and list all the lines where A raises it.

| ID | Class | Verbatim quote (line) | What A wants | Blocking? |
|----|-------|------------------------|--------------|-----------|
| **A1** | RIGOR / CORR | L28 "The paper does not clearly separate kernel authoring issues, example code quality, and compiler/code-generation limitations… RC0 is framed as 'TileLang Compiler-Level Deficiencies,' yet the evidence largely reflects the use of `T.serial`… Similarly, the reported FP32 correctness issue is not traced to either the compiler or the kernel implementation." · restated L54 (RC0 a/b split) · L56 (FP32 not attributed) · L79 (Q1: clarify RC0 components; "FP32 correctness failure (2067× max error), has the root cause been identified") | Split RC0 into compiler-vs-authoring; root-cause the FP32 GEMM failure | **YES — lead objection** |
| **A2** | RIGOR / REPR | L29 "The evaluation setup is not fully consistent across operator categories. Convolution is compared against cuDNN with best-algorithm selection (including Winograd), whereas normalization relies on PyTorch's unfused eager implementation. This asymmetry… makes aggregate metrics such as 'Overall ∼65%' potentially misleading." | Fix baseline asymmetry; don't blend incomparable ratios into one "~65%" | **YES** |
| **A3** | RIGOR | L30 "The evidence supporting the identified root causes is uneven. Only RC1 and RC2 are backed by both profiling data and mitigation experiments. RC3 lacks profiling evidence… RC4 is inferred primarily by elimination… hardware counter analysis is introduced in Section 3.3 [but] not consistently used." · restated L62 (counter data never shown) · L58 (RC3 a hypothesis) · L60 (RC4: disable Winograd) · L85 (Q4: give counters — "vectorized load utilization for convolution vs. GEMM, register spill indicators for RC3, and warp stall breakdowns for RC0") | Show the Nsight counters; validate RC3/RC4 with controlled isolation, not elimination | **YES — most acceptance-threatening** |
| **A4** | SCOPE | L31 "While the paper shows that manual kernel rewrites can recover performance, it does not demonstrate corresponding improvements within the DSL frameworks themselves… unclear whether the identified issues can be systematically addressed by the compiler or tooling." | Show whether issues are fixable *inside* the DSL/compiler, not just by hand | partly (Weak-Reject; framing-level) |
| **A5** | SCOPE | L32 "The experimental scope is limited. All results are obtained on a single GPU (RTX 4000 Ada), and a large portion of the evaluated kernels are relatively simple element-wise or reduction operators. Convolution… is evaluated on only two input shapes with a single filter configuration (3×3, stride 1)." · Q5 L87 ("evidence that the identified root causes… generalize to data-center GPUs such as A100 or H100?") | Broaden GPUs; address the simple-kernel-heavy mix; broaden conv configs | **YES** (multi-part) |
| **A6** | NOV | L40 "**Novelty:** … the novelty mainly lies in the organization and interpretation rather than fundamentally new techniques." | (Soft) acknowledge limited methodological novelty | no (typical for empirical study) |
| **A7** | REPR | L42 "**Soundness:** The main concerns lie here. Several claims are not fully supported by consistent experimental coverage or clear attribution (e.g., conflation of compiler vs. kernel issues, limited convolution evaluation, and unvalidated root causes such as RC3)." | (Summary bullet — same substance as A1/A3/A6; soundness is the central worry) | YES (summary) |
| **A8** | REPR / SCOPE | L46 "Section 3.1 claims coverage of Conv2d with filter sizes from 1×1 to 7×7, including depthwise and strided variants. However, Table 2 only reports results for 3×3, stride-1. The 1×1 case is deferred, and the 5×5/7×7 cases cited later (e.g., in RC3) are not evaluated." · Q2 L81 (provide ≥1×1 + 5×5) | Report the conv configs the paper claims but never benchmarks | **YES** |
| **A9** | RIGOR / CLAR | L48 "Section 5 reports Δ = 0pp from heuristic tuning across all 22 kernels, yet Section 7.3 shows a 1.66× speedup for matmul when expanding the autotuning space. It is also unclear what 'heuristically tuned' means in practice…" · Q3 L83 (why §7.3 search not applied in RQ1) | Resolve the §5↔§7.3 apparent contradiction; define "heuristic tuning" | **YES** |
| **A10** | RIGOR | L50 "Table 7 notes a 9% variation in baseline performance across profiling runs for the same convolution setup, attributed to GPU clock fluctuations. It is not stated whether clocks were locked. Without that, small efficiency differences (e.g., 94.6% vs. 97.8%) may not be meaningful." | Lock clocks / show significance so small efficiency gaps are meaningful | **YES — the single most-quoted rigor line** |
| **A11** | RIGOR / SCOPE | L64 "RC3 references A100 register characteristics, even though all experiments are run on Ada Lovelace (sm_89), which raises some concern about architectural relevance." | Reconcile the A100 register citation with the Ada-only study | partly (Weak-Reject but A's own framing is mild: "some concern") |
| **A12** | SCOPE | L68 "The convolution mitigation is evaluated on a single configuration only. Other shapes and filter sizes are not revisited, so it is unclear how broadly the result applies." | Re-evaluate the RQ3 conv mitigation across more shapes/filters | YES |
| **A13** | CLAR | L70 "The paper mentions that LayerNorm and argmax fixes required '18' and '13' iterations, but does not define what counts as an iteration (e.g., manual edits vs. tuning steps)." | Define "iteration" | no (clarity only) |
| **A14** | MINOR | L74 "Kernel count is inconsistent: '22 kernels' (abstract, Section 3) vs. '21 kernels' (Section 1)." | Fix 21↔22 | no |
| **A15** | MINOR | L75 "Typo in the abstract: 'anamoly' → 'anomaly.'" | Fix typo | no |

> Note on A's structure: bullets L28–32 ≈ details L46–70 ≈ questions L79–87 are the same five themes
> (attribution, asymmetry, evidence, scope, conv coverage) plus the tuning/clock/iteration/minor details.
> I count **15 distinct topics**; an alternative count that merges the duplicated framings is ~11 (≈ the
> prior analysis's W-handles). Either way the *substantive* asks are: A1, A2, A3, A4, A5, A8, A9, A10,
> A11, A12, A13, minor A14/A15, soft-novelty A6.

### Reviewer B (#4134B — Weak Reject)

| ID | Class | Verbatim quote (line) | What B wants | Blocking? |
|----|-------|------------------------|--------------|-----------|
| **B1** | REPR | L127 "The construction process of the benchmark is unclear… it is unclear how the authors selected from them [TritonBench/TileLang examples]… how the authors decided which implementations to add… should justify why the resulting benchmark suite is representative." · Q1 L142 | Document selection criteria + representativeness justification | **YES — B's lead reject reason** |
| **B2** | REPR / SCOPE | L129 "First, while the authors state that there are Conv2d with 1 × 1 to 7 × 7 filters, Table 2 only presents the results for 3 × 3 filter. Second, while there are 15 element-wise tasks, the paper only reports overall results for this category, without results for each kernel." · Q2 L144 (all 1×1–7×7) · Q3 L146 (per-kernel element-wise) | Conv all filters + per-kernel element-wise breakdown (two sub-asks) | **YES** |
| **B3** | RIGOR | L131 "Some root cause analysis is not supported by experiments. For example, Table 5 shows that the root causes (RC2b, RC3, and RC4) have no direct mitigation experiment conducted… less convincing without experimental results. The authors should provide additional experimental results, or justify why such experiments are infeasible." · Q4 L148 ("Is it possible to do mitigation experiments for RC2b, RC3, and RC4?") | Run (or justify-as-infeasible) RC2b/RC3/RC4 mitigation experiments | **YES** |
| **B4** | CLAR | L133 "In RQ 1, the authors 're-evaluated all 22 kernels with heuristically tuned configurations…' but 'every kernel returned identical efficiency to its default configuration (Δ = 0pp across all rows)' (lines 502-504). More details on the tuning process should be provided. For example, which parameters are tuned, and what is the search space?" · Q5 L150 | Define the tuning search space/parameters | **YES** (overlaps A9) |
| **B5** | MINOR | L136 "Section 1 (line 138) says there are 21 kernels, but there are 22." | Fix 21↔22 | no |
| **B6** | MINOR / CLAR | L137 "The notations 16384² and 64 × 128² in Table 1 are not clear enough" | Spell out Table 1 notation | no |
| **B7** | MINOR | L138 "Typo in Abstraction: 'anamoly' (line 27) should be 'anomaly'?" | Fix typo | no |

> B distinct concerns: **7** (4 substantive + 3 minor). B4's "(lines 502-504)" is a citation to the
> *paper's* line numbers, not reviews.txt — see Task 3 note. B2 is genuinely two asks bundled in one bullet.

### Reviewer C (#4134C — Weak Accept)

| ID | Class | Verbatim quote (line) | What C wants | Blocking? |
|----|-------|------------------------|--------------|-----------|
| **C1** | SCOPE | L173 "External validity is the main concern. The benchmark has only 22 kernels, usually two configurations per cell, one GPU architecture, and one software snapshot." · also L178 (design space small) | Calibrate claims to the small evaluated design space | not blocking (Weak Accept) but central |
| **C2** | CORR | L174 "The study under-discusses correctness validation. A TileLang FP32 GEMM correctness failure is discovered and then excluded, but the paper gives limited detail on correctness tests, tolerances, and whether all measured kernels are semantically equivalent to the baselines." · L180 ("how many correctness failures occurred, what tolerances… were edge cases tested, and were the mitigation kernels revalidated?") · Q L186 | Make correctness a first-class part of the benchmark; report tolerances/inputs/edge cases/revalidation | not blocking, but C's strongest ask |
| **C3** | RIGOR / REPR | L182 "The baseline construction is uneven across categories… normalization and element-wise/reduction use PyTorch eager paths, which can be unfused… a single metric named 'library efficiency' means different things across categories. Authors may want to report separate metrics for vendor-library efficiency, eager-PyTorch efficiency, and compiler-fused baseline efficiency." | Split the "library efficiency" metric (vendor / eager / fused) | not blocking but overlaps A2 (which IS blocking) |
| **C4** | SCOPE | Q L187 "Do the main conclusions hold on A100/H100 or another data-center GPU, especially for TileLang and convolution?" | Cross-architecture generality (esp. TileLang + conv) | not blocking but overlaps A5/Q5 |

> C distinct concerns: **4** (external validity, correctness, baseline split, cross-arch). C is the only
> acceptance vote; none of C's points are independently acceptance-critical, but C3↔A2 and C4↔A5 reinforce
> A's blocking concerns, and C2 reinforces A1's FP32 sub-point.

### Catalog totals

| Reviewer | Distinct concerns | Acceptance-critical (blocking) |
|----------|-------------------|--------------------------------|
| A (Weak Reject) | 15 topics (≈11 if duplicate framings merged) | **A1, A2, A3, A5, A8, A9, A10** (+ A12 strong, A4/A11 partial); A6 novelty & A13–A15 minor not blocking |
| B (Weak Reject) | 7 (4 substantive + 3 minor) | **B1, B2, B3, B4** |
| C (Weak Accept) | 4 | none independently blocking; **C2/C3/C4 reinforce A's blockers** |

**Cross-reviewer convergence** (these are the load-bearing themes):
- Conv coverage 1×1–7×7: **A8 + B2** (both with explicit Q2).
- Tuning §5↔§7.3 + "heuristic tuning" undefined: **A9 + B4** (both Q).
- RC2b/RC3/RC4 + counters unshown: **A3 + B3** (both Q4).
- Single-GPU / external validity: **A5 + C1 + C4** (all three reviewers).
- Baseline asymmetry / split metric: **A2 + C3**.
- FP32 + correctness rigor: **A1 (FP32 half) + C2**.

---

## TASK 2 — CONTEXT CONFIRMATION (located in the PDF)

Every concern that references paper internals was located. **No paper context could NOT be located** —
all referenced sections/tables exist. Locations:

| Concern → paper claim | Located at | Confirmed content |
|---|---|---|
| **A9 §5↔§7.3 tuning "contradiction"** | §5 "Evaluation (RQ1)" p.5 (PDF lines 502–509): *"We re-evaluated all 22 kernels with heuristically tuned configurations for both DSLs to its default configuration (Δ = 0pp across all rows)"*; §7.3 "GEMM and Reduction Kernels" p.7 (PDF lines 800–812): square matmul *"@triton.autotune with 12 tile configurations and a GROUP_SIZE_M L2 cache swizzle… reduces latency from 2.72 ms to 1.63 ms (1.66×, E_lib = 108%) on a 4096 × 4096 matmul after 6 iterations."* | **CONFIRMED.** §5 = 16384² shape, Δ=0pp; §7.3 = 4096² shape, +GROUP_SIZE_M/num_warps/num_stages, 1.66×. The two use different shapes AND different search spaces — exactly as the rebuttal §1-A&B-3 claims. |
| **A10 "94.6% vs 97.8% may not be meaningful without locked clocks"** | "94.6%" = **Table 3** (Normalization, p.5): LayerNorm BF16 8192² E_lib = **94.6%** (PDF lines 591–593). "97.8%" = **Table 6** (Normalization before/after RC0 fix, p.8): LayerNorm "After" 0.89 ms, E_lib = **97.8%** (PDF lines 835–837). Table 7 (p.8, PDF lines 844–854) conv before/after RC1+RC2 with footnote: *"PyTorch baseline re-measured… Tab. 2 reports 10.96 ms… a 9% difference attributable to run-to-run GPU clock variation between profiling sessions."* | **CONFIRMED that the numbers exist and the 9%/clock context is real.** ⚠️ **IMPORTANT NUANCE for Task 4:** the "94.6% vs 97.8%" pair is **LayerNorm pre-fix (Table 3) vs LayerNorm post-fix (Table 6)** — i.e. the *same kernel before and after the RC0 mitigation*. It is NOT a cross-kernel pair. The rebuttal §3 (line 228) claims this pair "maps to **layer_norm (94.5%)** and **softmax (95.2%)**" — that is a **misidentification** (see Task 4-d). |
| **A11/W13 "RC3 cites A100 register specs in an Ada-only study"** | §6 RC3 p.6 (PDF lines 687–690): *"Each streaming multiprocessor on **A100** has 65,536 32-bit registers per block. A block of 128 threads with 64 registers per thread occupies the full register file…"* | **CONFIRMED** — the paper literally says "A100" inside RC3 while §3.5 (p.3) fixes hardware to "NVIDIA RTX 4000 Ada Generation GPU (Ada Lovelace, sm_89)." A's concern is factually grounded in the text. |
| **A1/A7 "RC0 conflates compiler vs kernel-authoring"** | §6 "RC0: TileLang Compiler-Level Deficiencies" p.5 (PDF lines 527–567): two mechanisms — (i) *"original TileLang normalization kernels implement the reduction… using `T.serial`… The native alternative, `T.reduce`, lowers to a parallel butterfly reduction"*; (ii) *"absent 128-bit vectorized loads (LDG.128)."* §7.1 p.7 (PDF lines 768–771): *"This result does not represent a new optimization technique: `T.reduce` already existed in the API."* | **CONFIRMED** — the "Compiler-Level" label and the "no new technique" admission both appear, exactly as A1/L79 quote. |
| **A2/C3 baseline asymmetry (conv-vs-cuDNN-Winograd vs norm-vs-eager)** | §3.2 "Baseline Construction" p.3 (PDF lines 278–289): GEMM=cuBLAS via `torch.matmul`; Conv=`nn.Conv2d` NHWC `allow_tf32=False`; Normalization=`F.layer_norm`/`F.rms_norm`; element-wise="standard PyTorch eager." §2.4 (p.2) says cuDNN selects Winograd for 3×3. "Overall ~65%" = **Table 4** (p.6). | **CONFIRMED** — both baselines and the blended "Overall (excl. attn) ~65% / ~30%" exist (Table 4). NOTE: §3.2 lists normalization baseline as `F.layer_norm`/`F.rms_norm` (library calls), which complicates the "unfused eager" framing — see Task 4. |
| **FP32 GEMM exclusion (A1 L56/L79, C2 L174)** | §6 "TileLang float32 correctness" p.5 (PDF lines 549–558): *"TileLang's GEMM kernel produces numerically incorrect results under float32 precision, with 99.6% of output elements mismatched… (greatest relative difference 2067× at tested problem shapes of 4096×2048×1024). The float16 variant passes correctness checks in all cases. For this reason, all GEMM measurements in this paper use FP16; float32 TileLang GEMM results are excluded."* | **CONFIRMED** — 99.6%, 2067×, 4096×2048×1024, FP16-only decision all present. |
| **Winograd "primary remaining cause" (RC4)** | §6 RC4 p.7 (PDF lines 703–716); §7.2 p.7 (PDF lines 791–797): *"The residual 20% gap reflects a limit… cuDNN selects among algorithm families including Winograd… Winograd support within a DSL kernel is identified as future work."* Abstract (p.1, PDF lines 37–39): *"The remaining gap is primarily associated with missing Winograd support."* §7.4 (p.8, PDF lines 907–909): *"the remaining 20% deficit is attributed to the absent Winograd algorithm selection (RC4)."* | **CONFIRMED** — the paper does call Winograd the **primary** remaining cause (abstract + §7.4), which is exactly what the rebuttal's internal audit (§2 row W3, §3) says it must walk back to ≈2–3%. |
| **Counters never tabulated (A3/B3)** | §3.3 "Profiling Setup" p.3 (PDF lines 305–311): lists memory/compute/instruction counter classes. §3.4 (PDF lines 326–328): *"[secondary metrics] are used qualitatively in root-cause analysis (RQ2) to interpret counter measurements; they are not tabulated separately."* Table 5 (p.8): RC2b/RC3/RC4 rows show "—" under "Contribution." | **CONFIRMED** — the paper admits counters are not tabulated; Table 5 "—" for RC2b/RC3/RC4 is real. |
| **Conv only 3×3 stride-1 (A8/B2)** | §3.1 (p.3, PDF lines 261–265) claims conv "1×1 to 7×7 filters, including depthwise and strided cases"; **Table 2** (p.6, PDF lines 584–587) reports only two rows, both 3×3 stride-1: (8,64,56,56) and (32,256,128,128). §5.1 (PDF line 474) defers 1×1 ("reserved for a follow-up experiment"). | **CONFIRMED** — the §3.1-claim-vs-Table-2-coverage gap is exactly as stated. |
| **"18"/"13" iterations undefined (A13)** | §7.1 (p.7, PDF line 737) "over 18 iterations" (LayerNorm); §8 Table 5/8 region — argmax "13 iterations" (p.8, PDF line 863 "8192 × 32768 reduction after 13 iterations"). | **CONFIRMED** — both counts appear with no definition of "iteration" in the paper. |
| **21 vs 22 kernels (A14/B5)** | "22 kernels" abstract (p.1 line 16) & §3.1 (p.3 line 259); **"21 kernels"** §1/§2 (p.2, PDF line 138: *"a suite of 21 kernels spanning five operator categories"*). | **CONFIRMED** — B5 even pinpoints "line 138" of the paper, which matches PDF line 138. |
| **"anamoly" typo (A15/B7)** | Abstract p.1 (PDF line 27): *"severe normalization **anamoly** in TileLang."* | **CONFIRMED.** |

**Flag: no concern's paper context was unlocatable.** Every internal reference resolves to a concrete
section/table. The one place where the paper text and concern diverge in a way that matters is A2/C3's
"normalization relies on unfused eager" — §3.2 actually lists `F.layer_norm`/`F.rms_norm` (library
functions), so A is partly imprecise *about LayerNorm specifically*; this is the seed of the rebuttal's
"LayerNorm is fused, so 94.6% is fair" correction (which is itself defensible — see Task 4).

---

## TASK 3 — CITATION-ACCURACY CHECK

Every `reviews.txt:<line>` citation in `REBUTTAL.md` §2 (the W1–W13 mapping table) and §4 was opened at
the exact line and compared to the referenced content. Result: **the line-number citations are
substantially ACCURATE.** Below is the per-handle verification, then the few discrepancies.

### §2 mapping-table citations — verified

| Handle (rebuttal) | Cited lines | At those lines? | Verdict |
|---|---|---|---|
| W1 RC0/FP32 | `:28,54,79` | 28 = RC0 conflation bullet ✓; 54 = RC0 (a)/(b) split ✓; 79 = Q1 RC0+FP32 2067× ✓ | **ACCURATE** |
| W1/W11 FP32 GEMM | `:56,79,174,180` | 56 = FP32 not attributed ✓; 79 = Q1 (FP32 2067×) ✓; 174 = C "FP32 GEMM correctness failure… excluded" ✓; 180 = C "how many failures… tolerances… revalidated" ✓ | **ACCURATE** |
| W2 baseline asymmetry | `:29` (A), `:182` (C) | 29 = conv-vs-cuDNN / norm-vs-eager / "Overall ∼65%" ✓; 182 = C "baseline construction is uneven… separate metrics" ✓ | **ACCURATE** |
| W3 counters + RC2b/3/4 | `:30,62` (A), `:85` (A-Q4), `:131,148` (B) | 30 = "evidence… uneven… RC3 lacks profiling… RC4 by elimination" ✓; 62 = "Section 3.3… does not show counter data" ✓; 85 = Q4 (vectorized-load conv vs GEMM / spill RC3 / warp-stall RC0) ✓; 131 = "Table 5… RC2b, RC3, RC4… no direct mitigation" ✓; 148 = Q4 "mitigation experiments for RC2b, RC3, and RC4" ✓ | **ACCURATE** |
| W4 in-DSL fix | `:31` | 31 = "manual kernel rewrites… does not demonstrate… within the DSL frameworks" ✓ | **ACCURATE** |
| W6 conv coverage + mitig | `:46,68` (A), `:81,129,144` (B/A-Q2) | 46 = "Section 3.1 claims 1×1 to 7×7… Table 2 only 3×3" ✓; 68 = "convolution mitigation… single configuration only" ✓; 81 = A-Q2 (1×1 + 5×5) ✓; 129 = B "Conv2d with 1×1 to 7×7… only 3×3" ✓; 144 = B-Q2 ✓ | **ACCURATE** |
| W7 tuning | `:48,83` (A), `:133,150` (B) | 48 = "Section 5… Δ = 0pp… Section 7.3… 1.66×… 'heuristically tuned'" ✓; 83 = A-Q3 ✓; 133 = B "re-evaluated all 22 kernels with heuristically tuned… Δ = 0pp… (lines 502-504)" ✓; 150 = B-Q5 ✓ | **ACCURATE** |
| W8 clock locking | `:50` | 50 = "Table 7… 9% variation… not stated whether clocks were locked… 94.6% vs. 97.8%… may not be meaningful" ✓ | **ACCURATE** (the quote at line 50 is verbatim correct) |
| W9 provenance | `:127,142` | 127 = "construction process… unclear… selected from them… representative" ✓; 142 = B-Q1 ✓ | **ACCURATE** |
| W10 per-kernel EW | `:129,146` | 129 = "15 element-wise tasks… only overall results" ✓ (same bullet as conv — see note); 146 = B-Q3 "per-kernel results for the element-wise category" ✓ | **ACCURATE** (note: line 129 is *shared* with W6's conv ask — it's one B bullet covering both, which is fine) |
| W11 correctness | `:174,180,186` | 174 = C correctness under-discussed ✓; 180 = C "how many failures… tolerances… edge cases… revalidated" ✓; 186 = C-Q "tolerances, input distributions, and edge cases" ✓ | **ACCURATE** |
| W5/W13 cross-arch | `:32,64` (A), `:87` (A-Q5), `:173,187` (C) | 32 = "single GPU (RTX 4000 Ada)… simple element-wise/reduction… two input shapes" ✓; 64 = "RC3 references A100 register characteristics… Ada Lovelace (sm_89)" ✓; 87 = A-Q5 (A100/H100) ✓; 173 = C "only 22 kernels… one GPU architecture… one software snapshot" ✓; 187 = C-Q (A100/H100, TileLang+conv) ✓ | **ACCURATE** |
| W12 iteration | `:70` | 70 = "'18' and '13' iterations… does not define what counts as an iteration" ✓ | **ACCURATE** |
| M1/M2/M3 minor | `:74,75` (A), `:136,137,138` (B) | 74 = "22 vs 21 kernels" ✓; 75 = A "anamoly→anomaly" ✓; 136 = B "line 138 says 21… but 22" ✓; 137 = B "16384² and 64×128²… not clear" ✓; 138 = B "anamoly (line 27)" ✓ | **ACCURATE** |

### §4 citations — verified

| §4 item | Cited line | At that line? | Verdict |
|---|---|---|---|
| A100/H100 runs | `reviews.txt` A-Q5 (`:87`) implicit, C | line 87 = A-Q5 ✓ | **ACCURATE** |
| Gap 6 (RC3 A100 specs) | `reviews.txt:64` | 64 = RC3 A100 register characteristics ✓ | **ACCURATE** |
| Gap 7 (kernel-mix) | `reviews.txt:32` | 32 = "a large portion… relatively simple element-wise or reduction operators" ✓ | **ACCURATE** |
| Gap 8 (novelty, not contested) | `reviews.txt:40` | 40 = "novelty mainly lies in the organization and interpretation" ✓ | **ACCURATE** |
| §3 novelty aside | `:40` | 40 ✓ | **ACCURATE** |

### Citation discrepancies found

1. **NOT a rebuttal error — a paper-internal cross-ref inside a quote.** `REBUTTAL.md:188` (W7 row) and
   §1-A&B-3 quote Reviewer B's own phrasing including **"(lines 502-504)"**. Those are the **paper's**
   line numbers, not reviews.txt's — and in `reviews.txt` lines **502–504 are blank** (the file is only
   187 lines long). The rebuttal does **not** itself cite `reviews.txt:502-504`; it correctly anchors B's
   tuning concern to `reviews.txt:133,150`. So this is fine, but flag it so no one later "corrects"
   B's quote by looking up reviews.txt:502. (Paper §5 text is at PDF lines 502–509, confirming B meant the paper.)

2. **Minor under-citation, not a wrong line.** W3's A-side could also cite A's detailed bullets L58
   (RC3 "reads more like a hypothesis than a validated finding") and L60 (RC4 "disabling Winograd in
   cuDNN"); the rebuttal cites only `:30,62,85`. Content is fully covered, but L58/L60 are the most
   specific A statements of the RC3/RC4 asks and are uncited. Not an *error*, an omission of the sharpest lines.

3. **W8 single-line citation is exact but narrow.** `:50` is verbatim correct. (No off-by-N anywhere in W8.)

**Bottom line for Task 3: ZERO off-by-N / wrong-line citation errors were found in REBUTTAL.md §2 or §4.**
Every `reviews.txt:<line>` reference lands on content that genuinely supports the handle. The only
line-number artifact ("502-504") is inside a *quotation of the reviewer* and refers to the paper, not
reviews.txt — and the rebuttal handles it correctly.

---

## TASK 4 — COVERAGE & FIDELITY CHECK

### (a) Reviewer concerns the rebuttal does NOT address at all

**None of the substantive concerns are entirely unaddressed.** Every A/B/C concern in the catalog maps to
a rebuttal handle:

- A1→W1; A2→W2; A3→W3; A4→W4; A5→W5; A8→W6; A9→W7; A10→W8; A11→W13/§4-Gap6; A12→W6 (mitigation half);
  A13→W12; A14/A15→M1/M2.
- B1→W9; B2→W6+W10; B3→W3; B4→W7; B5/B6/B7→M1/M3/M2.
- C1→W5/To-All-#3 (scope calibration); C2→W11; C3→W2; C4→W5/W13.

Two items are **acknowledged-but-deliberately-not-rebutted**, which the rebuttal states openly (so not a
coverage failure):
- **A6 / Gap 8 — novelty** ("organization/interpretation, not new techniques", L40). REBUTTAL §3 line 254
  and §4 line 295 explicitly choose not to contest it. Defensible for an empirical paper.
- **A4 — in-DSL fix** is answered only as *framing* (W4: "lower bound on what tooling could automate"), no
  new experiment. The rebuttal is honest that this is framing-only (⚠️ flag in §2). Acceptable but thin.

The closest thing to a genuine gap: **A5's "large portion of kernels are simple element-wise/reduction"
sub-point** (the *kernel-mix* worry, distinct from the GPU-count worry). The rebuttal's §1 response
(To-All-#2, line 32: "we report efficiency per category… so the simpler element-wise/reduction operators…
do not inflate the headline") addresses the *metric-inflation* angle but NOT whether the suite is
*biased toward easy kernels* — which is the actual representativeness charge. §4-Gap7 (line 290) and the
margin note at `REBUTTAL.md:37` ("how many… are easy-to-implement… how many… complicated… list a
per-kernel table") show the authors KNOW this is unaddressed and are deferring it to the revision. So:
**partially addressed, with the core "is the mix biased?" question deferred, not answered.**

### (b) Concerns MISCHARACTERIZED or answered OFF-TARGET

1. **🔴 "94.6% vs 97.8%" misidentified in §3 (the most concrete fidelity error).**
   `REBUTTAL.md:228` (§3 "Clock locking" narrative) states: *"that pair of numbers maps to
   **layer_norm (94.5%)** and **softmax (95.2%)** in our locked run."* But in the paper the pair is
   **LayerNorm before fix (94.6%, Table 3) vs LayerNorm after fix (97.8%, Table 6)** — i.e. the SAME
   kernel pre/post the RC0 mitigation, NOT layer_norm-vs-softmax. A's line 50 lifts both numbers from the
   normalization results, and 97.8% appears ONLY in Table 6 (LayerNorm "After"). Mapping it to "softmax
   95.2%" is a misattribution of what A's quoted numbers refer to. This does not change whether the gaps
   are "real," but it means the rebuttal's narrative answers a *different* comparison than the one A
   flagged. **Recommend: re-anchor to LayerNorm pre/post-fix (94.6%→97.8%), or measure those exact two
   table cells under locked clocks.** (Note: this error is confined to §3's prose; the §2 W8 row and §1
   To-A-#7 do not repeat it — they stay generic, "small efficiency differences," which is fine.)

2. **🟡 "LayerNorm baseline is fused, so 94.6% is fair" subtly reframes A2/C3 (defensible but one-sided).**
   `REBUTTAL.md:75` and Appendix-A line 333 assert the LayerNorm baseline is fused `F.layer_norm` and
   therefore Triton's 94.6% is a fair library comparison. Paper §3.2 (PDF lines 281–283) **supports this**
   (it lists `F.layer_norm`/`F.rms_norm` as the normalization baselines), so the rebuttal is *factually
   right about LayerNorm* and A's blanket "normalization relies on unfused eager" (L29) is imprecise for
   LayerNorm. **However** A/C's actual charge is broader — that "library efficiency" *blends* incomparable
   denominators across categories. The rebuttal concedes RMSNorm + element-wise are eager and adopts the
   split metric (good), so the substance is covered — but the framing "94.6% is fair" risks *appearing*
   to rebut the whole asymmetry concern when it only neutralizes the LayerNorm-specific instance. Not an
   error, a rhetorical narrowing. (Also note the prior analysis flags N1: §3.2 claims NHWC/`allow_tf32=
   False`/`cudnn.benchmark=False` that the artifact code allegedly does not set — but **the paper §3.5
   does state all three** (PDF lines 340–342: "disable cuDNN benchmark mode (torch.backends.cudnn.
   benchmark=False)"), so any mismatch is artifact-vs-paper, NOT paper-vs-reviewer; out of scope here.)

3. **🟡 RC4 "Winograd ≈ 2–3%, not primary" — the rebuttal's own evidence CONTRADICTS the paper, and the
   response wording hides it.** This is an internal-consistency issue the rebuttal handles deliberately:
   the paper says Winograd is the **"primary"** remaining conv cause (abstract PDF line 38; §7.4 PDF
   lines 907–909), but the rebuttal's §2 W3 row + §3 line 239 report the Winograd isolation measured only
   **≈2–3%**. The *response text* (§1-A&B-1, line 53) says counters "sharpen an attribution… we are
   updating it accordingly" — i.e. it neither states the 2–3% nor admits the paper over-claimed
   "primary." This is intentional (the rebuttal's stated "ground & sharpen, never re-assert" stance), and
   it is **not a mischaracterization of a reviewer** — but it IS a place where the rebuttal's qualitative
   wording could read to A as evasive about a finding (RC4≈"primary") that A explicitly distrusted
   (L30 "RC4 is inferred primarily by elimination", L60 "disabling Winograd… would make this claim more
   convincing"). Worth surfacing: A asked for exactly the experiment that overturned the paper's claim.

### (c) Places the rebuttal claims a reviewer asked for something they did NOT

1. **🟡 "Adopting @Reviewer_C's suggestion… split metrics — vendor-library, **fused-library**, and
   eager-PyTorch" (`REBUTTAL.md:75`, also §2 W2 row, Appendix-A line 357).** C's line 182 suggests
   "vendor-library efficiency, eager-PyTorch efficiency, and **compiler-fused baseline** efficiency." The
   rebuttal's "fused-library" ≈ C's "compiler-fused baseline" — close, but the rebuttal **adds
   `torch.compile` fused baselines for element-wise/normalization** and presents it as C's suggestion.
   C did ask for a compiler-fused metric, so this is **largely faithful**; the only stretch is implying
   C asked for the `torch.compile` *experiment* specifically (C asked for the *metric*, the experiment is
   the authors' chosen instrument). Minor.

2. **🟡 W3 attributes a specific 3-way isolation list to "Reviewer2-Q4" precisely.** `REBUTTAL.md:118`
   (To-B-#1) says the RC2b/RC3/RC4 isolations are "where… your Q4 asks." B-Q4 (L148) asks "Is it
   **possible** to do mitigation experiments for RC2b, RC3, and RC4?" — a yes/no feasibility question,
   not a request for a specific counter list. The *counter list* (vectorized-load, spill, warp-stall)
   comes from **A-Q4 (L85)**, not B-Q4. The rebuttal mostly attributes correctly (W3 cites both A-Q4 and
   B-Q4), but the To-B-#1 phrasing slightly over-credits B's Q4 with asking for the isolations *as
   experiments* when B asked whether they were *feasible*. Substantively fine (the experiments answer
   both), faithfulness is just loose.

3. **No fabricated asks.** I found **no** instance where the rebuttal attributes a concern to a reviewer
   who did not raise it. The reviewer→handle attributions in §2 are correct (e.g., W4 correctly A-only;
   W12 correctly A-only; W9 correctly B+C; W8 correctly A-only; W10 correctly B-only).

### Fidelity summary

| Check | Finding |
|---|---|
| Unaddressed concern | None fully unaddressed. **A5 kernel-mix bias** is only partially addressed (metric-inflation answered; "is the suite biased toward easy kernels?" deferred to §4-Gap7). A4 & A6 intentionally framing-only / not contested (stated openly). |
| Mischaracterization | **(1) "94.6% vs 97.8%" → "layer_norm vs softmax" in §3 line 228 is wrong** (it's LayerNorm pre/post-fix). (2) "LayerNorm fused ⇒ 94.6% fair" rhetorically narrows A2/C3. (3) RC4≈2–3% finding contradicts the paper's "primary" claim; response wording is deliberately oblique. |
| False "reviewer asked X" | None fabricated. Two mild over-attributions: "fused-library" framed as C's exact suggestion (C said "compiler-fused"); B-Q4 framed as asking for the isolations-as-experiments (B asked feasibility; the counter list is A-Q4). |

---

## APPENDIX — verbatim line spot-checks used (reviews.txt)

- L12 `2. Weak reject` (A) · L105 `2. Weak reject` (B) · L159 `3. Weak accept` (C) · L91 `3. Satisfactory` (A = Artifact reviewer).
- L502–504 of reviews.txt: **blank** (file ends at line 187; line 188 has no trailing newline). B's
  "(lines 502-504)" at L133 therefore refers to the PAPER (§5 is at PDF lines 502–509), not reviews.txt.

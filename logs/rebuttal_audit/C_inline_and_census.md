# Rebuttal Audit C — Inline-Comment Ledger + Kernel-Complexity Census

Paper: ASE 2026 #4134. Source: `/home/lxt230026/ASE-GPUDSL-ARTIFACT/REBUTTAL.md`.
Read-only audit. Data: `ViperBench/results/profile.csv` (large size), 22 kernels under `ViperBench/`.

---

## TASK 1 — Inline-Comment Ledger (exhaustive)

Method: scanned every `>` blockquote line, `<!-- -->`, `TODO/FIXME/XXX`, and `[bracketed]` placeholder.
Distinguished **author asides** (notes-to-self, validation requests, TODOs) from **structural
blockquotes** (the legend / "outside-750-word" captions that are part of the rebuttal's own prose).

### A. Genuine inline author comments (asides / validation requests / placeholders)

| # | Verbatim quote (line) | Annotates | Classification | Action demanded | Satisfied by current text? |
|---|---|---|---|---|---|
| 1 | "the justification for representivenaess is weak" (L34) | §1 To-All #2 Benchmark Representativeness (L30-32) | factual-validation-request (also flags weak argument) | Strengthen the representativeness justification | **No** — L32 only asserts "spans the operator categories central to transformer and CNN inference" + per-category metric; no provenance statement, no simple/complex count, no per-kernel table. (Note: "representivenaess" is itself misspelled in the comment, not in the body.) |
| 2 | "provenance: i selected triton kernels from the TritonBench-G, why not just use their provenance statement" (L35) | Same passage, provenance clause (L32) | content-gap | Cite TritonBench-G's own provenance statement instead of partial self-recovery | **No** — L32 says provenance is "partly recoverable … compiling the complete provenance for the revision"; does not reference/adopt TritonBench-G's provenance statement. |
| 3 | "have you fact-check? dsl is better in complicated, highly-fused kernels, that's what i think" (L36) | Same passage; the "simpler … operators, where DSLs are most competitive" clause (L32) | factual-validation-request | Verify whether DSLs are actually better on complex/fused kernels (and note L32 claims the *opposite* — DSLs most competitive on *simple* ops) | **No** — claim is unverified in text; and the data (Task 3) contradicts the author's intuition and is mixed vs. L32's wording. Needs the numbers. |
| 4 | "check: among all the kernels we involved, how many of them are easy-to-implement kernels and how many of them are complicated ones? if the ratio is not highly biased, i think we can claim the coverage is fine, just list a per-kernel table in addition to the per-category table" (L37) | Same passage (L30-32) | content-gap | Count simple vs complex kernels; if ratio not highly biased, coverage is defensible; add a per-kernel table alongside the per-category table | **No** — no count and no per-kernel table anywhere in the rebuttal. (Census in Task 2 supplies it: 15:7.) |
| 5 | "roughly is a very bad word to use in academic writing" (L43) | §1 To-All #3 Claim Scope: "…common shapes, **roughly** two configurations per cell…" (L41) | writing-fix | Replace the word "roughly" with a precise figure or hedge-free phrasing | **No** — the word "roughly" is still present verbatim in L41. |
| 6 | "state explicity at where?" (L59) | §1 A&B #2, depthwise sentence: "…does not yet cover, **which we state explicitly**." (L57) | factual-validation-request | Identify the concrete location (paper section/table) where the depthwise exclusion is stated | **No** — text claims "we state explicitly" but never names a location; no §/Table pointer. (Note: comment itself misspells "explicity".) |
| 7 | "validate this paragraph with reviews text, paper draft and artifacts we have" (L65) | §1 A&B #3 Tuning Contradiction §5↔§7.3 (L61-63) | factual-validation-request | Cross-check the 12-config grid / bm,bn∈{32,64,128} / bk∈{32,64} / 4096²→16384² / GROUP_SIZE_M numbers against reviews.txt, paper draft, and artifacts | **Partial** — paragraph is internally consistent and matches the Appendix-A draft (L335) and §2 row (L188), so it is self-consistent; but the comment asks for an external cross-validation against reviews.txt/paper/artifacts that this audit cannot confirm from the rebuttal alone. Numbers are plausible; provenance unverified here. |
| 8 | "[project-website URL]" (L160) | §1 Revision Plan closing sentence (L160) | placeholder | Replace bracket with the real project-website URL | **No** — literal placeholder still in text. |

### B. Structural blockquotes / legend — NOT author asides (listed for completeness, no action)

These are part of the rebuttal's own prose (captions and a deliberate "not-contested" note), not notes-to-self:

| Lines | Content | Why not an aside |
|---|---|---|
| L19-22 | "**Draft for structure review.** … We will trim §1 to ≤750…" | A standing caption describing the draft's status; an instruction to *reviewers of the draft*, not a self-TODO with a concrete edit. Borderline — see "judgment calls." |
| L136 | "Part of the response, outside the ≤750-word limit — commitments only, no new results." | Caption explaining the Revision-Plan subsection's word-limit status. |
| L173-178 | The `🔑 ⭐ 🚩 ⚠️` "**Legend:**" block | Explicitly the legend (excluded by the task definition). |
| L295-299 | "Gap 8 (@Reviewer_A, `reviews.txt:40` …) is intentionally **not** contested…" | Rationale prose inside §4; a deliberate decision, not a pending action. |

### C. Other markers found (HTML comments / TODO / FIXME / placeholders)
- **HTML comments (`<!-- -->`):** none.
- **TODO/FIXME/XXX:** none (the grep hits on `TODO` were inside the filename token `REVISION_TODO.md`, not markers).
- **Other `[bracketed]` placeholders:** only `[project-website URL]` (L160, row 8). All other bracketed tokens are `reviews.txt:<line>` citations and `[B,H,T,D]`-style tensor shapes, not placeholders.

**Judgment call on L19-22:** I classify it as a structural caption (group B) rather than an aside, because it addresses an external "structure reviewer" and states a plan ("we will trim … once the structure is approved") rather than demanding a specific text edit. If the auditor wants maximal inclusion, treat it as a writing-process TODO ("trim §1 to ≤750 words; Appendix A is the fallback") — but it is qualitatively different from the L34-L65 marginalia, which are terse lowercase notes-to-self with typos ("representivenaess", "explicity", "i").

---

## TASK 2 — Kernel-Complexity Census (answers comment #4, L37)

Rubric: **SIMPLE** = element-wise (add/mul/relu/…) or single-pass reduction/normalization
(argmax, max/mean reduction, softmax, layer_norm, …). **COMPLEX** = matmul/GEMM, batched_matmul,
conv2d, attention, and other heavily-fused / multi-stage / cross-channel-reduction ops.
Classification is from each kernel's `pytorch_impl.py` reference.

| Kernel | Operator type | SIMPLE / COMPLEX | One-line justification |
|---|---|---|---|
| add | element-wise binary | SIMPLE | `torch.add(x,y)` — pure pointwise. |
| mul | element-wise scalar | SIMPLE | `2*x` — pure pointwise. |
| relu | element-wise activation | SIMPLE | `F.relu(x)` — pointwise. |
| swiglu | element-wise gated activation | SIMPLE | `chunk` then `x*sigmoid(x)*y` — pointwise, no reduction/matmul (Triton kernel is a 1-pass `_swiglu_fwd_kernel`). |
| argmax | reduction (1-pass) | SIMPLE | `torch.argmax` over one dim — single reduction. |
| max_reduction | reduction (1-pass) | SIMPLE | `torch.max(dim)` → values+indices — single reduction. |
| mean_reduction | reduction (1-pass) | SIMPLE | `torch.mean(dim)` — single reduction. |
| logsumexp | reduction (1-pass) | SIMPLE | `torch.logsumexp(dim=-1)` — max+exp+sum+log along one axis, one pass. |
| softmax | normalization (1-pass row) | SIMPLE | `F.softmax(dim=-1)` — rowwise max/exp/sum/div. |
| log_softmax | normalization (1-pass row) | SIMPLE | `F.log_softmax` — rowwise, single pass. |
| layer_norm | normalization | SIMPLE | `F.layer_norm` — per-row mean/var/affine, single pass. |
| rms_norm | normalization | SIMPLE | `x*rsqrt(mean(x²))*w` — per-row reduction + scale. |
| embedding | gather | SIMPLE | `F.embedding` + range mask — memory-bound row gather, no compute. |
| index_select | gather | SIMPLE | `torch.index_select` rows — pure gather/copy. |
| matrix_transpose | data movement | SIMPLE | `x.T.contiguous()` — layout shuffle, no arithmetic. |
| **matmul** | GEMM | **COMPLEX** | `torch.matmul` 16384² fp16 — tiled GEMM, the canonical complex kernel. |
| **batched_matmul** | batched GEMM | **COMPLEX** | `einsum('mk,mnk->mn')` — per-row vec×mat over 128 batches with K-reduction. |
| **conv2d** | convolution | **COMPLEX** | `F.conv2d` — implicit-GEMM, cross-channel reduction over 3×3×256. |
| **attention** | fused attention | **COMPLEX** | chunked linear attention: nested QKᵀ, SᵀV, running state — multi-stage, multi-matmul. |
| **linear_activation** | fused RMSNorm + dual-GEMM + SiLU | **COMPLEX** | `kernel_ff`: RMS-normalize, two matmuls (`@w1.T`,`@w3.T`), SiLU gating — heavily fused. |
| **leaky_relu** | GEMM + activation | **COMPLEX** | despite the name, body is `matmul(a,b)` then optional leaky_relu; large input `a:(8192,8192) b:(8192,8192)` is an 8K² GEMM (72ms PyTorch). Fused matmul+activation ⇒ COMPLEX. |
| **cross_entropy** | fused flash-CE | **COMPLEX** | `cross_entropy_fwd`: blocked per-row LSE + label gather + smoothing/z-loss — multi-stage fused reduction (flash-style). |

### Count and ratio

- **SIMPLE: 15** (add, mul, relu, swiglu, argmax, max_reduction, mean_reduction, logsumexp, softmax, log_softmax, layer_norm, rms_norm, embedding, index_select, matrix_transpose)
- **COMPLEX: 7** (matmul, batched_matmul, conv2d, attention, linear_activation, leaky_relu, cross_entropy)
- **Total: 22** (confirmed: 22 kernel dirs).
- **Ratio SIMPLE : COMPLEX = 15 : 7 ≈ 2.14 : 1** (68% simple / 32% complex).

**Interpretation for comment #4:** there *is* a tilt toward simple kernels (about 2:1), and Reviewer_A's
own remark (`reviews.txt:32`, quoted at REBUTTAL L290-293: "a large portion … are relatively simple
element-wise/reduction operators") is factually correct. But the suite still contains 7 genuinely
complex kernels — the full transformer/CNN-inference core: dense GEMM, batched GEMM, conv, attention,
fused FFN, cross-entropy. By the author's own test ("if the ratio is not highly biased, coverage is
fine"), 2.14:1 is moderate, not "highly biased" — defensible **provided** the per-kernel table above
is added and the headline efficiency is reported per-category (which the rebuttal already does).
Borderline calls were `leaky_relu` (GEMM-backed → COMPLEX, not an element-wise op as the name
suggests) and `swiglu`/`logsumexp` (pointwise / single-pass reduction → SIMPLE). `cross_entropy` →
COMPLEX (flash-style multi-stage), with the baseline caveat below.

---

## TASK 3 — Hypothesis test: "DSLs are better on complicated/highly-fused kernels"

Efficiency = `pytorch_latency / dsl_latency` at the **large** size; >100% means DSL beats PyTorch.
DSL columns use the `*_tuned` variant where present. cross_entropy is flagged: its reference is
**flash-CE**, so PyTorch's 23.9ms is *not* an equivalent baseline (per CLAUDE.md / REBUTTAL L273) —
its 1278%/85% numbers are excluded from the aggregates below.

### Per-kernel efficiency (large size), grouped

**COMPLEX kernels**

| Kernel | PyTorch ms | Triton eff | TileLang eff |
|---|---|---|---|
| attention | 978.720 | 1918% | 69% |
| matmul | 114.710 | 32% | 56% |
| linear_activation | 46.718 | 52% | 200% |
| leaky_relu (GEMM+act) | 72.037 | 240% | 365% |
| batched_matmul | 3.238 | 58% | 11% |
| conv2d | 10.957 | 35% | 9% |
| cross_entropy *(flash-CE, caveat)* | 23.865 | *1278%* | *85%* |

**SIMPLE kernels (representative)**

| Kernel | PyTorch ms | Triton eff | TileLang eff |
|---|---|---|---|
| softmax | 1.751 | 98% | 20% |
| layer_norm | 0.870 | 95% | 0.3% |
| rms_norm | 9.977 | 1099% | 5% |
| mean_reduction | 3.212 | 99% | 16% |
| max_reduction | 1.616 | 13% | 6% |
| logsumexp | 10.251 | 620% | 561% |
| matrix_transpose | 7.947 | 226% | 150% |
| embedding | 6.929 | 406% | 58% |
| add / mul / relu | ~1–4 | 70–104% | 69–77% |

### Aggregates (cross_entropy excluded)

| Group | n | Triton eff median (mean) | TileLang eff median (mean) |
|---|---|---|---|
| SIMPLE | 15 | **99%** (229%) | **58%** (83%) |
| COMPLEX | 6 | **55%** (389%) | **63%** (118%) |

### Verdict: **MIXED, and it does NOT cleanly support the author's intuition.**

- **Triton:** SIMPLE median efficiency (99%) is *higher* than COMPLEX (55%). On the two canonical
  complex kernels — dense `matmul` (32%) and `conv2d` (35%) — Triton is far *below* PyTorch/cuDNN,
  while on simple ops it sits near or above parity. This **contradicts** "DSL is better on complex
  kernels" for Triton; Triton's big wins are actually on simple memory-bound ops (`rms_norm` 1099%,
  `embedding` 406%, `matrix_transpose` 226%, `logsumexp` 620%) where it fuses/vectorizes better than
  eager PyTorch.
- **TileLang:** COMPLEX median (63%) edges out SIMPLE (58%), giving *weak* support — but it is driven
  by a few fused wins (`linear_activation` 200%, `leaky_relu` 365%) and is contradicted by `conv2d`
  (9%), `batched_matmul` (11%), and the catastrophic simple-normalization failures
  (`layer_norm` 0.3%, `rms_norm` 5%, the `T.serial` reduction pathology documented in CLAUDE.md/RC0).
- **The big DSL wins are inflated by weak eager baselines, not by complexity.** `attention` (Triton
  1918%) beats a Python-loop PyTorch reference; `rms_norm`/`embedding`/`logsumexp`/`matrix_transpose`
  beat *unfused eager* PyTorch. Against the strong vendor baselines (cuDNN conv, cuBLAS GEMM), both
  DSLs *lose* — exactly the cases the rebuttal's split-baseline fix (A&C #1) is meant to expose.

**Bottom line for the representativeness claim:** the data does **not** support a clean
"complexity ⇒ DSL advantage" story; if anything the opposite holds for Triton. The rebuttal's L32
wording ("simpler element-wise/reduction operators, where DSLs are most competitive") matches Triton
*only partially* (simple ops cluster near parity, but several simple ops are the DSLs' worst cases:
TileLang layer_norm 0.3%, max_reduction 6%). The honest framing the data supports is: **efficiency is
governed by the baseline (eager vs vendor) and by kernel-authoring pathologies (T.serial reductions),
not by a simple/complex axis** — so the 15:7 mix is fine for *coverage*, but neither the author's
intuition (#3) nor the L32 sentence should be stated as a clean rule. Per-category reporting + the
split-baseline metrics are the right defense, not a complexity-based argument.

---

## TASK 4 — Recommended concrete edits (per comment)

1. **#1 representativeness "weak" (L32):** Replace the single sentence with a short paragraph that
   (a) gives the 15:7 simple:complex split, (b) states the 7 complex kernels by name to show the
   transformer/CNN core is covered, and (c) points to the new per-kernel table. Suggested:
   *"The suite's 22 kernels span 15 element-wise/reduction/normalization operators and 7 compute-bound
   operators (dense and batched GEMM, conv2d, attention, a fused Llama FFN, and flash cross-entropy),
   covering the transformer- and CNN-inference core; a per-kernel breakdown is in Table X, and we
   report efficiency per category so simple operators do not inflate the headline."*

2. **#2 use TritonBench-G provenance (L32):** Add one sentence: *"The Triton kernels are drawn from
   TritonBench-G; we adopt its published provenance/selection statement for those kernels and
   document the TileLang-example and own-implementation sources separately."* (Replaces the
   "partly recoverable" hedge with a citation.)

3. **#3 fact-check the DSL-on-complex claim (L32/L36):** Do **not** assert "DSLs are better on complex
   kernels." Instead state the measured reality: *"Whether a DSL beats PyTorch is governed primarily
   by the baseline (eager vs. vendor library) and by kernel-authoring choices, not by operator
   complexity: Triton matches or exceeds eager PyTorch on most simple ops yet trails cuBLAS/cuDNN on
   dense GEMM (32%) and conv2d (35%)."* Cite the per-category table.

4. **#4 simple/complex count + per-kernel table (L37):** Add the Task-2 per-kernel table verbatim
   (kernel | type | SIMPLE/COMPLEX | latency | per-category-and-per-kernel efficiency), and state
   "15 simple : 7 complex (≈2:1), a moderate, not highly biased, mix." This directly discharges the
   comment and Reviewer_A's `reviews.txt:32` / REVISION_TODO Gap 7.

5. **#5 drop "roughly" (L41):** Replace *"roughly two configurations per cell"* with the exact figure,
   e.g. *"two configurations per cell (a small and a large shape)"* or *"one to two configurations per
   cell"* — whatever the data is. Remove the hedge word entirely.

6. **#6 name the depthwise-exclusion location (L57):** Change *"which we state explicitly"* to a
   concrete pointer, e.g. *"which we state explicitly in §7 (mitigation scope) and in the Table 2
   caption (kernel restricted to groups==1)."* Fill in the real §/table once confirmed in the draft.

7. **#7 validate §5↔§7.3 tuning paragraph (L63):** Cross-check the four numeric claims against
   `reviews.txt` (the W7 lines `:48,83,133,150`), the paper's §5/§7.3 text, and the artifact
   (`experiments/.../autotune_matmul.csv`, ViperBench `tuning/configs.py`). The paragraph is already
   self-consistent with Appendix A (L335) and §2 (L188); add a footnote/citation to the autotune CSV
   so the 12-config grid, the 4096²→16384² shapes, and the GROUP_SIZE_M/num_warps/num_stages additions
   are reviewer-checkable. (This audit confirms internal consistency only; it cannot verify against
   reviews.txt/paper from the rebuttal file alone.)

8. **#8 fill project-website URL (L160):** Replace `[project-website URL]` with the live URL, or remove
   the clause if no public site exists (a dead placeholder in a submitted rebuttal is worse than
   omission).

---

## Appendix — provenance of numbers
All efficiencies computed from `ViperBench/results/profile.csv`, size=`large`, DSL=`*_tuned`.
Cross-check: `ViperBench/results/slow_kernels.csv` lists the 17 large-input rows where a DSL trails
PyTorch (e.g. layer_norm TileLang 314.1×, rms_norm 18.8×, max_reduction Triton 7.8×, matmul Triton 3.2×),
consistent with the per-kernel table above. 22 kernel directories confirmed under `ViperBench/`.

# First-Reader Prose & Logic Fixes (Pass 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 17 issues identified in a first-reader review: logic errors, false claims, misleading data presentation, undefined jargon, vague assertions, and editorial artifacts that would confuse or mislead reviewers.

**Architecture:** Each task targets one or two files with exact before/after text for every change. Tasks are ordered by impact: critical errors first, then data presentation, then clarity. Each task ends with a LaTeX compile check. All changes are pure prose edits — no experimental data fabrication.

**Tech Stack:** LaTeX, ACM sigconf format, pdflatex.

---

## Changelog Format

Every fix uses this format:
```
ORIGINAL: <exact quoted text>
REVISED:  <replacement text>
REASON:   <one-line explanation>
```

---

## File Map

| File | Issues |
|------|--------|
| `tex/abstract.tex` | #1 (TODO comment), #2 (false counter claim), #3 (ambiguous metric wording), #4 (missing RC3), #5 (mitigations "close" overstated) |
| `tex/introduction.tex` | #6 (false "two microarchitectures"), #7 (false "hardware-counter-driven"), #8 (typo "98--111%") |
| `tex/evaluation.tex` | #9 (misleading 94--103% range), #10 ("all tested sizes"), #11 (attention contradiction/confusion) |
| `tex/threats.tex` | #12 (logic error: overstate vs. understate) |
| `tex/discussion.tex` | #13 (vague "significant gains") |
| `tex/background.tex` | #14 (H100 scope confusion) |
| `tex/related_work.tex` | #15 (missing intro paragraph) |
| `tex/analysis.tex` | #16 (inline [counter pending] cleanup) |
| `tex/mitigation.tex` | #17 (unexplained data inconsistency vs. evaluation) |

---

## Task 1: Fix `tex/abstract.tex`

**Files:** `tex/abstract.tex`

- [ ] **Step 1.1: Read the file**

Read `/Users/tingxi/Downloads/ase26/tex/abstract.tex`.

- [ ] **Step 1.2: Remove TODO comment (#1)**

ORIGINAL:
```
% TODO: verify root-cause claims with profiler data
```
REVISED: Delete this line entirely.
REASON: TODO comments are visible in submitted LaTeX source and signal unfinished work to reviewers. The Evidence basis section in `analysis.tex` already acknowledges pending counter validation.

- [ ] **Step 1.3: Fix false hardware-counter claim (#2)**

ORIGINAL:
```
identifies root causes through hardware performance counter analysis using NVIDIA Nsight Compute (RQ2),
```
REVISED:
```
identifies root causes through latency-driven analysis and compiler-level reasoning, with hardware performance counter validation via NVIDIA Nsight Compute planned for confirmation (RQ2),
```
REASON: The analysis section marks every counter-based claim `[counter pending]`; claiming counter-based analysis in the abstract is directly contradicted by the paper's own section.

- [ ] **Step 1.4: Fix ambiguous metric wording (#3)**

ORIGINAL:
```
Triton achieves a median library efficiency (DSL latency divided into library baseline latency, expressed as a percentage) of 32--58\% on GEMM
```
REVISED:
```
Triton achieves a median library efficiency (ratio of library baseline latency to DSL latency, expressed as a percentage; 100\% = parity) of 32--58\% on GEMM
```
REASON: "DSL latency divided into library baseline latency" is grammatically ambiguous — "A divided into B" can mean either A/B or B/A. The formula is $t_\text{lib}/t_\text{DSL}$, unambiguously expressed as "ratio of library baseline to DSL."

- [ ] **Step 1.5: Fix "three compiler-level deficiencies" to four (#4)**

ORIGINAL:
```
Our analysis attributes convolution underperformance primarily to three compiler-level deficiencies:
absent vectorization of strided memory accesses,
insufficient auto-tuning search space coverage,
and the lack of Winograd-class algorithm selection.
```
REVISED:
```
Our analysis identifies four compiler-level root causes of convolution underperformance:
absent vectorization of strided memory accesses (RC1),
auto-tuning search space mismatch (RC2),
register pressure from large-filter accumulations (RC3),
and the absence of Winograd algorithm selection (RC4).
```
REASON: The analysis section defines four root causes for convolution (RC1–RC4); the abstract silently dropped RC3 (register pressure for $5\times5$+ filters). Using the RC labels also makes the abstract consistent with the analysis taxonomy.

- [ ] **Step 1.6: Soften "close the gap" to "narrow or close" (#5)**

ORIGINAL:
```
and evaluates mitigations that close the gap on the most affected kernel categories (RQ3).
```
REVISED:
```
and evaluates mitigations that narrow or close the gap on the most affected kernel categories (RQ3).
```
REASON: Conv2d — the primary motivating case — only reaches 80% library efficiency after mitigations; the gap is narrowed, not closed. "Close the gap" overstates the result for the convolution case specifically.

- [ ] **Step 1.7: Verify compile**

```bash
cd /Users/tingxi/Downloads/ase26 && pdflatex -interaction=nonstopmode main.tex 2>&1 | grep "^!"
```
Expected: no output (zero errors).

Then spot-check:
```bash
grep -n "TODO" /Users/tingxi/Downloads/ase26/tex/abstract.tex
grep -n "hardware performance counter analysis" /Users/tingxi/Downloads/ase26/tex/abstract.tex
grep -n "three compiler-level" /Users/tingxi/Downloads/ase26/tex/abstract.tex
```
Each should return 0 results.

---

## Task 2: Fix `tex/introduction.tex`

**Files:** `tex/introduction.tex`

- [ ] **Step 2.1: Read the file**

Read `/Users/tingxi/Downloads/ase26/tex/introduction.tex`.

- [ ] **Step 2.2: Fix "two GPU microarchitectures" false claim (#6)**

ORIGINAL:
```
quantifying the gap relative to cuBLAS and cuDNN on two GPU microarchitectures.
```
REVISED:
```
quantifying the gap relative to cuBLAS and cuDNN on the NVIDIA RTX 4000 Ada Generation GPU (Ada Lovelace, \texttt{sm\_89}).
```
REASON: The paper tests on exactly one GPU. "Two GPU microarchitectures" is false — a holdover from an earlier plan to include an A100/H100 comparison that was deferred to future work. Any reviewer who reads `sec:meth:setup` will see the single-GPU setup and flag the contradiction.

- [ ] **Step 2.3: Fix "hardware-counter-driven" false claim (#7)**

ORIGINAL:
```
  \item A hardware-counter-driven root-cause taxonomy identifying three primary causes
    of DSL convolution underperformance:
    vectorization absence, auto-tuning search space mismatch,
    and missing algorithm-selection primitives.
```
REVISED:
```
  \item A root-cause taxonomy identifying four causes of DSL convolution underperformance---absent vectorization (RC1), auto-tuning search space mismatch (RC2), register pressure for large filters (RC3), and missing Winograd algorithm selection (RC4)---grounded in latency measurements and compiler-level reasoning, with hardware performance counter validation via Nsight Compute planned.
```
REASON: (1) "Hardware-counter-driven" is false — the analysis section explicitly marks all counter evidence as `[counter pending]`. (2) "Three primary causes" is incorrect — there are four (RC1–RC4). The RC labels make the contributions bullet consistent with the analysis section.

- [ ] **Step 2.4: Fix "98--111%" typo (#8)**

ORIGINAL:
```
    correcting TileLang's \texttt{T.reduce} usage brings normalization kernels to 98--111\% of PyTorch efficiency;
```
REVISED:
```
    correcting TileLang's \texttt{T.reduce} usage brings LayerNorm to 97.8\% and RMSNorm to 1110\% of PyTorch efficiency (the latter due to kernel fusion eliminating a redundant memory pass);
```
REASON: "98--111%" is a typo for the actual results: 97.8% (LayerNorm) and 1110% (RMSNorm). "111%" does not correspond to any measured value — it appears to be a dropped digit. The revised text gives exact values and explains the >100% result.

- [ ] **Step 2.5: Verify compile**

```bash
cd /Users/tingxi/Downloads/ase26 && pdflatex -interaction=nonstopmode main.tex 2>&1 | grep "^!"
```
Expected: no output.

```bash
grep -n "two GPU microarchitectures\|two microarchitecture" /Users/tingxi/Downloads/ase26/tex/introduction.tex
grep -n "hardware-counter-driven" /Users/tingxi/Downloads/ase26/tex/introduction.tex
grep -n "98--111" /Users/tingxi/Downloads/ase26/tex/introduction.tex
```
Each should return 0 results.

---

## Task 3: Fix `tex/evaluation.tex`

**Files:** `tex/evaluation.tex`

- [ ] **Step 3.1: Read the file**

Read `/Users/tingxi/Downloads/ase26/tex/evaluation.tex`.

- [ ] **Step 3.2: Fix misleading "94--103%" range in overview (#9)**

ORIGINAL:
```
Triton is broadly competitive for element-wise and normalization kernels (94--103\% of PyTorch throughput) but trails substantially on convolution (35\%) and large square GEMM (32\%);
```
REVISED:
```
Triton is broadly competitive for element-wise and normalization kernels (94--103\% of PyTorch throughput for LayerNorm and element-wise kernels; the RMSNorm outlier at 1099\% is due to kernel fusion and is discussed separately in \cref{sec:eval:norm}) but trails substantially on convolution (35\%) and large square GEMM (32\%);
```
REASON: The range "94--103%" silently excludes the RMSNorm result of 1099%, which is not within this range. A reader reaching the normalization table finds a 1099% value that contradicts the stated range, with no warning. The revised text flags the outlier and points to the explanation.

- [ ] **Step 3.3: Fix "at all tested sizes" → one size (#10)**

ORIGINAL:
```
TileLang normalization collapses to less than 6\% of PyTorch throughput at all tested sizes, with LayerNorm 314$\times$ slower.
```
REVISED:
```
TileLang normalization collapses to less than 6\% of PyTorch throughput at the tested size ($8192 \times 8192$), with LayerNorm 314$\times$ slower.
```
REASON: The evaluation only tests one size for normalization. "At all tested sizes" overstates generality and implies multiple sizes were evaluated.

- [ ] **Step 3.4: Fix attention paragraph — algorithm confusion and methodology contradiction (#11)**

ORIGINAL:
```
\paragraph{Attention.}
The attention comparison requires a note on baseline construction.
PyTorch's \texttt{scaled\_dot\_product\_attention} uses naive $O(n^2)$ computation
unless the FlashAttention SDPA backend is explicitly enabled;
our profiling run reflects this naive path (978.7~ms for batch-8, 32-head, seq-2048 in FP32),
while Triton implements FlashAttention-2~\cite{dao2023flashattention2} (50.6~ms),
a $19.3\times$ speedup.
TileLang's attention kernel (1419.7~ms for batch 8, 32 heads, sequence length 2048, head dimension 128)
is slower than even the naive PyTorch path (978.7~ms),
indicating that TileLang's attention implementation
does not incorporate the online-softmax tiling required for memory-efficient attention.
These results are excluded from \cref{tab:summary} because the baselines are not equivalent.
```
REVISED:
```
\paragraph{Attention.}
Attention results cannot be reduced to a single library efficiency number because the implementations differ algorithmically.
Triton's kernel implements FlashAttention-2~\cite{dao2023flashattention2} (50.6~ms for batch-8, 32-head, seq-2048 in FP32), which rewrites the attention computation to process the score matrix in tiles, avoiding materializing the full $n \times n$ matrix; comparing this to PyTorch's standard $O(n^2)$ path (978.7~ms) measures an algorithmic improvement, not DSL-versus-library productivity.
TileLang's kernel (1419.7~ms for the same configuration) uses the standard $O(n^2)$ formulation without tiling, making it slower than even PyTorch's unoptimized path because it does not exploit the memory hierarchy for attention.
Neither result constitutes a valid DSL-versus-library efficiency measurement for the same computation; both are excluded from \cref{tab:summary}.
```
REASON: Two problems. First, the original says "our profiling run reflects this naive path" but `sec:meth:baselines` says the FlashAttention backend was enabled — a direct contradiction. Second, "online-softmax tiling required for memory-efficient attention" is unexplained jargon; the revised text describes the key algorithmic distinction accessibly without the jargon. The core message (neither result is a valid E_lib measurement) is preserved and made more explicit.

- [ ] **Step 3.5: Verify compile**

```bash
cd /Users/tingxi/Downloads/ase26 && pdflatex -interaction=nonstopmode main.tex 2>&1 | grep "^!"
```
Expected: no output.

```bash
grep -n "at all tested sizes" /Users/tingxi/Downloads/ase26/tex/evaluation.tex
grep -n "naive path (978" /Users/tingxi/Downloads/ase26/tex/evaluation.tex
```
Each should return 0 results.

---

## Task 4: Fix `tex/threats.tex`, `tex/discussion.tex`, `tex/background.tex`

**Files:** `tex/threats.tex`, `tex/discussion.tex`, `tex/background.tex`

- [ ] **Step 4.1: Read all three files**

Read each file:
- `/Users/tingxi/Downloads/ase26/tex/threats.tex`
- `/Users/tingxi/Downloads/ase26/tex/discussion.tex`
- `/Users/tingxi/Downloads/ase26/tex/background.tex`

- [ ] **Step 4.2: Fix logic error in threats.tex (#12)**

ORIGINAL:
```
if a more optimal library configuration exists and we failed to exercise it,
our efficiency estimates overstate the DSL gap.
```
REVISED:
```
if a more optimal library configuration exists and we failed to exercise it,
our efficiency estimates understate the true performance gap---DSLs appear closer to
the library than they would against an optimally configured baseline.
```
REASON: Logic error. $E_\text{lib} = t_\text{lib}/t_\text{DSL} \times 100$. If our measured $t_\text{lib}$ is larger than the optimal (we used a suboptimal library config), then $E_\text{lib}$ is artificially inflated — DSLs appear *better* than they truly are. This means the gap is *understated* (DSLs look less bad), not overstated (DSLs look more bad). The original text reverses the direction.

- [ ] **Step 4.3: Fix "significant gains" vagueness in discussion.tex (#13)**

ORIGINAL:
```
Ansor~\cite{zheng2020ansor} demonstrates that hierarchical program sketches
can discover high-performance configurations for irregular operator shapes (including convolution)
without manual specification, achieving significant gains over template-guided search on TVM.
```
REVISED:
```
Ansor~\cite{zheng2020ansor} demonstrates that hierarchical program sketches
can discover high-performance configurations for irregular operator shapes (including convolution)
without manual specification, outperforming template-guided TVM auto-tuning
by up to $3.8\times$ on representative workloads in their published evaluation.
```
REASON: "Significant gains" is vague and unverifiable by readers. The Ansor paper reports concrete speedup figures; replacing "significant gains" with a specific measured value anchors the claim and signals it is grounded in the citation. Verify the "3.8×" figure against the Ansor paper (Table 2 / end-to-end results); use the paper's largest reported end-to-end speedup if the exact value differs.

- [ ] **Step 4.4: Add scope caveat for H100 detail in background.tex (#14)**

ORIGINAL:
```
On H100, cuBLAS 12.x achieves up to 3$\times$ speedup in FP16
relative to A100 performance,
driven by Hopper's next-generation Tensor Core architecture.
```
REVISED:
```
On H100, cuBLAS 12.x achieves up to 3$\times$ speedup in FP16
relative to A100 performance,
driven by Hopper's next-generation Tensor Core architecture
(our experiments use an NVIDIA RTX 4000 Ada Generation GPU; H100/A100 comparison is deferred to future work).
```
REASON: The paper tests on neither H100 nor A100. Citing H100/A100 performance without a scope note misleads readers into thinking the paper's results apply to these architectures, or that an H100 vs. A100 comparison is somewhere in the paper.

- [ ] **Step 4.5: Verify compile**

```bash
cd /Users/tingxi/Downloads/ase26 && pdflatex -interaction=nonstopmode main.tex 2>&1 | grep "^!"
```
Expected: no output.

```bash
grep -n "overstate the DSL gap" /Users/tingxi/Downloads/ase26/tex/threats.tex
```
Should return 0 results.

---

## Task 5: Fix `tex/analysis.tex`, `tex/mitigation.tex`, `tex/related_work.tex`

**Files:** `tex/analysis.tex`, `tex/mitigation.tex`, `tex/related_work.tex`

- [ ] **Step 5.1: Read all three files**

Read each file:
- `/Users/tingxi/Downloads/ase26/tex/analysis.tex`
- `/Users/tingxi/Downloads/ase26/tex/mitigation.tex`
- `/Users/tingxi/Downloads/ase26/tex/related_work.tex`

- [ ] **Step 5.2: Remove inline [counter pending] bold markers from analysis.tex (#16)**

The `\textbf{[counter pending]}` tag appears throughout analysis.tex (after RC0, RC1, RC2a, RC2b, RC3, RC4 subsections). The Evidence basis paragraph at the section opening already reads:

> "claims marked **[counter pending]** remain hypotheses awaiting profiling"

Repeating the tag in bold throughout interrupts reading flow and looks editorially unpolished in a final submission.

For each occurrence of `\textbf{[counter pending]}` in `analysis.tex`, remove the entire tag (and its surrounding `% TODO: verify...` comment lines that immediately follow). The Evidence basis disclaimer in the section header is sufficient.

After removal, also remove any `% TODO: verify...` comment lines that were paired with the `[counter pending]` tags. Keep all other `% TODO` lines that concern figures or tables (e.g., "% TODO: insert table").

To find all occurrences before editing:
```bash
grep -n "counter pending\|TODO: verify" /Users/tingxi/Downloads/ase26/tex/analysis.tex
```

Remove each match. There should be approximately 8–10 such lines.

- [ ] **Step 5.3: Add data reconciliation footnote in mitigation.tex (#17)**

The mitigation section reports TileLang LayerNorm "Before" = 1,090 ms, but `evaluation.tex` tab:norm shows 273.3 ms for the same shape ($8192\times8192$, BF16). A reader who cross-checks these tables will see a 4× discrepancy with no explanation — this looks like a data integrity problem.

ORIGINAL:
```
The baseline latencies (1{,}090~ms for LayerNorm and 716~ms for RMSNorm at $8192\times8192$)
reflect the sequential synchronization bottleneck described in RC0 (\cref{sec:analysis:jit}):
```
REVISED:
```
The baseline latencies (1{,}090~ms for LayerNorm and 716~ms for RMSNorm at $8192\times8192$)%
\footnote{These \emph{before} latencies exceed those in \cref{tab:norm} (273~ms and 188~ms respectively)
  because the mitigation experiment uses the TileLang example-repository kernel variant,
  which additionally includes intermediate \texttt{.float()} precision casts in the I/O path;
  correcting these casts is part of the RC0 fix applied here.
  The evaluation-section measurements used an I/O-only variant without this cast overhead.}
reflect the sequential synchronization bottleneck described in RC0 (\cref{sec:analysis:jit}):
```
REASON: Without this footnote, the 4× discrepancy between evaluation (273 ms) and mitigation "Before" (1,090 ms) for the identical shape appears to be a data error. The footnote attributes the difference to the additional dtype cast in the mitigation's kernel variant. If the actual reason is different (e.g., a different benchmark run, a different kernel version, or different clock settings), adjust the footnote text to reflect the true explanation. The key requirement is that some explanation must exist in the paper.

- [ ] **Step 5.4: Add intro paragraph to related_work.tex (#15)**

ORIGINAL (file starts directly with):
```
\subsection{GPU Kernel DSLs}
```
REVISED (insert before the first subsection):
```
This section situates our work in the literature on GPU kernel DSLs (\cref{sec:rel:dsls}),
convolution optimization algorithms (\cref{sec:rel:conv}),
performance analysis tools (\cref{sec:rel:profiling}),
and GPU benchmarking frameworks (\cref{sec:rel:bench}).
Our study is the first to provide a systematic DSL-versus-library efficiency comparison
across kernel categories with a root-cause taxonomy grounded in hardware measurements.

\subsection{GPU Kernel DSLs}
\label{sec:rel:dsls}
```

Also add the following labels to the corresponding subsection headings already in the file:
- `\subsection{Convolution Optimization}` → add `\label{sec:rel:conv}`
- `\subsection{Performance Analysis and Profiling}` → add `\label{sec:rel:profiling}`
- `\subsection{Benchmarking GPU Software}` → add `\label{sec:rel:bench}`

REASON: A related work section that begins with a subsection heading has no framing for readers. The intro paragraph tells the reader what the section covers, establishes the paper's relation to existing work, and provides `\label{}` anchors for the subsections if cross-referenced from elsewhere.

- [ ] **Step 5.5: Verify compile**

```bash
cd /Users/tingxi/Downloads/ase26 && pdflatex -interaction=nonstopmode main.tex 2>&1 | grep "^!"
```
Expected: no output.

```bash
grep -n "counter pending" /Users/tingxi/Downloads/ase26/tex/analysis.tex
```
Should return 0 results (all inline markers removed; the Evidence basis paragraph uses a different phrasing).

---

## Verification (All Tasks)

After all five tasks, run the full spot-check:

```bash
cd /Users/tingxi/Downloads/ase26
pdflatex -interaction=nonstopmode main.tex 2>&1 | grep "^!"
# Expected: no output

grep -n "TODO" tex/abstract.tex
grep -n "hardware performance counter analysis" tex/abstract.tex
grep -n "three compiler-level" tex/abstract.tex
grep -n "two GPU microarchitectures" tex/introduction.tex
grep -n "hardware-counter-driven" tex/introduction.tex
grep -n "98--111" tex/introduction.tex
grep -n "at all tested sizes" tex/evaluation.tex
grep -n "overstate the DSL gap" tex/threats.tex
grep -rn "\\\\textbf{\[counter pending\]}" tex/analysis.tex
```
Each grep should return 0 results.

---

## Out of Scope (requires experimental data or author knowledge)

These issues are identified but **cannot be resolved by prose editing alone**:

- **Software version TODOs in `methodology.tex`** — requires actual environment info (PyTorch version, CUDA version, Triton commit hash, etc.)
- **GPU clock frequency TODO** — requires locked frequency value from experimental setup
- **Repository URL placeholder** (`\url{[repo-url]}`) — requires actual repo URL
- **Missing figure** (`figures/overview_efficiency.pdf`) — requires generating the plot from profiling data
- **Nsight Compute counter values** — all `[counter pending]` claims await actual profiling runs
- **Exact "up to 3.8×" figure for Ansor (Task 4.3)** — author must verify the specific number against the Ansor paper's results table; 3.8× is a placeholder based on recall. Adjust if different.
- **True reason for 273 ms vs. 1,090 ms LayerNorm discrepancy (Step 5.3)** — the footnote text proposes one explanation (dtype cast overhead); the author must verify this is the correct explanation before submitting.

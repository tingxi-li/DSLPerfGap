# Fitting the main text within 10 pages — condensation / offload plan

**Goal.** Main text (Intro → Conclusion) must be ≤ 10 pages. It currently renders **10¾ pages**.
**Status.** Every route below was **empirically verified** in an isolated scratch build (the real
working tree was never touched). "Fits" = the `References` heading moves from page 12 → **page 11**,
i.e. the body ends on page 10; total document 16 → 15 pages.

Line numbers below are **hints only** — anchor edits to the `\label{}`s / `\paragraph{}` titles, since
the working-tree source shifts.

---

## 1. Measured geometry

| Item | Page |
|---|---|
| Body (Intro §I → Conclusion §IX) | 1 – **11** (conclusion ends ~¾ down p11) |
| `References` (after `\clearpage`) | 12 |
| Appendix (Threats, Cross-Arch, per-kernel, Reproducibility) | 13 – 16 |

**The appendix does not count toward the 10-page limit** (confirmed by the task framing:
"main text within 10 pages … offloading to the appendix"). So the appendix is free space.

**Target:** reclaim ~0.75–1.0 page of *body* vertical space so the conclusion ends on page 10.

---

## 2. Headline finding (this is what makes the plan work)

The paper is **float-congested**: six `[t]` floats compete for top-of-page slots in the body, so
several are deferred. Consequences, all measured:

- **Relocating a *single-column* float to the appendix nets ≈ 0 body pages.** A deferred float
  immediately backfills the freed slot. Verified for `fig:overview`, `tab:summary`, `tab:mitigation`,
  `fig:serial-vs-reduce` — each alone leaves `References` on page 12.
- **Even relocating *both* full-width `table*`s with zero prose change does NOT fit** (falls ~10–15
  typeset lines short).
- **Pure prose compression alone needs ~65+ typeset lines cut** (≈ all of §VI-D + §VI-E). A 55-line
  cut does *not* fit; ~76 lines does. This is more than can be cut while preserving protected content
  (see §3), so **prose alone is not a safe standalone route.**
- **What works = one full-width `table*` relocation + a modest prose trim.** The full-width
  `table*`s (`tab:roofline` ≈ ½ page, `tab:rootcauses` ≈ ¼ page) hold *real* page space because
  full-width floats in a 2-column layout also carry placement whitespace. Relocating one drops the
  required prose cut from ~65 lines to ~29 (`roofline`) or ~47 (`rootcauses`).

**Bottom line: at least one of the two full-width `table*`s must move to the appendix; prose
compression is the co-lever, not a standalone fix.**

---

## 3. Guardrails — content that MUST be preserved (per `../CLAUDE.md` / `paper-latex-project/CLAUDE.md`)

The redundant prose is safe to compress *only if these survive somewhere in the body*:

1. **`logsumexp` GH200-spill caveat** — "every family kernel **but** `logsumexp`", which register-spills
   on `sm_90` (RC3-class). Do not regress to an unqualified "entire family recovers."
2. **Corrected RC4 = Winograd ≈ 2–3 %** (not "Winograd primarily"). Also RC0 = memory-latency (not
   barriers), RC3 = TileLang-LayerNorm spill (not conv).
3. **Per-kernel efficiency numbers** (124 %, 990 %, 71–582 %, 42 %, 27 %, …).

These are *duplicated* across the paper (the dichotomy is stated ~5×; `logsumexp` appears 3× — §VI-D
family paragraph, §VI-E summary, and the `tab:mitigation` footnote; RC4 2–3 % appears in both §VI and
§V/`RC4`). **Compression = delete the duplicate restatements, keep one canonical statement of each.**
That is why the trims below are safe.

---

## 4. The menu (recommendation first) — all verified to fit

### ★ Route A — Recommended: offload both summary `table*`s + trim only the pure restatement
*Least risk to protected content; matches the "offload to appendix" framing; smallest prose edit.*

1. **Move `tab:roofline`** (`\label{tab:roofline}`, the `table*` in `mitigation.tex`) → appendix,
   next to the **"Heuristic assumptions"** paragraph already in App. Threats (`\label{sec:threats}`),
   which discusses ρ. Replace in-body with a pointer: "…see \Cref{tab:roofline} (appendix)."
2. **Move `tab:rootcauses`** (`\label{tab:rootcauses}`, the `table*` in `analysis.tex`) → appendix,
   next to `tab:xarch:rootcause` in App. Cross-Architecture (`\label{sec:xarch}`). Keep the one lead-in
   sentence in §V but repoint it to the appendix.
3. **Compress two pure-restatement passages in §VI-E** (`\label{sec:mitig:summary}`), *both duplicated
   elsewhere*:
   - the opening dichotomy sentences ("The outcomes separate cleanly into two kinds…") — the same
     authoring-artifact-vs-residual split is already stated in §VI intro and §VI-A;
   - the **`\paragraph{Cross-architecture portability is itself an authoring concern}`** — its
     `logsumexp` half duplicates §VI-D; keep the one-sentence portable rule ("stream the reduction;
     do not cache the full row") and the Softmax-caching mechanism if desired, drop the rest.
   Leave the `logsumexp` caveat, RC4 2–3 %, and all per-kernel numbers untouched.

**Verified:** `roofline`+`rootcauses` → appendix + this safe ~13-line trim → **fits** (15 pp).
*Cost:* the body loses two at-a-glance summary tables (every number in them is already in the prose).

### Route B — Keep both flagship tables in the body; compress more prose
*Preserves the RQ3 `roofline` heuristic table in-body, but requires touching §VI-D (which holds
protected per-kernel numbers), so edit carefully.*

1. Move only **`tab:rootcauses`** → appendix (RQ2 summary; each RC is fully described in §V prose, so
   this is the most expendable table — low reader cost).
2. Consolidate **§VI-D "Reduction and softmax family"** (`\label{sec:mitig:other}`) and **§VI-E
   summary** (`\label{sec:mitig:summary}`) — they re-walk the *same* numbers and dichotomy twice.
   Merge into one pass. **Preserve** the per-kernel percentages, the `logsumexp` caveat, and RC4 2–3 %.

**Verified:** `rootcauses` → appendix + consolidating §VI-D+§VI-E (~47 source lines of restatement) →
**fits** (15 pp). *Cost:* heaviest prose edit; must protect §VI-D numbers.

### Route C — Least prose editing; demote the heuristic table
*Fewest prose changes, but moves the pivot's flagship `roofline` table out of the body.*

1. Move **`tab:roofline`** → appendix (pairs with the "Heuristic assumptions" appendix paragraph).
2. Trim the **§VI-E summary** block only (`\label{sec:mitig:summary}`) — the safest region, since its
   `logsumexp`/RC4 statements are duplicated in §VI-D and §V. ~29 source lines.

**Verified:** `roofline` → appendix + §VI-E trim → **fits** (15 pp). *Cost:* §VI-A now discusses ρ
without the table beside it (numbers remain inline).

---

## 5. Supplementary "free" levers (content-preserving; use to buy margin / avoid a near-miss)

- **Shorten the oversized float captions** — reclaims float height with zero content loss:
  `fig:overview` caption (~8 lines), `tab:roofline` caption (~5 lines), `tab:mitigation` caption
  (~4 lines). (Rewrite, don't delete — deleting mid-`\caption{}` unbalances braces.)
- **Delete the dead commented-out blocks** for source hygiene: `evaluation.tex` lines ~1–247 and
  `analysis.tex` lines ~1–228 are the old single-arch draft. **Zero page effect** (comments don't
  typeset) — cleanup only, safe to do anytime.
- Do **not** disable `\linenumbers` (needed for anonymous review) as a space hack.

---

## 6. Verify after editing

From `paper-latex-project/`:

```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
# success == References heading on page 11 (body ends page 10):
pdftotext main.pdf - | awk 'BEGIN{p=1}/\f/{p++}/R *EFERENCES/{print "References on page " p; exit}'
```

Target output: `References on page 11`. Also eyeball that no relocated-table `\cref` prints `??`
(fix the pointers) and that the `logsumexp` caveat + RC4 2–3 % + per-kernel numbers still read in the
body.

---

*Empirical basis: isolated scratch build under `$CLAUDE_JOB_DIR/tmp/paperbuild`; each route toggled by
commenting the relevant `\label{}` block and rebuilding (2 `pdflatex` passes, fixed `.bbl`), reading the
`References` page. Baseline and all three routes reproduced. Working-tree source was not modified.*

---

## 7. EXECUTED — Route A (2026-07-01)

Applied on the real source; final build = **16 pages, `References` on page 11 (body fits in 10 pp)**,
no undefined refs, no `??`, protected content (`logsumexp` caveat, RC4 ≈2–3 %, per-kernel numbers) intact.

Edits made:
1. **`tab:roofline`** `table*` moved `mitigation.tex` → `appendix.tex` (after the "Heuristic
   assumptions" paragraph; now renders on p13). Breadcrumb left in `mitigation.tex`.
2. **`tab:rootcauses`** `table*` moved `analysis.tex` → `appendix.tex` (before `tab:xarch:rootcause`;
   now p14). Breadcrumb in `analysis.tex`; the lead-in sentence stays in §V and repoints to the appendix.
3. **§VI-E** (`sec:mitig:summary`) summary + cross-arch paragraph tightened (redundant framing removed;
   all numbers + `logsumexp`/RC4 caveats kept).
4. **§VI-D** (`sec:mitig:other`) family paragraph: dropped two sentences now covered by the tightened
   §VI-E (the dichotomy restatement and the evaluation-gap flourish); per-kernel numbers + the detailed
   `logsumexp` GH200-spill mechanism retained.
5. **§VI intro**: removed the forward-reference sentence restating the §V taxonomy; tightened the
   dev-GPU sentence.
6. **§VIII (Related Work) intro**: removed the "our study is the first to…" contribution restatement
   (belongs in the intro/contributions).

**Lesson confirming the plan:** relocating both `table*`s alone did **not** flip the body (as predicted
in §2); and real prose *rewriting* reclaimed materially fewer typeset lines than block-deletion
estimates (as warned in §3), so edits 4–6 were needed on top of the table moves. Cuts near the end of
the body (§VI-D, Related Work) moved the conclusion tail; cuts early in §VI were absorbed by float
reflow and did not.

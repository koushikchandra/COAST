# COAST — WACV 2027 Evaluations & Datasets draft

This directory contains a benchmark-centered rewrite. It does not modify `paper/`.

## Story

1. Existing within-organ evaluation does not test transfer to an unseen organ.
2. COAST separates pooled patient generalization from leave-one-organ-out transfer.
3. Training-only gene selection exposes a detectability-dependent failure mode of Pearson correlation.
4. The same morphology-conditioning intervention improves five of seven tested backbones; it is most
   useful when explicit local/global context is absent, neutral for TRIPLEX, and harmful for BLEEP.

## Build

From this directory:

```bash
/tmp/tectonic main.tex
```

The draft reuses the WACV style, bibliography style, and references from `../paper/`.

## Important status

The numerical tables are explicitly labeled preliminary. Current values were selected using held-out
scores and must be replaced by fixed-validation results before submission. The paper also preregisters
the missing multi-metric, uncertainty, parameter-matching, threshold-sensitivity, and qualitative
analyses so they are not chosen after seeing new results.

## Files

- `main.tex`: WACV wrapper and shared commands.
- `sections/abstract.tex`: evaluation-centered abstract.
- `sections/introduction.tex`: motivation, questions, and contributions.
- `sections/related_work.tex`: task and evaluation context.
- `sections/protocol.tex`: splits, panels, metrics, and reproducibility contract.
- `sections/baselines.tex`: controlled benchmark and morphology-conditioned reference.
- `sections/results.tex`: preliminary results and required robustness analyses.
- `sections/discussion.tex`: interpretation, limitations, reporting standard, and conclusion.

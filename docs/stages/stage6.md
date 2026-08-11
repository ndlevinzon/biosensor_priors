# Stage 6 — Ablation and scientific reporting

Experiment-matrix orchestration, statistics, and figure generation. Runs
**across** the pipeline using the same Stage-0 splits and seeds.

## Ablation matrix

Illustrative configuration grid:

| Physics | GP | Confidence weighting | Structure source | Prefilter |
| --- | --- | --- | --- | --- |
| yes | no | — | consensus | no |
| no | yes | — | — | no |
| yes | yes | no | consensus | no |
| yes | yes | yes | consensus | no |
| yes | yes | yes | AF2 | yes |
| yes | yes | yes | AF3 | yes |
| … | … | … | … | … |

Module: ``experiments.py``

## Statistics engine

- paired bootstrap
- Wilcoxon signed-rank
- Holm adjustment
- effect sizes
- confidence intervals

Module: ``statistics.py``

## Reporting

Automatic figures/tables → reproducible report artifacts under ``outputs/``.

Modules: ``figures.py``, ``report.py``

## Relation to gates

Stage 6 does not replace Gates 0–5; it provides the scientific evidence matrix
(including confidence-weighting and structure-source ablations) that those
gates summarize for operational decisions.

# Stage 6 — Ablation and scientific reporting

Experiment-matrix orchestration, statistics, and figure generation. Runs
**across** the pipeline using the same Stage-0 splits and seeds.

## Ablation matrix

Configured in ``configs/ablation.yaml``:

| Physics | GP | Confidence weighting | Structure source | Prefilter |
| --- | --- | --- | --- | --- |
| yes | no | — | consensus | no |
| no | yes | — | — | no |
| yes | yes | no | consensus | no |
| yes | yes | yes | consensus | no |
| yes | yes | yes | AF2 | yes |
| yes | yes | yes | AF3 | yes |

Every configuration uses identical splits and ``random_seed``.

Module: ``experiments.py``

```bash
biosensor-stage6
# or
python -m biosensor_priors.stage6_ablation.run --pairwise
```

## Statistics engine

- paired bootstrap (RMSE / MAE deltas + CIs)
- Wilcoxon signed-rank
- Holm adjustment across comparisons
- effect sizes (paired Cohen's d, Cliff's delta)
- confidence intervals

Default: each config vs the reference
(``physics_gp_conf_consensus``). ``--pairwise`` runs the full matrix.

Module: ``statistics.py``

## Reporting

Automatic tables + figures (matplotlib when installed) under
``outputs/stage6/``:

- ``ablation_metrics.csv``
- ``ablation_comparisons.csv``
- ``ablation_statistics.json``
- ``ablation_predictions.parquet``
- ``ablation_report.md``
- ``fig_*.png`` (optional)

Modules: ``figures.py``, ``report.py``

## Relation to gates

Stage 6 does not replace Gates 0–5; it provides the scientific evidence matrix
(including confidence-weighting and structure-source ablations) that those
gates summarize for operational decisions.

Until Stage-1 AF2/AF3 confidence tables exist, those structure-source slots
use a documented deterministic proxy (``structure_available=false``) so the
matrix remains runnable.

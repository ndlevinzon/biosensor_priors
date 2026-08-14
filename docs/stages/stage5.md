# Stage 5 - Prospective wet-lab loop

Data ingestion, frozen-prediction recording, validation, and model lifecycle.
The system is genuinely verified only when it predicts biology it has not seen.

## Components

### 5A. Freeze predictions before experiment

Before synthesis, write an **immutable** file, e.g.
``round_03_predictions.parquet``, containing:

- candidate, predicted fitness, 95% interval
- physics component, GP component, structural confidence
- selection algorithm, selection rank

The 95% interval is $\hat\mu \pm z_{0.975}\hat\sigma$ using the std attached
to the Stage-4 batch. When Stage 4 loaded
``outputs/stage3/uncertainty_calibration.json``, that std is the calibrated
$\sigma_{\mathrm{cal}} = q\,\sigma_{\mathrm{eff}}$.

Hash the file and **never rewrite**. Prevents hindsight leakage.

```bash
python -m biosensor_priors.stage5_prospective.run freeze \
  --round 3 --batch outputs/stage4/batch_design_bo.csv

# Or freeze directly from Stage 4:
python -m biosensor_priors.stage4_search.run --freeze-round 3 --freeze-strategy bo
```

Module: ``freeze_predictions.py``

### 5B. Experimental result importer

New measurements use the **same cleaning pathway** as the historical database
(Stage 0) - not a second code path.

```text
new plate/results -> Stage-0 cleaning -> experiment_master
```

Module: ``import_results.py``

### 5C. Prospective validation

Compare frozen predictions to observations:

- Pearson, Spearman, RMSE, MAE
- ranking precision
- prediction-interval coverage
- fitness improvement rate / best fitness found
- performance by algorithm
- physics re-validation

Module: ``prospective_validation.py``

### 5D. Model-update engine

Only **after** evaluation (Gate 4):

1. append new data
2. refit physics weights, $\alpha$, and version intercept
3. refit GP
4. rerun calibration gates (Stage-3 CV / Gate 3)
5. generate next batch (same ``build_search_policies()`` as Stage 4)

Store physics coefficients by round:

| Round | w_RIF_Ac | w_RIF_Prop | w_delta_RIF | alpha |
| --- | --- | --- | --- | --- |

Physics weights trending toward zero as real data accumulates is a
**legitimate** scientific outcome and should be logged, not suppressed.

```bash
python -m biosensor_priors.stage5_prospective.run ingest \
  --round 3 --results path/to/plate.xlsx --strategy bo
```

Module: ``update_model.py``

## Gate

``gate4.py`` records freeze integrity + prospective validation status before
allowing model update into the next round
(``pipeline.gates.stage5: required_before_model_update``).

## CLI

```bash
biosensor-stage5 freeze --round 3 --batch outputs/stage4/batch_design_bo.csv
biosensor-stage5 ingest --round 3 --results path/to/plate.xlsx
```

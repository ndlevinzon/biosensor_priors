# Stage 3 - Physics-informed GP

Pure Python / statistical ML. Formulation:

$$
\mu_0(x) = w_1\,\mathrm{RIF}_{\mathrm{Ac}}(x) + w_2\,\mathrm{RIF}_{\mathrm{Prop}}(x) + w_3\,\Delta\mathrm{RIF}_{\mathrm{sel}}(x)
$$

$$
F(x) = \alpha\,\mu_0(x) + \alpha_{\mathrm{version}}(x) + f_{\mathrm{residual}}(x),\quad \alpha\in[0,1]
$$

Physics weights use RidgeCV (optional horseshoe / Bayesian ridge / ridge /
OLS). Physics features are **mean only** - they are not extra ARD kernel
dimensions. The residual kernel is a Hamming mutation-set kernel on
``{pos,AA}`` bags plus a small physicochemical Matern.

Optional **multi-output** heads model percentile scores for $S, A, FC, B$
and combine with preregistered weights ($0.40/0.25/0.20/0.15$). Acquisition
can treat affinity / brightness as constraints.

LOCO residuals calibrate $\lambda_s, \lambda_p$ and a CV+ conformal quantile
so Stage 4 uses $\sigma_{\mathrm{cal}} = q\,\sigma_{\mathrm{eff}}$.

Implemented as an **equivalent residual pipeline** that is easier to validate:

```text
fit physics regression on TRAIN -> mu_0
residual = observed fitness - alpha * mu_0 - version intercept
GP fits residual
prediction: hat F(x) = alpha * mu_0(x) + version intercept + GP
```

## Modules

### 3A. Feature builder

Inputs: canonical sequence, physchem, physics scan, structural confidence.

Feature vector (illustrative): mutation-bag bits + physchem, RIF Ac/Prop,
$\Delta$RIF, burial/SASA (later), structural confidence.

**Preprocessing parameters are fitted inside each training split** - no global
standardization before cross-validation. Binary ``mut_`` / ``oh_`` /
physchem-flag columns are not standardized. ``gp_block()`` excludes physics
and structural confidence when ``physics_in_gp: false``. Stage 1/2 priors
are joined before fit (missing confidence = 0, not 1). CV labels use
train-fold percentiles.

Module: ``features.py``

### 3B. Physics mean model

Fit $\mu_0$ on train only (``shrinkage: ridge_cv`` default); expose
coefficients and $\alpha$ for per-round logging.

Module: ``physics_mean.py``

### 3C. Version intercept

Scaffold / version grouped intercept on $y-\alpha\mu_0$ (column
``version``, not ``construct_id``).

Module: ``construct_intercept.py``

### 3D. Structural-confidence weighting

$$
\mathrm{RIF}^* = C_{\mathrm{structure}} \cdot \mathrm{RIF}
$$

(and likewise for other physics components). Preserve **both** raw and
confidence-discounted physics so Stage 6 can ablate weighting.

Module: ``confidence_weighting.py``

### 3E. GP residual learner

Default kernel ``hamming`` (``kernels.py``): Hamming on mutation-set
indicators plus a small physchem Matern. Alternative ``matern52``.

Outputs retain the decomposition:

- ``fitness_mean``, ``fitness_std``
- ``physics_mean``, ``GP_residual_mean``
- optional phenotype means / stds

Modules: ``gp_residual.py``, ``kernels.py``, ``phenotypes.py``

### 3F. Cross-validation and calibration

Primary: **leave-one-construct-out** for physics weights, intercept, and GP
residual. Writes ``outputs/stage3/uncertainty_calibration.json``
($\lambda_s$, $\lambda_p$, conformal $q$).

Sensitivity (extension): version-grouped or lineage-grouped validation, because
V1-V2.4 are not independent random sequences.

Modules: ``cross_validate.py``, ``calibration.py``

### 3G. Gate 3

Compare on **identical Stage-0 splits**:

| Model | Role |
| --- | --- |
| Physics only | Baseline |
| GP zero-mean | Baseline |
| Physics + GP | Candidate fused model |

Metrics: RMSE, MAE, Spearman, Pearson, rank correlation, top-k ranking
accuracy; paired Wilcoxon + Holm; bootstrap CIs.

**Gate 3 passes only if** the fused model shows evidence of improvement over
the prespecified baselines.

Module: ``gate3.py``

## Implementation status

```bash
py -3.12 -m biosensor_priors.stage3_surrogate.run
```

Encodings (``configs/thresholds.yaml`` -> ``gp.encoding``):

* ``mutation_bag`` - mutation physchem deltas + mutation-code one-hots (default with Hamming kernel)
* ``onehot`` - per variable site, classical 20-AA one-hot ($N \times 20$)
* ``georgiev`` - per-site 19-D physicochemical vector (AAIndex-style stand-in)
* ``hybrid`` - onehot + georgiev

Defaults (``gp:``): ``kernel: hamming``, ``encoding: mutation_bag``,
``shrinkage: ridge_cv``, ``physics_in_gp: false``, ``fit_physics_alpha: true``,
``version_intercept: true``, ``multi_output: true``.

Artifacts under ``outputs/stage3/`` and ``manifests/stage3_manifest.json``.

Until Stage 2 physics columns exist, ``physics_only`` is an intercept/mean
baseline; fused = mean + GP residual. Gate records both hard statistical
evidence and an operational soft pass on point RMSE improvement.

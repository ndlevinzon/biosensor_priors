# Stage 3 — Physics-informed GP

Pure Python / statistical ML. Formulation:

$$
\mu_0(x) = w_1\,\mathrm{RIF}_{\mathrm{Ac}}(x) + w_2\,\mathrm{RIF}_{\mathrm{Prop}}(x) + w_3\,\Delta\mathrm{RIF}_{\mathrm{sel}}(x)
$$

$$
F(x) = \alpha\,\mu_0(x) + \alpha_{\mathrm{version}} + f_{\mathrm{residual}}(x),\quad \alpha\in[0,1]
$$

Physics weights use RidgeCV (optional horseshoe). Physics features are **mean
only** — they are not extra ARD kernel dimensions. The residual kernel is a
Hamming mutation-set kernel on ``{pos,AA}`` bags plus a small physicochemical
Matérn.

Optional **multi-output** heads model percentile scores for $S, A, FC, B$
and combine with preregistered weights ($0.40/0.25/0.20/0.15$). Acquisition
can treat affinity / brightness as constraints.

LOCO residuals calibrate $\lambda_s, \lambda_p$ and a CV+ conformal quantile
so Stage 4 uses $\sigma_{\mathrm{cal}} = q\,\sigma_{\mathrm{eff}}$.

Implemented as an **equivalent residual pipeline** that is easier to validate:

```text
fit physics regression on TRAIN → μ₀
residual = observed fitness − μ₀
GP fits residual
prediction: F̂(x) = μ̂₀(x) + f̂_GP(x)
```

## Modules

### 3A. Feature builder

Inputs: canonical sequence, physchem, physics scan, structural confidence.

Feature vector (illustrative): one-hot AA, physchem descriptors, RIF Ac/Prop,
ΔRIF, burial/SASA (later), structural confidence.

**Preprocessing parameters are fitted inside each training split**—no global
standardization before cross-validation.

Module: ``features.py``

### 3B. Physics mean model

Fit μ₀ on train only; expose coefficients for per-round logging.

Module: ``physics_mean.py``

### 3C. Structural-confidence weighting

$$
\mathrm{RIF}^* = C_{\mathrm{structure}} \cdot \mathrm{RIF}
$$

(and likewise for other physics components). Preserve **both** raw and
confidence-discounted physics so Stage 6 can ablate weighting.

Module: ``confidence_weighting.py``

### 3D. GP residual learner

Outputs retain the decomposition:

- ``fitness_mean``, ``fitness_std``
- ``physics_mean``, ``GP_residual_mean``

Module: ``gp_residual.py``

### 3E. Cross-validation engine

Primary: **leave-one-construct-out** for physics weights and GP residual.

Sensitivity (extension): version-grouped or lineage-grouped validation, because
V1–V2.4 are not independent random sequences.

Module: ``cross_validate.py``

### 3F. Gate 3

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

Runnable now (GP-only / physics-optional path):

```bash
py -3.12 -m biosensor_priors.stage3_surrogate.run
```

Encodings (``configs/thresholds.yaml`` → ``gp.encoding``):

* ``mutation_bag`` — mutation physchem deltas + mutation-code one-hots (default with Hamming kernel)
* ``onehot`` — per variable site, classical 20-AA one-hot ($N \\times 20$)
* ``georgiev`` — per-site 19-D physicochemical vector (AAIndex-style stand-in)
* ``hybrid`` — onehot + georgiev

Artifacts under ``outputs/stage3/`` and ``manifests/stage3_manifest.json``.

Until Stage 2 physics columns exist, ``physics_only`` is an intercept/mean
baseline; fused = mean + GP residual. Gate records both hard statistical
evidence and an operational soft pass on point RMSE improvement.

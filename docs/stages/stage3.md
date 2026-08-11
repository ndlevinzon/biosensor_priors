# Stage 3 — Physics-informed GP

Pure Python / statistical ML. Formulation:

$$
\mu_0(x) = w_1\,\mathrm{RIF}_{\mathrm{Ac}}(x) + w_2\,\mathrm{RPX}(x) + w_3\,\Delta\mathrm{RIF}_{\mathrm{sel}}(x)
$$

$$
F(x) = \mu_0(x) + f_{\mathrm{residual}}(x),\quad f_{\mathrm{residual}} \sim \mathrm{GP}(0, k)
$$

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
ΔRIF, RPX, burial/SASA (later), structural confidence.

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

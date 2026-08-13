# Configuration

Analysis knobs live under ``configs/``. Stages must load YAML rather than embed
constants in Python.

| File | Purpose |
| --- | --- |
| ``configs/pipeline.yaml`` | Paths, seeds, round, gate policy, canonical/active versions |
| ``configs/fitness.yaml`` | Preregistered fitness formulation, weights, observation policies, design constraints |
| ``configs/search.yaml`` | Batch size, UCB/AdaLead/MCMC, uncertainty λs, prefilter, diversification |
| ``configs/thresholds.yaml`` | Structure predictors/seeds/states, confidence cutoffs, physics score direction, GP defaults |
| ``configs/physics.yaml`` | Stage 2 backend (mock/external), ligand/Rosetta tool paths, job scheduler, mock control deltas |
| ``configs/ablation.yaml`` | Stage 6 ablation matrix |

## Fitness preregistration

For the first real pipeline, retain **scalar weighted fitness** with weights
frozen before modeling:

$$
F = 0.40\,S + 0.25\,A + 0.20\,\mathrm{FC} + 0.15\,B
$$

with components normalized to $[0,1]$. Changing weights after seeing results is
not allowed within an analysis round; open a new round instead.

Pareto / multi-objective formulations remain future options but are not the
default.

## Observation policies

Explicit policies are required for:

| Observation type | Default policy (first pipeline) |
| --- | --- |
| Exact | Use as-is |
| Left-censored | Interval-aware (do not treat LOD as a point) |
| Right-censored | Interval-aware |
| Missing phenotype | Exclude from fitness (or freeze an impute policy per round) |
| Qualitative | Map to ordinal, then normalize |

## Physics score direction

**Frozen convention** (``configs/thresholds.yaml``):

> More negative physics score = better interaction
> (RF3 docking writes −confidence into ``rif_ac`` / ``rif_prop``).

Downstream code must not guess sign convention. Derived selectivity is:

$$
\Delta\mathrm{RIF}_{\mathrm{sel}} = \mathrm{RIF}_{\mathrm{Ac}} - \mathrm{RIF}_{\mathrm{Prop}}
$$

Raw terms are always retained alongside derived scores.

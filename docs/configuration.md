# Configuration

Analysis knobs live under ``configs/``. Stages must load YAML rather than embed
constants in Python.

| File | Purpose |
| --- | --- |
| ``configs/pipeline.yaml`` | Paths, seeds, round, gate policy, canonical/active versions |
| ``configs/fitness.yaml`` | Preregistered fitness formulation, weights, observation policies, design constraints |
| ``configs/search.yaml`` | Batch size, UCB/AdaLead/MCMC/Thompson, calibrated $\sigma$, prefilter, diversification |
| ``configs/thresholds.yaml`` | Structure predictors/seeds/states, confidence cutoffs, physics score direction, GP defaults |
| ``configs/structures.yaml`` | Stage 1 CHPC SLURM / predictor settings and ipSAE cutoffs |
| ``configs/physics.yaml`` | Stage 2 backend (mock/external), ligand/QM tool paths, job scheduler, mock control deltas |
| ``configs/rf3_physics.yaml`` | Foundry RF3 docking (ipSAE metric keys, templates, GPU jobs) |
| ``configs/ablation.yaml`` | Stage 6 ablation matrix |

## Fitness preregistration

For the first real pipeline, retain **scalar weighted fitness** with weights
frozen before modeling:

$$
F = 0.40\,S + 0.25\,A + 0.20\,\mathrm{FC} + 0.15\,B
$$

with components normalized to $[0,1]$. Changing weights after seeing results is
not allowed within an analysis round; open a new round instead.

Stage 3 can still fit **multi-output phenotype heads** ($S,A,\mathrm{FC},B$)
and combine them with these weights. Affinity / brightness constraints in
Thompson sampling use those heads; they are not a Pareto / multi-objective
search. Pareto formulations remain a future option, not the default.

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
> (RF3 docking writes -ipSAE, or -ipTM fallback, into ``rif_ac`` / ``rif_prop``).

Downstream code must not guess sign convention. Derived selectivity is:

$$
\Delta\mathrm{RIF}_{\mathrm{sel}} = \mathrm{RIF}_{\mathrm{Ac}} - \mathrm{RIF}_{\mathrm{Prop}}
$$

Raw terms are always retained alongside derived scores.

## Source encoding

All documentation, YAML, and Sphinx sources must be **UTF-8** (no BOM) and
**ASCII-only** in the source text:

- Punctuation: hyphen ``-``, ASCII quotes ``"`` ``'``, ``...`` not em-dashes
  or smart quotes.
- Math: MyST dollarmath / LaTeX (``$\mu$``, ``$\sigma$``, ``$\Delta$``), not
  raw Greek or Unicode minus (U+2212).
- ASCII diagrams: ``->``, ``v``, ``|``, ``+--``.

Python identifiers and YAML keys stay ASCII. Ligand SMILES are ASCII. Tests
under ``tests/test_docs_encoding.py`` enforce this for ``docs/`` (including
figures), ``configs/``, ``README.md``, and ``CHANGELOG.md``.

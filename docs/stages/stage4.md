# Stage 4 - Active-learning engine

Pure Python combinatorial / search code. Physics filtering, calibrated
uncertainty, and Random / AdaLead / MCMC / UCB BO / Thompson on the fused
surrogate.

## Components

### 4A. Design-space generator

From parents **V1.0 and V2.4**:

```text
parents x mutable positions x allowed amino acids x indel events x mutation counts
```

Mutable sites include V1-V2.4 diffs and experimental hotspots. Indels
(``insNterm``, ``delNterm``, ``ins104``) occupy a mutation slot. Do **not**
enumerate the unconstrained full sequence space ($M_{\max}=2$ with 20 AA).

Each candidate receives: ``candidate_id``, parent sequence, proposed
mutations, canonical edit bag (scaffold + proposed), ``mutation_cost``,
physchem features, then Stage 1/2 physics and confidence joined from
mutation tables (``sum`` / ``max_abs``). Missing physics is not treated
as a favorable 0; prefilter missing scores as ``PASS``.

Module: ``design_space.py``

### 4B. Physics prefilter

Return **categories**, not silent deletion:

| Category | Intent |
| --- | --- |
| ``PASS`` | Eligible for main acquisition |
| ``SOFT_FAIL`` | Down-weighted / secondary |
| ``HARD_FAIL`` | Usually excluded when confidence is high |
| ``EXPLORATION_RESERVED`` | Budget for physics-misspecification protection |

Bad physics + high confidence -> usually exclude
Bad physics + low confidence -> exploration pool

Module: ``prefilter.py``

### 4C. Search-policy interface

All methods implement:

```python
propose(observed, candidate_pool, surrogate, batch_size) -> batch
```

Plug-ins: Random, AdaLead, MCMC, BO (UCB), Thompson sampling - without changing
downstream consumers. ``build_search_policies()`` is shared by ``run.py``,
``campaign.py``, and Stage 5 ``update_model.py``.

Modules: ``policy.py``, ``random_search.py``, ``adalead.py``, ``mcmc.py``,
``bo.py``, ``thompson.py``, ``bo_evo.py``

### 4D. Uncertainty-aware acquisition

Standard UCB:

$$
\mathrm{UCB}(x) = \mu(x) + \kappa\,\sigma(x)
$$

with **explicit** uncertainty decomposition and Stage-3 conformal scale:

$$
\sigma_{\mathrm{eff}}^2 = \sigma_{\mathrm{GP}}^2 + \lambda_s\,\sigma_{\mathrm{structure}}^2 + \lambda_p\,\sigma_{\mathrm{physics}}^2
$$

$$
\sigma_{\mathrm{cal}} = q\,\sigma_{\mathrm{eff}}
$$

BO uses $\sigma_{\mathrm{cal}}$ when ``outputs/stage3/uncertainty_calibration.json``
exists (``search.yaml`` -> ``uncertainty.use_effective: true``). Structural
uncertainty is not buried inside the GP kernel.

Module: ``acquisition.py``

### 4E. Batch diversification

After ranking, enforce practical campaign constraints:

- maximum candidates per position
- minimum sequence distance
- minimum physicochemical diversity
- mix of high-confidence exploitation and uncertainty exploration

Module: ``batch_design.py``

## Implementation status

Paper-faithful solvers (BO-EVO SI) are implemented:

* **Random** - parent mutation at rate 1/N -> collect M -> sample B
* **AdaLead** - parents with $F \ge (1-\kappa)F_{\max}$, local/recombinant
  children beating the corresponding parent, top-B by $\mu$
* **MCMC** - parallel MH with $\pi \propto \exp(\mu/T)$, collect M, rank by $\mu$, top B
* **BO** - enumerative UCB $\mu + \kappa\sigma$, top B (uses calibrated $\sigma$ when Stage 3 wrote $\lambda, q$)
* **Thompson** - one posterior draw per candidate, top B; optional brightness / FC PropCoA constraints (``search.yaml`` -> ``thompson``)

Primary outputs of ``biosensor-stage4``:

* ``outputs/stage4/proposals_exploit.csv`` - constructs likely to improve $F$
  (brightness / FC PropCoA floors; $\mu - \lambda\,\mathrm{cost}$ must
  compensate for the edit)
* ``outputs/stage4/proposals_explore.csv`` - constructs that reduce
  design-space uncertainty (rank by $\sigma$; no cost filter)

Per-strategy CSVs are still written. Freeze defaults to the exploit batch.

Encodings: ``onehot``, ``georgiev`` (19-D AAIndex-style stand-in), ``hybrid``,
``mutation_bag``.

**Campaigns stay paper-faithful.** ``biosensor-stage4-campaign`` forces
``kind=gp_zero_mean``, scalar fitness, Matern-5/2, and no version intercept.
Operational ``biosensor-stage4`` uses the Stage-3 fused surrogate.

Multi-round paired campaigns:

```bash
py -3.12 -m biosensor_priors.stage4_search.campaign
```

Metrics: success ratio, cumulative best / batch max / batch mean fitness.

# Stage 4 — Active-learning engine

Pure Python combinatorial / search code. Physics filtering, uncertainty
adjustment, and AdaLead / MCMC / BO on the augmented surrogate.

## Components

### 4A. Design-space generator

From active background (e.g. V2.4):

```text
mutable positions × allowed amino acids × allowed mutation counts
```

Examples: all singles; doubles among selected positions; selected triples.
Do **not** enumerate the unconstrained full sequence space.

Each candidate receives: ``candidate_id``, parent sequence, mutations, canonical
positions, physchem features, physics terms, structural confidence.

Module: ``design_space.py``

### 4B. Physics prefilter

Return **categories**, not silent deletion:

| Category | Intent |
| --- | --- |
| ``PASS`` | Eligible for main acquisition |
| ``SOFT_FAIL`` | Down-weighted / secondary |
| ``HARD_FAIL`` | Usually excluded when confidence is high |
| ``EXPLORATION_RESERVED`` | Budget for physics-misspecification protection |

Bad physics + high confidence → usually exclude  
Bad physics + low confidence → exploration pool

Module: ``prefilter.py``

### 4C. Search-policy interface

All methods implement:

```python
propose(observed, candidate_pool, surrogate, batch_size) -> batch
```

Plug-ins: Random, AdaLead, MCMC, BO (UCB), Thompson sampling—without changing
downstream consumers.

Modules: ``policy.py``, ``random_search.py``, ``adalead.py``, ``mcmc.py``,
``bo.py``, ``thompson.py``, ``bo_evo.py``

### 4D. Uncertainty-aware acquisition

Standard UCB:

$$
\mathrm{UCB}(x) = \mu(x) + \kappa\,\sigma(x)
$$

with **explicit** uncertainty decomposition:

$$
\sigma_{\mathrm{eff}}^2 = \sigma_{\mathrm{GP}}^2 + \lambda_s\,\sigma_{\mathrm{structure}}^2 + \lambda_p\,\sigma_{\mathrm{physics}}^2
$$

Acquisition uses $\sigma_{\mathrm{eff}}$ rather than burying structural uncertainty
inside the GP kernel.

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

* **Random** — parent mutation at rate 1/N → collect M → sample B
* **AdaLead** — parents with $F \\ge (1-\\kappa)F_{\\max}$, local/recombinant
  children beating the corresponding parent, top-B by $\\mu$
* **MCMC** — parallel MH with $\\pi \\propto \\exp(\\mu/T)$, collect M, rank by $\\mu$, top B
* **BO** — enumerative UCB $\\mu + \\kappa\\sigma$, top B (uses calibrated $\\sigma$ when Stage 3 wrote $\\lambda, q$)
* **Thompson** — one posterior draw per candidate, top B; optional affinity / brightness constraints (`search.yaml` → `thompson`)

Encodings: ``onehot``, ``georgiev`` (19-D AAIndex-style stand-in), ``hybrid``,
``mutation_bag``.

Multi-round paired campaigns:

```bash
py -3.12 -m biosensor_priors.stage4_search.campaign
```

Metrics: success ratio, cumulative best / batch max / batch mean fitness.

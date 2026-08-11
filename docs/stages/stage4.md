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

Plug-ins: Random, AdaLead, MCMC, BO, (later) BO-EVO—without changing
downstream consumers.

Modules: ``policy.py``, ``random_search.py``, ``adalead.py``, ``mcmc.py``,
``bo.py``, ``bo_evo.py``

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

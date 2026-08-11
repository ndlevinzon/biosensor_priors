# Build order

Do **not** begin with AlphaFold or RIF code.

## First implementation block

1. **Shared infrastructure** — config, identifiers, provenance/manifests, gates, canonical numbering
2. **Stage 0** — experimental loader, fitness transform, frozen splits, validation tests
3. **Stage 3 skeleton** — features + GP-only residual path (physics mean optional / zero)
4. **Stage 4 benchmark** — Random / AdaLead / MCMC / BO on the same interface

There is already enough experimental and sequence data to make these portions
robust.

## Second block (plug-in when ready)

5. **Stage 1** — job generation, predictor adapters, confidence tables
6. **Stage 2** — ligand ensembles, RIF/RPX wrappers, mutation scans, Gate 2 controls

Interfaces are defined so Stage 1/2 outputs drop into Stage 3 without rewriting
search.

## Third block

7. **Stage 5** — freeze predictions, import results, prospective validation, update
   (implemented: ``biosensor-stage5`` / ``python -m biosensor_priors.stage5_prospective.run``)
8. **Stage 6** — ablation matrix, statistics, figures, report

## Capability today vs later

**Runnable now (target):**

```text
experimental data + canonical sequence + physchem
        ↓
     GP-only
        ↓
Random / AdaLead / MCMC / BO
```

**Later addition (same machinery):**

```text
structure confidence + RIF/RPX
        ↓
   physics prior (μ₀)
        ↓
same GP residual + same search policies
```

# Stage 1 — Structural modeling and uncertainty

Primarily an **HPC / workflow orchestration** problem plus structural analysis.
Multiple predictors, seeds, per-residue confidence, PAE, and cross-model RMSD.

Downstream stages never parse raw AF2/AF3/RFAA/ESMFold layouts—they read the
standardized tables this stage emits.

## Layers

### 1A. Structure job generator

Reads ``Version`` + ``Sequence`` and emits jobs, e.g.:

```text
V1.0 / AF2 / seed1 / apo
V1.0 / AF2 / seed2 / apo
V1.0 / AF3 / AcCoA
V2.4 / ...
```

Does **not** predict structures itself. Produces predictor inputs and
shell/scheduler scripts.

Module: ``biosensor_priors.stage1_structures.make_jobs``

### 1B. Predictor-specific adapters

Each predictor has an adapter (``parse_AF2``, ``parse_AF3``, ``parse_RFAA``,
``parse_ESMFold``) converting outputs into one internal schema:

- ``structure_model_id``, ``version``, ``method``, ``seed``, ``state``
  (apo / AcCoA / PropCoA), PDB/mmCIF path
- per residue: ``canonical_position``, AA, pLDDT, PAE summary

Package: ``biosensor_priors.stage1_structures.adapters``

### 1C. Structural comparison engine

- Align models; map residues through canonical numbering
- Superpose Cα atoms; per-position and global RMSD
- Ligand contacts; pocket-specific PAE summaries

Example: consensus at canonical position 324 across AF2 seeds, AF3, RFAA.

Module: ``biosensor_priors.stage1_structures.structural_compare``

### 1D. Structural-confidence calculator

Combines pLDDT and cross-model RMSD (plus pocket PAE) into a confidence score
and reliability flag.

**Output:** ``structural_confidence.parquet`` (illustrative columns):

| Version | Canonical key | pLDDT | RMSD | PAE pocket | Confidence | Reliable |
| --- | --- | --- | --- | --- | --- | --- |
| V2.4 | 324 | 91 | 0.6 | 3.2 | 0.93 | yes |
| V2.4 | 355 | 78 | 1.5 | 6.7 | 0.68 | yes |
| V2.4 | 401 | 56 | 3.1 | 17 | 0.21 | no |

This is the first real **uncertainty channel**. Everything downstream reads
this file.

Module: ``biosensor_priors.stage1_structures.confidence``

## Gate 1

Completeness / quality checks on the structure ensemble and confidence table
(``gate1.py``). Advisory vs required policy is set in ``pipeline.yaml``.

# Stage 0 - Ground truth and success criteria

Wet-lab measurements are the **only** ground truth. This stage is almost
entirely Python data-engineering and validation - not ML.

No modeling starts until Stage-0 tests pass.

## Components

### Experimental database loader

Reads construct-level experimental fields (examples):

- Construct, Version, Parent, mutations
- FC AcCoA, Kd AcCoA, FC PropCoA, Kd PropCoA, Brightness, ...

Combines with:

- ``data/constructs/`` canonical mapping
- ``data/constructs/`` physicochemical annotations

**Authoritative output:** ``experiment_master.parquet``

Module: ``biosensor_priors.stage0_ground_truth.load_experiments``

### Fitness-definition module

```text
experimental measurements
           |
           v
    fitness_transform()
           |
           v
 normalized fitness in [0, 1]
```

Default: preregistered scalar weights (see {doc}`../configuration`).
Policies cover exact, left-/right-censored, missing, and qualitative
observations.

Module: ``biosensor_priors.stage0_ground_truth.fitness``

### Frozen train/test split generator

Writes deterministic files such as:

```text
splits/
    split_001.json
    split_002.json
```

Each file contains:

- training construct IDs
- held-out construct IDs
- random seed
- split strategy

**The same splits are reused** for physics-only, GP-only, and physics+GP so
comparisons are paired.

Module: ``biosensor_priors.stage0_ground_truth.splits``

## Stage-0 tests (required)

Automated tests must verify:

- all constructs uniquely identified
- canonical mappings valid
- fitness reproducible
- no train/test overlap
- no missing required fields
- ``Q324R`` present
- ``A355R`` present

Module: ``biosensor_priors.stage0_ground_truth.validate``

Run:

```bash
py -3.12 -m biosensor_priors.stage0_ground_truth.load_experiments
py -3.12 -m pytest tests -q
```

Artifacts:

- ``data/processed/experiment_master.parquet`` (+ ``.pkl`` with full Python objects)
- ``data/processed/splits/split_XXX.json``
- ``manifests/stage0_manifest.json``

## Primary scientific criterion

Stage 0 exists so Stage 3 / Stage 6 can compare the **fused model** against
**physics-only** and **GP-from-scratch** on held-out experimental measurements
using identical splits.

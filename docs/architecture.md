# Overall architecture

Wet-lab measurements are the only ground truth. Structural models and physics
scores are priors and uncertainty channels that feed a physics-informed GP and
an active-learning loop.

## Pipeline flow

```text
Wet-lab database
      │
      ├──────────────► canonical sequence / physchem database
      │
      ▼
Stage 0: Ground truth + fitness definition
      │
      ▼
Stage 1: Structure ensemble + confidence
      │
      ▼
Stage 2: RIF/RPX physics landscape
      │
      ▼
Stage 3: Physics-informed GP
      │
      ▼
Stage 4: Search / active learning
      │
      ▼
Candidate batch ─────► Wet lab
                         │
                         ▼
                    Stage 5
                  prospective update
                         │
                         └──────────► next round

Across everything:
      Stage 6 = ablation + validation + reporting
```

## Independence contract

| Change… | Must not force re-running… |
| --- | --- |
| GP kernel / residual model | Stage 1 structure prediction |
| BO / AdaLead / MCMC acquisition | Stage 2 RIF/RPX jobs |
| Fitness weights (new *analysis round* only) | Historical manifests (freeze prior rounds) |
| Search batch size | Stages 0–3 artifacts |

Stages consume **files** (parquet / JSON / PDB/mmCIF) plus a **manifest** that
records hashes, parameters, tool versions, seeds, and gate status.

## Code character by stage

| Stage | Main type of code | Computational character |
| --- | --- | --- |
| 0 | Python data engineering + validation | Lightweight |
| 1 | Python orchestration + shell/HPC + structural analysis | Compute-heavy external jobs |
| 2 | Python orchestration + external physics executables + score parsing | Compute-heavy |
| 3 | Python numerical / ML | Moderate |
| 4 | Python search / optimization | Moderate, potentially combinatorial |
| 5 | Python data ingestion / model lifecycle | Lightweight–moderate |
| 6 | Python statistics / plotting | Moderate |

This is intentionally **not** one giant script. Shared infrastructure is built
first; stages plug into stable interfaces.

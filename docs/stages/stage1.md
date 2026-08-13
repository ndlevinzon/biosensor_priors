# Stage 1 — Structural modeling and uncertainty

Primarily an **HPC / workflow orchestration** problem plus structural analysis.
Multiple predictors, seeds, per-residue confidence, PAE, and cross-model RMSD.

Downstream stages never parse raw Boltz2/AF3/ESMFold/RF3 layouts—they read the
standardized tables this stage emits.

## CHPC (University of Utah) predictors

Stage 1 writes SLURM scripts matching CHPC documentation:

| Predictor | Module / tool | Job shape |
| --- | --- | --- |
| **Boltz2** | ``boltz2/2.2.1`` | Single GPU: ``boltz predict`` + CHPC ColabFold MSA server |
| **AF3** | ``alphafold/3.0.0`` | Two-step: ``--norun_inference`` → ``--norun_data_pipeline`` |
| **ESMFold** | ``esmfold/1.0.3`` | Single GPU: fair-esm Python API |
| **RF3** | Foundry ``rf3 fold`` (not a CHPC module yet) | Single GPU; install ``rc-foundry[rf3]`` |

AlphaFold2 and RoseTTAFold2 were **removed** (replaced by Boltz2 and RF3).

**AF3 access:** email ``helpdesk@chpc.utah.edu`` for weight license access.

**RF3 setup (user env):**

```bash
pip install 'rc-foundry[rf3]'
foundry install base-models
# optional: set configs/structures.yaml → rosettafold3.conda_activate
```

```bash
# Generate inputs + SLURM (does not run predictors locally)
biosensor-stage1 --jobs-only --version V2.4

# Boltz2 only
biosensor-stage1 --jobs-only --predictors Boltz2 --seeds 1

# On the cluster:
bash data/structures/jobs/V2.4/submit_all.sh

# After jobs finish:
biosensor-stage1 --ingest-only
```

Defaults (Granite): CPU ``granite`` / GPU ``granite-gpu``, account ``cheatham``,
matching ``--qos``.

## Layers

### 1A. Structure job generator

Emits jobs such as:

```text
V2.4 / Boltz2 / seed1 / apo
V2.4 / AF3 / seed1 / apo
V2.4 / RF3 / seed1 / apo
```

Module: ``biosensor_priors.stage1_structures.make_jobs``

### 1B. Adapters

``parse_Boltz2``, ``parse_AF3``, ``parse_ESMFold``, ``parse_RF3`` → common schema
(``structure_model_id``, pLDDT, …).

### 1C–1D. Comparison + confidence

Same as before → ``structural_confidence.parquet``.

## Gate 1

Advisory vs required in ``pipeline.yaml`` (``gates.stage1``).

## CLI

```text
biosensor-stage1 [--version V2.4] [--jobs-only | --ingest-only]
                 [--submit] [--predictors Boltz2 AF3] [--seeds 1 2 3]
```

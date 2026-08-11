# Stage 1 — Structural modeling and uncertainty

Primarily an **HPC / workflow orchestration** problem plus structural analysis.
Multiple predictors, seeds, per-residue confidence, PAE, and cross-model RMSD.

Downstream stages never parse raw AF2/AF3/RFAA/ESMFold layouts—they read the
standardized tables this stage emits.

## CHPC (University of Utah) AlphaFold

Stage 1 writes **two-step SLURM scripts** matching CHPC documentation:

| Predictor | Module | Step 1 (CPU MSA) | Step 2 (GPU) |
| --- | --- | --- | --- |
| AF2 | ``alphafold/2.3.2`` | ``db_to_tmp_232.sh`` + ``run_alphafold_full.sh … --run_feature=1`` | ``run_alphafold_full.sh`` (no ``--run_feature``) |
| AF3 | ``alphafold/3.0.0`` | ``run_alphafold.sh … --norun_inference`` | ``run_alphafold.sh … --norun_data_pipeline`` on ``*_data.json`` |
| ESMFold | ``esmfold/1.0.3`` | — (single GPU job) | ``esm-fold -i FASTA -o OUT`` |
| RF2 | ``rosettafold2/1.0`` | — (single GPU job) | ``run_RF2.sh FASTA -o OUT`` |

Alias ``RFAA`` / ``RoseTTAFold2`` maps to method ``RF2`` (CHPC RoseTTAFold2 module).

Step 1 AF scripts chain step 2 with ``sbatch -d afterok:$SLURM_JOBID``. Defaults
(partitions, accounts, GRES, memory) live in ``configs/structures.yaml`` —
edit those to match your allocation.

**AF3 access:** CHPC requires emailing ``helpdesk@chpc.utah.edu`` for weight
license access before ``ml alphafold/3.0.0`` will work for your account.

```bash
# Generate FASTA/JSON + SLURM (does not run AF locally)
biosensor-stage1 --jobs-only --version V2.4

# ESMFold only
biosensor-stage1 --jobs-only --predictors ESMFold --seeds 1

# On the cluster, after syncing the repo / data/structures tree:
bash data/structures/jobs/V2.4/submit_all.sh

# After jobs finish, ingest PDBs/CIFs into confidence tables
biosensor-stage1 --ingest-only
```

## Layers

### 1A. Structure job generator

Reads ``Version`` + ``Sequence`` and emits jobs, e.g.:

```text
V2.4 / AF2 / seed1 / apo
V2.4 / AF2 / seed2 / apo
V2.4 / AF3 / seed1 / apo
```

Does **not** predict structures itself. Produces predictor inputs and
SLURM scripts under ``data/structures/jobs/``.

Module: ``biosensor_priors.stage1_structures.make_jobs``

Templates: ``biosensor_priors.stage1_structures.slurm_templates``

### 1B. Predictor-specific adapters

Each predictor has an adapter (``parse_AF2``, ``parse_AF3``, ``parse_ESMFold``,
``parse_RF2``) converting outputs into one internal schema:

- ``structure_model_id``, ``version``, ``method``, ``seed``, ``state``
  (apo / AcCoA / PropCoA), PDB/mmCIF path
- per residue: ``canonical_position``, AA, pLDDT, PAE summary

Package: ``biosensor_priors.stage1_structures.adapters``

### 1C. Structural comparison engine

- Align models; map residues through canonical numbering
- Superpose Cα atoms; per-position and global RMSD
- Ligand contacts; pocket-specific PAE summaries

Module: ``biosensor_priors.stage1_structures.structural_compare``

### 1D. Structural-confidence calculator

Combines pLDDT and cross-model RMSD (plus pocket PAE) into a confidence score
and reliability flag.

**Output:** ``structural_confidence.parquet``

Module: ``biosensor_priors.stage1_structures.confidence``

## Gate 1

Completeness / quality checks on the structure ensemble and confidence table
(``gate1.py``). Advisory vs required policy is set in ``pipeline.yaml``
(``gates.stage1``, default ``advisory``).

## CLI

```text
biosensor-stage1 [--version V2.4] [--sequence ...] [--jobs-only | --ingest-only]
                 [--submit] [--predictors AF2 AF3] [--seeds 1 2 3] [--states apo]
```

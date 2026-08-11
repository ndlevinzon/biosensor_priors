# Stage 2 — Physics landscape

External scientific-computing orchestration plus standardized scoring.
Ligand conformers, separate RIFs for AcCoA and PropCoA, 20-AA scans, RPX
packing, selectivity term, and uncertainty across structural models.

**HPC note:** External RIF/RPX binaries are optional. Default
``configs/physics.yaml`` uses ``backend: mock`` so the Python orchestration,
permanent IDs, long tables, uncertainty aggregation, and Gate 2 all run
locally.

**Ligands on CHPC (no OMEGA):** conformers via built-in **RDKit ETKDG**
(``builtin:rdkit``); QM via **Gaussian16** (``module load gaussian16/SSE4.C01``,
``builtin:gaussian16`` writes ``.gjf`` + SLURM). Install RDKit with
``pip install 'biosensor-priors[chem]'``, set ``backend: external``, then
submit QM with ``bash data/physics/ligands/AcCoA/qm/submit_all.sh``.

## Pieces

### 2A. Ligand conformer pipeline

```text
AcCoA / PropCoA starting structure (or SMILES)
       ↓ conformer generation (RDKit ETKDG — OMEGA replacement)
       ↓ geometry cleanup (MMFF)
       ↓ QM refinement (Gaussian16 Opt on CHPC)
       ↓ deduplication / clustering
       ↓ approved conformer ensemble
```

Each conformer receives a permanent ``conformer_id``.

Outputs: ``data/physics/ligands/AcCoA/``, ``.../PropCoA/``,
``data/physics/ligand_conformers.parquet``

Modules: ``ligand_ensemble.py``, ``conformer_generator.py``, ``gaussian_qm.py``

### 2B. RIF generation wrapper

Programmatic wrapper around the external RIF toolchain:

- construct command, write shell/sbatch scripts
- submit locally (opt-in) or leave for scheduler
- capture stdout/stderr, verify completion, parse scores, store ``job.json``

Inputs: ``structure_model_id`` + ligand conformer ensemble  
Outputs: ``data/physics/rif/.../rif_scores.tsv``

Modules: ``rif_jobs.py``, ``jobs.py``, ``score_parser.py``

### 2C. 20-AA scan engine

For every allowed canonical position × amino acid, generate a mutation
specification and score through RIF/RPX.

Long-format table:

| Version | Position | WT | Mutant | RIF_Ac | RIF_Prop | RPX | delta_RIF_sel |
| --- | --- | --- | --- | --- | --- | --- | --- |

Derived selectivity (definition only — direction is separate):

$$
\Delta\mathrm{RIF}_{\mathrm{sel}} = \mathrm{RIF}_{\mathrm{Ac}} - \mathrm{RIF}_{\mathrm{Prop}}
$$

**Always retain raw terms.** Score direction is frozen in
``thresholds.yaml`` → ``physics.score_direction``
(``more_negative_is_better``).

Module: ``mutation_scan.py``

### 2D. Physics-uncertainty propagation

Multiple structural models ⇒ distributional summaries per mutation:

```text
Q324R
  rif_ac_mean / std / n
  n_structures
  structural_confidence
```

Writes ``data/physics/physics_scores_summary.parquet`` and a Stage-3 drop-in
``data/processed/physics_mutation_scores.parquet``.

Module: ``physics_uncertainty.py``

### 2E. Gate 2 regression tests

Hard controls: **Q324R** and **A355R**.

```text
check_Q324R_direction()  PASS / FAIL
check_A355R_direction()  PASS / FAIL
```

(These are the Gate-2 regression checks; implemented as ``check_*`` so pytest
does not collect them as unit tests from the package namespace.)

If either fails: ``physics_gate = FAIL`` and Stage 3 falls back to
``gp_zero_mean`` (no silent full physics weight).

Module: ``gate2.py``

## CLI

```bash
biosensor-stage2
# or
python -m biosensor_priors.stage2_physics.run --require-gate
```

## Deploying external tools later

1. Install RDKit (``pip install 'biosensor-priors[chem]'``) for conformers;
   Gaussian16 is provided by CHPC (``gaussian16/SSE4.C01``).
2. Confirm ``ligands.tools`` uses ``builtin:rdkit`` / ``builtin:gaussian16``
   (or point at your own wrappers). Set ``ligands.qm.job.account`` / partition.
3. Set ``backend: external``, run Stage 2A (or full ``biosensor-stage2``), then
   ``bash data/physics/ligands/*/qm/submit_all.sh`` for Opt jobs.
4. Point ``rif.executable`` / ``rpx.executable`` when those tools are ready;
   optionally ``jobs.scheduler: slurm`` and ``jobs.submit: true``.
5. Gate 2 must still pass on real scores before Stage 3 trusts physics weights.

# Stage 2 — Physics landscape

Ligand conformers, Rosetta (PyRosetta) interface + packing scores for AcCoA
and PropCoA, 20-AA scans, selectivity term, and uncertainty across structural
models.

**HPC note:** External scoring is optional. Default ``configs/physics.yaml``
uses ``backend: mock`` so the Python orchestration, permanent IDs, long tables,
uncertainty aggregation, and Gate 2 all run locally. On CHPC use
``module load pyrosetta/4.0.0``.

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

### 2B. Rosetta interface + packing wrapper

Programmatic wrapper around **PyRosetta** (CHPC ``pyrosetta/4.0.0``):

- mutate → local pack → total energy (``rpx``)
- optional holo complexes → interface ΔE for AcCoA / PropCoA (``rif_ac`` /
  ``rif_prop``; legacy column names kept for Stage 3)
- write shell/sbatch scripts, capture logs, parse scores, store ``job.json``

Config: ``configs/rosetta_physics.yaml`` (complex PDB paths, pack radius,
score function).

Inputs: ``structure_model_id`` + optional holo complexes  
Outputs: ``data/physics/rif/.../rif_scores.tsv``, ``data/physics/rpx/...``

Modules: ``rif_jobs.py``, ``rpx_jobs.py``, ``wrappers/run_rosetta.py``,
``score_parser.py``

### 2C. 20-AA scan engine

For every allowed canonical position × amino acid, generate a mutation
specification and score through Rosetta interface / packing.

Long-format table:

| Version | Position | WT | Mutant | RIF_Ac | RIF_Prop | RPX | delta_RIF_sel |
| --- | --- | --- | --- | --- | --- | --- | --- |

(Column names are legacy; values are Rosetta energies.)

Derived selectivity:

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

## Deploying PyRosetta on CHPC

1. Install RDKit (``pip install 'biosensor-priors[chem]'``) for conformers;
   Gaussian16 is provided by CHPC (``gaussian16/SSE4.C01``).
2. ``module load pyrosetta/4.0.0`` and confirm ``import pyrosetta``.
3. Set ``configs/rosetta_physics.yaml`` → ``complexes.AcCoA`` / ``PropCoA`` to
   holo PDBs (protein + ligand). Apo-only runs still fill ``rpx``.
4. In ``configs/physics.yaml``: drop ``--scaffold`` from ``rif`` / ``rpx``
   command templates; set ``jobs.module_loads: [pyrosetta/4.0.0]``;
   set ``backend: external``.
5. Gate 2 must still pass on real scores before Stage 3 trusts physics weights.

## Wrapper CLIs

```bash
# Interface + packing (writes rif_scores.tsv; optional --write-rpx)
python -m biosensor_priors.stage2_physics.wrappers.run_rosetta \
  --structure model.pdb --ligands data/physics/ligands \
  --ligand-name 'AcCoA+PropCoA' --out /tmp/rosetta --scaffold

# Packing only → rpx_scores.tsv
python -m biosensor_priors.stage2_physics.wrappers.run_rpx \
  --structure model.pdb --mutation Q324R --out /tmp/rpx --scaffold
```

Entry points: ``biosensor-rosetta``, ``biosensor-rpx`` (``biosensor-rif`` is an
alias of ``biosensor-rosetta``).

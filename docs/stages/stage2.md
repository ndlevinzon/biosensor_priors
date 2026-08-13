# Stage 2 — Physics landscape

Ligand conformers, **RoseTTAFold3 docking** scores for AcCoA and PropCoA,
20-AA scans, selectivity term, and uncertainty across structural models.

**HPC note:** External scoring is optional. Default ``configs/physics.yaml``
uses ``backend: mock`` so the Python orchestration, permanent IDs, long tables,
uncertainty aggregation, and Gate 2 all run locally. On CHPC install Foundry
RF3 (``pip install 'rc-foundry[rf3]'``; same stack as Stage 1).

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

### 2B. RF3 docking wrapper

Programmatic wrapper around **RoseTTAFold3** (Foundry ``rf3 fold``):

- mutate sequence → optional backbone template from Stage-1 structure
- protein + AcCoA / PropCoA docking confidence → ``rif_ac`` / ``rif_prop``
  (negated for the frozen score direction)
- write shell/sbatch scripts, capture logs, parse scores, store ``job.json``

Config: ``configs/rf3_physics.yaml`` (ligand SMILES/SDF, template flags,
metric keys, GPU job defaults).

Inputs: ``structure_model_id`` + ligand SMILES (from ``physics.yaml``) or SDF  
Outputs: ``data/physics/rif/.../rif_scores.tsv``

Modules: ``rif_jobs.py``, ``wrappers/run_rf3_dock.py``, ``score_parser.py``

### 2C. 20-AA scan engine

For every allowed canonical position × amino acid, generate a mutation
specification and score through RF3 ligand docking.

Long-format table:

| Version | Position | WT | Mutant | RIF_Ac | RIF_Prop | delta_RIF_sel |
| --- | --- | --- | --- | --- | --- | --- |

(Values are negated RF3 docking confidences.)

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

If either fails: ``physics_gate = FAIL`` and Stage 3 falls back to
``gp_zero_mean`` (no silent full physics weight).

Module: ``gate2.py``

## CLI

```bash
biosensor-stage2
# or
python -m biosensor_priors.stage2_physics.run --require-gate
```

## Deploying RF3 docking on CHPC

1. Install RDKit (``pip install 'biosensor-priors[chem]'``) for conformers;
   Gaussian16 is provided by CHPC (``gaussian16/SSE4.C01``).
2. Install Foundry RF3: ``pip install 'rc-foundry[rf3]'`` and
   ``foundry install base-models`` (same as Stage 1).
3. Optional: set ``configs/rf3_physics.yaml`` → ``conda_activate`` and/or
   per-ligand SDF ``path`` under ``ligands.*.path``.
4. In ``configs/physics.yaml``: drop ``--scaffold`` from the ``rif``
   command template; set ``backend: external``; keep jobs on
   ``granite-gpu`` with ``gres: gpu:1``.
5. Gate 2 must still pass on real scores before Stage 3 trusts physics weights.

## Wrapper CLI

```bash
python -m biosensor_priors.stage2_physics.wrappers.run_rf3_dock \
  --structure model.pdb --ligands data/physics/ligands \
  --ligand-name 'AcCoA+PropCoA' --out /tmp/rf3 --scaffold
```

Entry point: ``biosensor-rf3-dock`` (``biosensor-rosetta`` is an alias).

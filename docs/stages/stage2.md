# Stage 2 — Physics landscape

External scientific-computing orchestration plus standardized scoring.
Ligand conformers, separate RIFs for AcCoA and PropCoA, 20-AA scans, RPX
packing, selectivity term, and uncertainty across structural models.

## Pieces

### 2A. Ligand conformer pipeline

```text
AcCoA / PropCoA starting structure
       ↓ conformer generation
       ↓ geometry cleanup
       ↓ QM refinement
       ↓ deduplication / clustering
       ↓ approved conformer ensemble
```

Each conformer receives a permanent ``conformer_id``.

Outputs: ``ligands/AcCoA/``, ``ligands/PropCoA/``, ``ligand_conformers.parquet``

Module: ``biosensor_priors.stage2_physics.ligand_ensemble``

### 2B. RIF generation wrapper

Programmatic wrapper around the external RIF toolchain:

- construct command, submit job, capture stdout/stderr
- verify completion, parse scores, store provenance

Inputs: ``structure_model_id`` + ligand conformer ensemble  
Outputs: ``RIF_AcCoA``, ``RIF_PropCoA``

The external program remains responsible for physics.

Modules: ``rif_jobs.py``, ``score_parser.py``

### 2C. 20-AA scan engine

For every allowed canonical position × amino acid, generate a mutation
specification and score through RIF/RPX.

Long-format table (illustrative):

| Version | Position | WT | Mutant | RIF Ac | RIF Prop | RPX |
| --- | --- | --- | --- | --- | --- | --- |

Derived selectivity:

$$
\Delta\mathrm{RIF}_{\mathrm{sel}} = \mathrm{RIF}_{\mathrm{Ac}} - \mathrm{RIF}_{\mathrm{Prop}}
$$

**Always retain raw terms**, not only the derived score. Score **direction
convention is frozen in config** (more negative = better).

Module: ``biosensor_priors.stage2_physics.mutation_scan``

### 2D. Physics-uncertainty propagation

Multiple structural models ⇒ multiple physics scores per mutation. Store
distributional summaries, not a single point:

```text
Q324R
  mean RIF = -12.5
  SD = 1.8
  N structures = 7
  structural confidence = 0.91
```

Module: ``biosensor_priors.stage2_physics.physics_uncertainty``

### 2E. Gate 2 regression tests

Hard controls: **Q324R** and **A355R**.

```text
test_Q324R_direction()  PASS / FAIL
test_A355R_direction()  PASS / FAIL
```

If either fails directionally: ``physics_gate = FAIL`` and Stage 3 must **not**
quietly use physics at full weight.

Module: ``biosensor_priors.stage2_physics.gate2``

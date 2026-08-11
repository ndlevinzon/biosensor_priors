# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Stage 2 physics backend: **removed RifDock / rpxdock scaffolds**; priors now
  come from CHPC **PyRosetta** (``configs/rosetta_physics.yaml``,
  ``wrappers/run_rosetta.py``). Legacy schema columns ``rif_ac`` / ``rif_prop`` /
  ``rpx`` retained for Stage 3. CLI ``biosensor-rosetta`` (``biosensor-rif`` alias).

### Added

- Stage 1 CHPC AlphaFold 2.3.2 / 3.0.0 two-step SLURM job generator
  (``configs/structures.yaml``, FASTA/JSON inputs, ``parse_AF2`` / ``parse_AF3``,
  structural confidence + Gate 1, CLI ``biosensor-stage1``); ESMFold
  (``esmfold/1.0.3``) and RoseTTAFold2 (``rosettafold2/1.0``, method ``RF2``)
  single-GPU job templates + adapters
- Stage 2 physics landscape orchestration: ligand conformer pipeline with
  permanent ``conformer_id``s, **RDKit ETKDG** conformer generator (OMEGA
  replacement), **Gaussian16** (``gaussian16/SSE4.C01``) GJF/SLURM writers,
  RIF/RPX job wrappers + provenance, **PyRosetta CLIs**
  (``biosensor-rosetta`` / ``biosensor-rpx``) for CHPC ``pyrosetta/4.0.0``,
  20-AA scan
  long table with ``ΔRIF_sel = RIF_Ac − RIF_Prop`` (Rosetta energies; legacy
  column names), uncertainty aggregation
  across structure models, Gate 2 Q324R/A355R directional tests, mock backend
  until HPC tools are deployed, CLI ``biosensor-stage2``
- Methodology documentation (``docs/methodology.md``) with equations, full
  stage-by-stage procedure, and architecture-flow figure
- Expanded Read the Docs API reference (``docs/api.md``) covering all
  implemented modules with Napoleon/NumPy autodoc; academic-style README
- Stage 6 ablation matrix (``configs/ablation.yaml``), shared-split runner,
  paired bootstrap / Wilcoxon / Holm / effect-size statistics, automatic
  tables+figures report, CLI ``biosensor-stage6``
- Stage 5 prospective wet-lab loop: immutable hashed prediction freeze (5A),
  Stage-0 cleaning importer (5B), prospective validation metrics (5C),
  model-update engine with physics weights-by-round + Gate 3 recalibration (5D),
  Gate 4 before append/refit, CLI ``biosensor-stage5``
- BO-EVO SI-faithful Stage 4 solvers (Random 1/N→M→B, AdaLead (1-κ) parents,
  MCMC collect-M/rank-μ, enumerative UCB BO) and multi-round campaign runner
  with success ratio / cumulative-best metrics
- Stage 3 encodings: onehot, georgiev, hybrid, mutation_bag
- Stage 3 physics-informed GP: feature builder (train-only preprocessing),
  physics mean, confidence weighting, GP residual, CV over Stage-0 splits,
  Gate 3 (Wilcoxon/Holm/bootstrap), CLI ``biosensor-stage3``
- Stage 4 active learning design space, physics prefilter categories, CLI
  ``biosensor-stage4`` / ``biosensor-stage4-campaign``
- Stage 0 implementation: experimental DB cleaning, canonical alignment,
  physicochemical residue database, preregistered scalar fitness, frozen splits,
  validation gates, and ``experiment_master`` artifacts
- Data inputs under ``data/experimental`` and ``data/constructs``
- CLI entry point ``biosensor-stage0``
- Full pipeline architecture documentation (Stages 0–6, shared infrastructure,
  configuration, identifiers, manifests, build order)
- Sphinx documentation site with MyST Markdown and ``sphinx-rtd-theme``
- Read the Docs config (``.readthedocs.yaml``)
- Package scaffold under ``src/biosensor_priors`` matching the stage layout
- YAML configs: ``pipeline``, ``fitness``, ``search``, ``thresholds``, ``ablation``, ``physics``
- Regression tests for Stage 0/2/3/4/5/6

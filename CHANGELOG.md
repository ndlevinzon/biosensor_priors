# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Dunbrack **ipSAE** (PAE-derived) for cross-model holo interface comparison
  (Stage 1) and as the preferred Stage 2 RF3 dock metric instead of native
  ipTM; fallback to ipTM when PAE is missing.
- Stage 3 multi-output phenotype heads (S, A, FC, B) combined with
  preregistered fitness weights; RidgeCV/horseshoe shrinkage mu_0 with
  alpha in [0, 1]; version intercept; Hamming mutation-set residual kernel;
  LOCO conformal / lambda calibration of sigma_eff.
- Stage-end gate dashboards under ``outputs/gate_reports/stageN/``
  (observations, metrics, statistics, and confidence figures).
- Canonical edit codes for substitutions and insertions/deletions
  (``ins104``, ``insNterm``, ``delNterm``), with a per-edit mutation cost.
- Stage 4 exploit / explore proposal CSVs
  (``proposals_exploit.csv``, ``proposals_explore.csv``).
- ``scripts/clean_pipeline_artifacts.py`` (+ ``.sh`` wrapper) to wipe
  generated Stage 0-6 artifacts for a fresh HPC redeploy while keeping
  experimental inputs, constructs, and configs.
- Stage 1 CHPC AlphaFold 2.3.2 / 3.0.0 two-step SLURM job generator
  (``configs/structures.yaml``, FASTA/JSON inputs, ``parse_AF2`` / ``parse_AF3``,
  structural confidence + Gate 1, CLI ``biosensor-stage1``); ESMFold
  (``esmfold/1.0.3``) and RoseTTAFold2 (``rosettafold2/1.0``, method ``RF2``)
  single-GPU job templates + adapters
- Stage 2 physics landscape orchestration: ligand conformer pipeline with
  permanent ``conformer_id``s, **RDKit ETKDG** conformer generator (OMEGA
  replacement), **Gaussian16** (``gaussian16/SSE4.C01``) GJF/SLURM writers,
  Rosetta job wrappers + provenance, **RF3 docking CLI**
  (``biosensor-rf3-dock``) for Foundry RoseTTAFold3,
  20-AA scan
  long table with ``delta_rif_sel = rif_ac - rif_prop`` (negated RF3
  confidences),
  uncertainty aggregation
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
- BO-EVO SI-faithful Stage 4 solvers (Random 1/N->M->B, AdaLead (1-kappa) parents,
  MCMC collect-M/rank-mu, enumerative UCB BO) and multi-round campaign runner
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
- Full pipeline architecture documentation (Stages 0-6, shared infrastructure,
  configuration, identifiers, manifests, build order)
- Sphinx documentation site with MyST Markdown and ``sphinx-rtd-theme``
- Read the Docs config (``.readthedocs.yaml``)
- Package scaffold under ``src/biosensor_priors`` matching the stage layout
- YAML configs: ``pipeline``, ``fitness``, ``search``, ``thresholds``, ``ablation``, ``physics``
- Regression tests for Stage 0/2/3/4/5/6

### Changed

- Frozen Stage-0 splits default to ``leave_one_construct_out`` (LOCO).
- Stage 3/4 join Stage-1 confidence and Stage-2 physics onto train, pool,
  and design rows (``sum`` or ``max_abs`` for multi-mutants). Missing
  confidence is 0, not 1.0; missing physics is not treated as a favorable 0.
- CV / Gate 3 labels use train-fold phenotype percentiles and fitness minmax
  (``FoldFitnessScaler``); the master ``fitness`` column remains a catalog score.
- ``MISMATCH`` mutation audits (including Pan1.0 Q324R) are excluded from
  fitness labels and are not parsed as mutation bags.
- Binary physchem flags are not z-scored; Georgiev ``_z`` slots load
  continuous AA z-scores.
- Scalar fitness is
  $F = 0.20S + 0.20A + 0.15\mathrm{FC}_{Ac} + 0.25B + 0.20\mathrm{FC}_{Prop}$
  (new analysis round). Brightness is a hard Thompson / exploit floor
  (min 0.55, above the similar cluster). FC PropCoA is in $F$ and a
  Thompson constraint; do not rely on the 7 Kd-ratio labels alone.
- Stage 4 design space uses parents V1.0 and V2.4, version-diff plus
  experimental mutable sites, and indel events. Exploit ranking subtracts
  mutation cost so only compensating edits are suggested.
- Stage 2 physics priors use **RoseTTAFold3 docking** (Foundry ``rf3 fold``)
  (``configs/rf3_physics.yaml``, ``wrappers/run_rf3_dock.py``). Schema columns
  ``rif_ac`` / ``rif_prop`` / ``delta_rif_sel`` (negated RF3 confidence). The
  separate RPX packing path was removed as redundant with RF3. CLI:
  ``biosensor-rf3-dock`` (``biosensor-rosetta`` alias).
- Stage 1 predictors: **Boltz2** (CHPC ``boltz2``) replaces AF2; **RF3**
  (Foundry ``rf3 fold``) replaces RoseTTAFold2. AF3 and ESMFold unchanged.
- Stage 1 SLURM scripts now set ``#SBATCH --output`` / ``--error`` under
  ``data/structures/logs/<version>/`` (one directory per design version).
- Stage 1 / Stage 2 RF3 SLURM headers use ``#SBATCH --ntasks-per-node=1``
  instead of ``-n 8`` so Foundry/Lightning Fabric does not abort under SLURM.
- Stage 1 Boltz2 / ESMFold / AF3 GPU jobs likewise use ``--ntasks-per-node=1``
  (PyTorch Lightning rejects bare ``#SBATCH -n`` when ``SLURM_NTASKS > 1``).
- Stage 1 Boltz inputs follow the CHPC FASTA example; input stems are
  sanitized (no dots); ``--cpus-per-task=16``; later seeds reuse the first
  seed's MSA via SLURM ``afterok`` (avoids ColabFold MMseqs2 overload ERRORs).
- Stage 1 GPU SLURM scripts use ``#SBATCH --export=NONE`` plus ``nvidia-smi``
  checks so an empty login-shell ``CUDA_VISIBLE_DEVICES`` does not hide GPUs
  from PyTorch (ESMFold / Boltz2 / RF3 / AF3 step-2). AF3 step-1 chains step-2
  with ``sbatch --export=NONE`` so the CPU MSA job cannot pass an empty
  ``CUDA_VISIBLE_DEVICES`` into inference.
- Stage 1 RF3 SLURM scripts source ``biosensor_priors`` conda
  (``configs/structures.yaml`` ``rosettafold3.conda_activate``); ``--export=NONE``
  does not inherit a login-shell conda env.
- Boltz2 later seeds reuse a **shared** MSA CSV at
  ``data/structures/msa/<version>/<version>_<state>.csv`` (copied from seed1).
  Boltz names MSA files ``{stem}_{entity}.csv``, not ``A.csv``.
- Documentation (README, Sphinx stages, methodology, architecture) updated to
  the current Stage 1-4 methodology; source docs/YAML are UTF-8 ASCII-only
  with LaTeX math.

# biosensor-priors

[![Documentation Status](https://readthedocs.org/projects/biosensor-priors/badge/?version=latest)](https://biosensor-priors.readthedocs.io/en/latest/?badge=latest)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Physics-informed Gaussian processes for biosensor design and active learning.**

`biosensor-priors` is a research software package for ranking and proposing
protein biosensor variants under limited wet-lab budgets. Experimental
measurements are treated as the only ground truth. Structural models (e.g.
Boltz2 / AF3 / ESMFold / RF3) and physics scores (RoseTTAFold3 docking)
enter as **priors and uncertainty
channels**, not as substitutes for fitness. The pipeline supports
leave-one-construct-out evaluation, BO-EVO-style search policies (Random,
AdaLead, MCMC, enumerative UCB, Thompson), prospective prediction freezes
before synthesis, and a full ablation / statistics reporting layer.

Documentation (methodology, stage contracts, and full API):  
**https://biosensor-priors.readthedocs.io**

---

## Scientific overview

The intended workflow is:

1. **Stage 0** - clean experimental data; preregister scalar fitness; freeze splits
2. **Stage 1** - multi-predictor structure ensembles, confidence, and ipSAE (HPC)
3. **Stage 2** - ligand ensembles, RF3 docking scans, $\Delta\mathrm{RIF}_{\mathrm{sel}}$, Gate 2
4. **Stage 3** - physics mean $\mu_0$ + Hamming GP residual; Gate 3 vs baselines
5. **Stage 4** - constrained design space + Random / AdaLead / MCMC / UCB / Thompson
6. **Stage 5** - immutable prediction freeze -> plate import -> prospective validation -> refit
7. **Stage 6** - ablation matrix on shared splits; bootstrap / Wilcoxon / Holm  

Each stage is independently runnable and communicates through versioned tables
plus `manifest.json` provenance (hashes, parameters, seeds, gate status).

For equations, gates, and the architecture diagram, see
[Methodology](https://biosensor-priors.readthedocs.io/en/latest/methodology.html)
(source: [`docs/methodology.md`](docs/methodology.md)).

---

## Features

- Preregistered fitness $F = 0.20S + 0.20A + 0.15\mathrm{FC}_{Ac} + 0.25B + 0.20\mathrm{FC}_{Prop}$ with explicit censoring policies  
- Canonical numbering across biosensor versions (e.g. V1.0 -> V2.4)
- Physics-informed GP: RidgeCV $\mu_0$, version intercept, Hamming mutation-set residual
- Dunbrack ipSAE for cross-model holo interfaces (preferred RF3 dock metric)
- Paper-faithful search policies plus Thompson sampling and multi-round campaigns  
- Prospective anti-leakage freezes (`round_NN_predictions.parquet` + SHA-256)  
- Ablation statistics (paired bootstrap, Wilcoxon, Holm, effect sizes)  
- Mock Stage-2 backend so orchestration and Gate 2 run before HPC tools are deployed  

---

## Installation

Requires **Python 3.11+**.

```bash
git clone https://github.com/ndlev/biosensor-priors.git
cd biosensor-priors
pip install -e ".[dev,docs]"
```

Optional extras:

| Extra | Purpose |
| --- | --- |
| `dev` | pytest, ruff, matplotlib |
| `docs` | Sphinx + Read the Docs theme |
| `ml` | optional GPyTorch / Torch stack |

---

## Quick start

### Fresh HPC redeploy (wipe generated artifacts)

Keeps ``data/experimental/``, ``data/constructs/``, ``configs/``, and
``weights/`` (RF3 checkpoints). Removes Stage 0-6 outputs under
``data/processed``, ``data/structures``, ``data/physics``, ``data/rounds``,
``outputs/``, and ``manifests/*.json``.

```bash
# On CHPC, from the clone under scratch:
cd /scratch/.../biosensor_priors
python scripts/clean_pipeline_artifacts.py --dry-run   # preview
python scripts/clean_pipeline_artifacts.py --yes        # delete
# optional: also clear __pycache__ / pytest caches
python scripts/clean_pipeline_artifacts.py --yes --caches

pip install -e ".[dev,chem]"
biosensor-stage0
biosensor-stage1 --jobs-only --version V2.4
# submit data/structures/jobs/.../submit_all.sh, then:
# biosensor-stage1 --ingest-only
biosensor-stage2   # mock until RF3 backend: external
biosensor-stage3
```

### Normal stage entry points

```bash
# Ground truth + frozen splits
biosensor-stage0

# CHPC Boltz2/AF3/ESMFold/RF3 SLURM scripts (+ ingest after HPC)
biosensor-stage1 --jobs-only

# Physics landscape (mock backend by default; see configs/physics.yaml)
biosensor-stage2

# Surrogate CV + fused model
biosensor-stage3

# Propose design batches
biosensor-stage4

# Campaign benchmark (Random / AdaLead / MCMC / BO / Thompson)
biosensor-stage4-campaign

# Prospective freeze / ingest
biosensor-stage5 freeze --round 3 --batch outputs/stage4/batch_design_bo.csv
biosensor-stage5 ingest --round 3 --results path/to/plate.xlsx

# Ablations + report
biosensor-stage6
```

Equivalent module entry points:

```bash
python -m biosensor_priors.stage0_ground_truth.load_experiments
python -m biosensor_priors.stage1_structures.run --jobs-only
python -m biosensor_priors.stage2_physics.run
python -m biosensor_priors.stage3_surrogate.run
python -m biosensor_priors.stage4_search.run
python -m biosensor_priors.stage4_search.campaign
python -m biosensor_priors.stage5_prospective.run --help
python -m biosensor_priors.stage6_ablation.run
```

Configuration lives under [`configs/`](configs/) (`pipeline`, `fitness`,
`search`, `thresholds`, `structures`, `physics`, `rf3_physics`, `ablation`).
Do not hard-code analysis constants in Python.

---

## Repository layout

```text
configs/                 YAML analysis contracts (preregistered)
data/                    experimental, constructs, ligands (inputs), structures, physics, rounds
scripts/                 HPC helpers (e.g. clean_pipeline_artifacts)
src/biosensor_priors/
  common/                config, IDs, manifests, gates, ipSAE
  stage0_ground_truth/   cleaning, fitness, splits, Gate 0
  stage1_structures/     CHPC Boltz2/AF3/ESMFold/RF3 jobs, adapters, ipSAE, Gate 1
  stage2_physics/        ligands, RF3 docking scores, scans, Gate 2
  stage3_surrogate/      features, physics mean, Hamming GP, Gate 3
  stage4_search/         design space, policies, Thompson, campaigns
  stage5_prospective/    freeze, import, validate, update
  stage6_ablation/       matrix, statistics, figures, report
tests/                   regression + gate controls (Q324R / A355R)
docs/                    Sphinx / Read the Docs source
manifests/               per-stage provenance
outputs/                 derived reports; gate dashboards in gate_reports/
```

---

## Documentation

| Resource | Location |
| --- | --- |
| Hosted docs | https://biosensor-priors.readthedocs.io |
| Methodology | [`docs/methodology.md`](docs/methodology.md) |
| Architecture | [`docs/architecture.md`](docs/architecture.md) |
| API reference | [`docs/api.md`](docs/api.md) (autodoc from package docstrings) |

Build locally:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

---

## Tests

```bash
pip install -e ".[dev]"
pytest tests -q
ruff check .
```

Gate-oriented checks include Stage-0 controls, Stage-2 directional physics
tests for `Q324R` / `A355R`, Stage-5 freeze immutability, and Stage-6 ablation
smoke tests.

---

## Citation

If you use this software in academic work, please cite the repository and
documentation. A formal paper citation will be added here when available.

```bibtex
@software{biosensor_priors,
  title        = {biosensor-priors: Physics-informed GPs for biosensor design},
  author       = {Nathan D. Levinzon},
  year         = {2026},
  url          = {https://github.com/ndlev/biosensor-priors},
  note         = {Documentation: https://biosensor-priors.readthedocs.io}
}
```

---

## Contributing

Issues and pull requests are welcome. Please:

1. Keep analysis constants in YAML, not in source.
2. Preserve stage independence (file + manifest contracts).
3. Add or update tests for gate-sensitive behavior.
4. Prefer NumPy-style docstrings so the API reference on Read the Docs stays complete.
5. Keep docs, YAML, and Sphinx sources UTF-8 with ASCII-only punctuation;
   write math as LaTeX (``$...$``), not raw Greek, Unicode minus, or em-dashes
   (see [`docs/configuration.md`](docs/configuration.md)).  

---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgments

Pipeline design follows a physics-informed GP + active-learning framing for
AcCoA-selective biosensor engineering, with search policies aligned to the
BO-EVO style interface (Random, AdaLead, MCMC, enumerative UCB, Thompson). External
structure and RF3 (Foundry) remain user-deployed; the Python layer
provides orchestration, provenance, and gates.

This project was inspired by the following papers:
 - https://doi.org/10.1093/bib/bbac570
 - https://doi.org/10.64898/2026.07.13.738243

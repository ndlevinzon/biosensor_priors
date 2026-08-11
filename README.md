# biosensor-priors

Physics-informed Gaussian process pipeline for biosensor design: wet-lab ground
truth → structural confidence → RIF/RPX physics → GP residual → active learning
→ prospective wet-lab rounds → ablations.

Each stage is independently runnable. Artifacts are versioned tables plus
``manifest.json`` provenance—not a single monolithic script.

## Documentation

- **Hosted (Read the Docs):** https://biosensor-priors.readthedocs.io  
  (activate the project on [readthedocs.org](https://readthedocs.org/) and point
  it at this repository; build config is ``.readthedocs.yaml``)
- **Local:**

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

Architecture, stage contracts, configuration, identifiers, and build order are
documented under ``docs/``.

## Installation

```bash
pip install -e ".[dev]"
# optional ML / docs extras:
pip install -e ".[ml,docs]"
```

Requires Python 3.11+.

## Repository layout

```text
configs/          YAML: pipeline, fitness, search, thresholds
data/             experimental, constructs, structures, physics, rounds
src/biosensor_priors/
  common/         config, IDs, manifests, gates, canonical maps
  stage0_…6/      independently runnable stages
tests/            numbering, controls Q324R/A355R, leakage, reproducibility
manifests/        per-stage provenance
outputs/          reports and derived artifacts
docs/             Sphinx source (Read the Docs)
```

## Build order (implementation)

1. Shared infrastructure + Stage 0  
2. Stage 3 skeleton (GP-only path) + Stage 4 search benchmark  
3. Plug in Stage 1 (structures) and Stage 2 (physics) when available  
4. Stage 5 prospective loop + Stage 6 ablations  

See [docs/build_order.md](docs/build_order.md).

## Development

```bash
py -3.12 -m pip install -e ".[dev,docs]"
py -3.12 -m biosensor_priors.stage0_ground_truth.load_experiments
py -3.12 -m biosensor_priors.stage3_surrogate.run
py -3.12 -m biosensor_priors.stage4_search.run
py -3.12 -m biosensor_priors.stage4_search.campaign
py -3.12 -m pytest tests -q
ruff check .
```

## License

MIT

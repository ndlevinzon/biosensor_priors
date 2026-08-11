# biosensor-priors documentation

Physics-informed Gaussian process pipeline for biosensor design, structural
uncertainty, RIF/RPX physics landscapes, and active learning.

```{toctree}
:maxdepth: 2
:caption: Overview

architecture
build_order
project_structure
```

```{toctree}
:maxdepth: 2
:caption: Shared infrastructure

infrastructure
configuration
identifiers
manifests
```

```{toctree}
:maxdepth: 2
:caption: Pipeline stages

stages/stage0
stages/stage1
stages/stage2
stages/stage3
stages/stage4
stages/stage5
stages/stage6
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
changelog
```

## Quick start

```bash
pip install -e ".[docs,dev]"
cd docs && sphinx-build -b html . _build/html
```

Hosted docs are built by [Read the Docs](https://readthedocs.org/) from
``.readthedocs.yaml``.

## Design principle

Each stage is **independently runnable**. Changing the GP kernel must not force
a re-run of AlphaFold; changing the BO acquisition function must not force a
re-run of RIF. Stages communicate through versioned tables, manifests, and
stable identifiers—not through shared in-memory state.

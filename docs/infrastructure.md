# Shared infrastructure

Built **before** stage implementations. Every later stage depends on these
contracts.

## Configuration layer

Human-readable YAML (see {doc}`configuration`) rather than hard-coded Python
constants. Typical contents:

- canonical reference (e.g. ``V1.0``)
- active design background (e.g. ``V2.4``)
- fitness weights
- allowed mutable positions and amino acids
- maximum mutations per construct
- structure confidence and ipSAE thresholds
- RF3 physics thresholds and score-direction convention
- GP settings (kernel, shrinkage, multi-output)
- UCB $\kappa$, AdaLead $\varepsilon$, MCMC temperature, Thompson constraints
- random seed and analysis round

Load via ``biosensor_priors.common.config``.

## Identifier system

Every object carries stable IDs (see {doc}`identifiers`). This prevents
ambiguous joins between wet-lab rows, structures, physics scans, model runs,
and candidates.

## Provenance / manifests

Each stage writes ``manifest.json`` (see {doc}`manifests`) recording inputs,
hashes, parameters, software versions, seeds, outputs, and gate pass/fail.

## Gates

``biosensor_priors.common.gates`` evaluates and records gate status. Failed
required gates must block silent use of that stage's outputs (e.g. Gate 2 fail
-> Stage 3 must not use physics at full weight without recording the failure).

At the end of each stage, ``biosensor_priors.common.gate_reports`` writes a
visual report under ``outputs/gate_reports/stageN/``: overview figure,
``index.md``, ``gate.json``, and ``stats.json`` (observations, metrics,
statistics, and confidence for that stage).

## Canonical numbering

``biosensor_priors.common.canonical`` maps version positions <-> canonical
positions so structural and physics features align across backgrounds.

## Encoding

Documentation, YAML, and Sphinx sources are UTF-8 with **ASCII-only**
punctuation and identifiers. Math uses MyST / LaTeX (``$...$``, ``$$...$$``)
rather than raw Greek letters, Unicode minus, em-dashes, or smart quotes.
See {doc}`configuration`.

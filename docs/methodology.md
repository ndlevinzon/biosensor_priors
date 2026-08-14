# Methodology

## Summary

**biosensor-priors** is a modular pipeline for designing AcCoA-selective
biosensor variants with a physics-informed Gaussian process (GP) and an
active-learning loop. Wet-lab measurements are the only ground truth.
Structural predictors (Boltz2, AF3, ESMFold, RF3) and RoseTTAFold3 docking
scores enter only as **priors and uncertainty channels**. The code never
treats a physics score or a predicted structure as a substitute for
experimental fitness.

The operational path is: clean and freeze experimental data and fitness
(Stage 0); assemble multi-model structures, confidence, and Dunbrack ipSAE
(Stage 1); generate ligand conformers and RF3 docking scans with selectivity
$\Delta\mathrm{RIF}_{\mathrm{sel}}$ (Stage 2); fit a decomposed surrogate
$\hat F(x)=\alpha\hat\mu_0(x)+\alpha_{\mathrm{version}}(x)+\hat f_{\mathrm{GP}}(x)$
under cross-validation (Stage 3); propose batches with Random / AdaLead /
MCMC / UCB BO / Thompson over a constrained design space (Stage 4); freeze
predictions before synthesis, ingest new plates through the same cleaning
path, validate prospectively, and refit (Stage 5). Stage 6 runs an ablation
matrix on shared splits and seeds and reports paired statistics.

Stages communicate only through versioned tables, manifests, hashes, and
stable identifiers. Gates (0-5) block silent misuse of failed artifacts - for
example Gate 2 failure forbids full physics weight in Stage 3. Score-direction
conventions, fitness weights, and analysis seeds are preregistered in YAML so
downstream code does not guess signs or redefine objectives after seeing
results.

```{figure} _static/figures/architecture_flow.svg
:alt: Architecture flow from Stage 0 ground truth through Stage 5 prospective update, with Stage 6 ablations spanning the pipeline
:width: 100%

Architecture flow. Solid arrows are the primary data path; dashed arrows mark
prospective feedback and cross-cutting Stage-6 evaluation.
```

---

## Complete methodological details

### 1. Design principles

1. **Independence.** Changing the GP kernel must not force re-running
   Boltz2 / AF3 / RF3; changing UCB $\kappa$ or Thompson constraints must not
   force re-running RF3 docking.
2. **File contracts.** Stages exchange parquet/JSON/PDB (or mmCIF) plus a
   `manifest.json` (inputs, parameters, software versions, seed, gate).
3. **Preregistration.** Fitness weights, observation policies, score
   direction, seeds, and mutable positions live in `configs/*.yaml`.
4. **Gates.** Failed required gates are recorded and enforced; they are not
   soft warnings that can be ignored downstream.

Identifier vocabulary (non-exhaustive): `construct_id`, `version`,
`canonical_position`, `structure_model_id`, `conformer_id`,
`physics_scan_id`, `candidate_id`, `split_id`, `experimental_round`.

---

### 2. Stage 0 - Ground truth and fitness

#### 2.1 Cleaning and alignment

Raw workbook rows are normalized (affinity units, fold-change, brightness
ordinals, mutation parsing from Construct/Description). Constructs are
resolved onto version backgrounds and aligned to a **canonical numbering**
(reference version, typically `V1.0`) so position `324` means the same site
across V1-V2.4. Physicochemical residue tables are attached per site.

#### 2.2 Scalar fitness (preregistered)

Phenotype components are oriented so **higher is better**, then mapped to
$[0,1]$ (percentile / ordinal normalization). Default weights:

$$
F = 0.40\,S + 0.25\,A + 0.20\,\mathrm{FC} + 0.15\,B
$$

where $S$ is selectivity, $A$ affinity, $\mathrm{FC}$ fold-change, and $B$
brightness. Missing phenotypes are **not imputed**. Weights of missing
components are redistributed over available components (policy
`missing_phenotype: redistribute_weights`), subject to a minimum number of
components (`min_components`, default 2).

The Stage-0 `fitness` column is a **catalog** score: global percentiles over
trusted identities. Rows with `mutation_audit = MISMATCH` are omitted from
the rank reference and receive no fitness (including Pan1.0 Q324R, previously
the global-best label). Stage 3/4 cross-validation refits percentiles and
minmax on the **training fold only**, then scores the held-out fold against
those train ranks (`FoldFitnessScaler`).

FC PropCoA is stored as an off-target auxiliary head and is **not** in $F$.

Censoring policies (frozen in `fitness.yaml`) include, among others:

- Affinity: exact used as $-\log_{10}(K_d/\mathrm{\mu M})$; left-censored
  `<x` uses $x$ conservatively; right-censored `>x` omitted.
- Selectivity: positive lower bound of $K_d(\mathrm{Prop})/K_d(\mathrm{Ac})$;
  non-informative bounds omitted.

#### 2.3 Splits and Gate 0

Frozen train/held-out splits (`split_*.json`) are written once and reused by
Stages 3 and 6. The default strategy is **leave-one-construct-out**
(`configs/pipeline.yaml` -> `splits.strategy`). Gate 0 checks unique IDs,
required control mutations (`Q324R`, `A355R`), fitness reproducibility, and
train/test non-overlap.

**Primary artifact:** `data/processed/experiment_master.{pkl,parquet}`.

---

### 3. Stage 1 - Structural ensemble and confidence

Stage 1 is HPC orchestration plus structural analysis. Predictors on CHPC
(University of Utah) Granite:

| Method | Role |
| --- | --- |
| Boltz2 | Primary folding / holo complex predictor (replaces AF2) |
| AF3 | Independent complex predictor |
| ESMFold | Fast monomer / apo check (typically no ligand PAE) |
| RF3 | Foundry `rf3 fold` (replaces RoseTTAFold2) |

For each `(version, method, seed, state)` the pipeline produces a
`structure_model_id` and per-residue confidence features (pLDDT, pocket PAE,
cross-model RMSD). These are combined into a scalar
**structural confidence** $C_{\mathrm{structure}}\in[0,1]$ and a reliability
flag, written to `structural_confidence.parquet`.

**ipSAE for cross-model interfaces.** Native ipTM is not comparable across
predictors. Holo AF3 / Boltz2 / RF3 jobs with PAE are scored with Dunbrack
ipSAE (PAE cutoff 10 A, $d_0$ from PAE-filtered residues; protein-ligand
treats ligand tokens as the partner chain). Apo jobs are skipped. Tables:

- `data/structures/ipsae_by_model.parquet`
- `data/structures/ipsae_across_models.parquet` (mean / std / range)

ipSAE std is the disagreement measure across predictors. ESMFold is omitted
when ligand PAE is absent.

Thresholds (from `thresholds.yaml` / `structures.yaml`):

- pLDDT minimum for reliability
- maximum C$\alpha$ RMSD across models
- maximum pocket PAE
- ipSAE PAE / distance cutoffs

Downstream stages **never** parse raw Boltz2/AF3/ESMFold/RF3 directory
layouts; they consume the standardized confidence and ipSAE tables.

---

### 4. Stage 2 - Physics landscape

Score direction is frozen:

$$
\text{more negative physics score} \equiv \text{better interaction}
\quad(\text{RF3: }-\text{ipSAE, fallback }-\text{ipTM}).
$$

Parsers and gates **do not infer** this convention. Schema column names
`rif_ac` / `rif_prop` / `delta_rif_sel` are retained for compatibility.

#### 4.1 Ligand conformer pipeline (2A)

For AcCoA and PropCoA. CHPC has no OpenEye OMEGA: conformers use **RDKit
ETKDG** (`builtin:rdkit`); QM uses **Gaussian16**
(`module load gaussian16/SSE4.C01`) via written `.gjf` + SLURM scripts.

```text
starting structure -> conformer generation -> geometry cleanup
  -> QM refinement -> deduplication / clustering -> approved ensemble
```

Each approved structure receives a permanent `conformer_id` derived from
content hash + schema version. Catalog:
`data/physics/ligand_conformers.parquet`.

#### 4.2 RoseTTAFold3 docking wrappers (2B)

Stage-2 priors are **RoseTTAFold3** (Foundry `rf3 fold`) ligand docking
confidences. Interface score prefers Dunbrack **ipSAE** from PAE (same
formula as Stage 1), falling back to native ipTM when PAE is missing, then
negated into `rif_ac` and `rif_prop` so `more_negative_is_better` holds.

For each `structure_model_id` and ligand ensemble the Python layer:

1. constructs the external command from templates in `physics.yaml` /
   `rf3_physics.yaml`
2. writes shell / optional Slurm scripts
3. captures stdout/stderr and `job.json` provenance
4. verifies completion and parses score tables

Until HPC binaries are configured, `backend: mock` exercises the same path
with deterministic pseudo-scores.

#### 4.3 Twenty-amino-acid scan (2C)

For every allowed canonical position $p$ and amino acid $a\in\mathcal{A}$:

$$
\text{mutation} = \mathrm{WT}(p)\,p\,a
$$

Long-format scores retain **raw** terms and the derived selectivity

$$
\Delta\mathrm{RIF}_{\mathrm{sel}}
  = \mathrm{RIF}_{\mathrm{Ac}} - \mathrm{RIF}_{\mathrm{Prop}}.
$$

#### 4.4 Uncertainty across models (2D)

If mutation $m$ is scored on $N$ structures:

$$
\overline{\mathrm{RIF}}(m)=\frac{1}{N}\sum_{i=1}^{N}\mathrm{RIF}_i(m),\qquad
\mathrm{SD}(m)=\sqrt{\frac{1}{N-1}\sum_{i=1}^{N}\bigl(\mathrm{RIF}_i-\overline{\mathrm{RIF}}\bigr)^2}
$$

(and likewise for $\Delta\mathrm{RIF}_{\mathrm{sel}}$), joined to
$C_{\mathrm{structure}}$ when available.

#### 4.5 Gate 2 (2E)

Controls `Q324R` and `A355R` must show $\Delta\mathrm{RIF}_{\mathrm{sel}}$ with
the expected sign for `favorable_AcCoA` under the frozen score direction
(for `more_negative_is_better`, expected $\Delta\mathrm{RIF}_{\mathrm{sel}}<0$).
If either fails: `physics_gate = FAIL` and Stage 3 must not use physics at
full weight (falls back to `gp_zero_mean`).

---

### 5. Stage 3 - Physics-informed GP

#### 5.1 Feature construction

For construct $x$, features may include:

- sequence encodings: mutation-bag (default), one-hot, Georgiev (19-D
  physchem), or hybrid
- physics block: $\mathrm{RIF}_{\mathrm{Ac}}$, $\mathrm{RIF}_{\mathrm{Prop}}$,
  $\Delta\mathrm{RIF}_{\mathrm{sel}}$
- structural confidence $C_{\mathrm{structure}}$

Standardization statistics are fit **inside each training split only**.
Binary mutation / one-hot / physchem-flag columns are not standardized.
Georgiev $z$ slots are continuous AA properties (not z-scored binary flags).

Physics and Stage-1 confidence are joined onto train, pool, and design rows
from Stage 2 mutation tables (`sum` or `max_abs` for multi-mutants;
`configs/thresholds.yaml` -> `priors.multi_mutant`). Missing physics stays
missing and is mean-imputed from **train** only; it is not filled with 0
before that (so `more_negative_is_better` cannot treat "unknown" as a
favorable score). Missing structural confidence is 0, not 1.

#### 5.2 Confidence weighting of physics

Raw physics features $z$ are retained, and a weighted copy is formed:

$$
z^\* = C_{\mathrm{structure}}\cdot z
$$

for each physics column. Ablations (Stage 6) compare weighted vs unweighted.

#### 5.3 Decomposed surrogate

Physics mean (train-only RidgeCV by default; optional horseshoe / Bayesian
ridge / ridge / OLS; intercept if physics absent):

$$
\mu_0(x)
  = w_{\mathrm{RIF\,Ac}}\,\mathrm{RIF}_{\mathrm{Ac}}(x)
  + w_{\mathrm{RIF\,Prop}}\,\mathrm{RIF}_{\mathrm{Prop}}(x)
  + w_{\Delta}\,\Delta\mathrm{RIF}_{\mathrm{sel}}(x)
  + b
$$

A shrinkage weight $\alpha\in[0,1]$ is fit on train so physics cannot
dominate labeled data:

$$
\alpha = \mathrm{clip}_{[0,1]}\!\left(
  \frac{\mu_0^\top y}{\mu_0^\top\mu_0}
\right).
$$

A **version / scaffold intercept** $\alpha_{\mathrm{version}}$ is fit on the
residual $y-\alpha\mu_0$ (grouped by `version`, not `construct_id`). The GP
then fits the leftover residual. Physics features are **mean only** - they
are not extra ARD kernel dimensions (`physics_in_gp: false`).

Default residual kernel: Hamming on mutation-set indicator bits
$|S\triangle T|$ plus a small physicochemical Matern-5/2. Alternative:
isotropic Matern-5/2 (`kernel: matern52`).

Fused prediction:

$$
\hat F(x)
  = \alpha\,\hat\mu_0(x)
  + \alpha_{\mathrm{version}}(x)
  + \hat f_{\mathrm{GP}}(x).
$$

**Multi-output heads** (default): percentile scores for $S,A,\mathrm{FC},B$
are modeled with the same stack, then combined with preregistered fitness
weights (missing-weight redistribution). Acquisition can treat affinity /
brightness as constraints rather than folding everything into a single
scalar.

Operational residual pipeline (train only):

```text
fit physics mean on TRAIN
alpha shrink; subtract alpha * mu_0
subtract version intercept
GP fits residual (Hamming + physchem Matern)
predict: alpha * mu_0 + version intercept + GP
```

Model kinds for CV / ablations:

| Kind | Prediction |
| --- | --- |
| `physics_only` | $\alpha\hat\mu_0$ (+ optional intercept) |
| `gp_zero_mean` | $\hat f_{\mathrm{GP}}$ on $F$ with $\mu_0=0$ |
| `physics_gp` | $\alpha\hat\mu_0+\alpha_{\mathrm{version}}+\hat f_{\mathrm{GP}}$ |

#### 5.4 Uncertainty calibration

LOCO residuals fit $\lambda_s$, $\lambda_p$ and a CV+ conformal quantile
$q$ written to `outputs/stage3/uncertainty_calibration.json`:

$$
\sigma_{\mathrm{eff}}^2
  = \sigma_{\mathrm{GP}}^2
  + \lambda_s\,\sigma_{\mathrm{structure}}^2
  + \lambda_p\,\sigma_{\mathrm{physics}}^2,
\qquad
\sigma_{\mathrm{cal}} = q\,\sigma_{\mathrm{eff}}.
$$

Stage 4 acquisition uses $\sigma_{\mathrm{cal}}$ when that file is present.

#### 5.5 Gate 3

On identical Stage-0 splits, compare fused vs baselines with RMSE, MAE,
Pearson, Spearman, top-$k$ ranking; paired Wilcoxon on absolute errors with
Holm adjustment; bootstrap CIs on $\Delta\mathrm{RMSE}$. Gate 3 requires
evidence that the fused model improves on both baselines.

---

### 6. Stage 4 - Active learning / search

#### 6.1 Design space

From the active background (e.g. V2.4), enumerate constrained mutants:

$$
\{\text{mutable positions}\} \times \{\text{allowed AAs}\} \times \{1,\ldots,M_{\max}\}
$$

Positions use canonical numbering mapped to version-local indices. Each
candidate gets `candidate_id`, mutations, physics placeholders / scores, and
confidence.

#### 6.2 Physics prefilter categories

Candidates are labeled, not silently deleted:

| Category | Role |
| --- | --- |
| `PASS` | Main acquisition pool |
| `SOFT_FAIL` | Secondary / down-weighted |
| `HARD_FAIL` | Typically excluded when confidence is high |
| `EXPLORATION_RESERVED` | Budget when physics is bad but confidence is low |

"Goodness" of a physics score respects `score_direction` (e.g. negate scores
when more-negative-is-better).

#### 6.3 Acquisition and policies

Standard UCB (with calibrated $\sigma$ when available):

$$
\mathrm{UCB}(x)=\mu(x)+\kappa\,\sigma_{\mathrm{cal}}(x).
$$

Paper-faithful solvers share `propose(observed, pool, surrogate, B)`:

- **Random:** mutate $1/N$ sites -> collect $M$ -> sample batch $B$.
- **AdaLead:** parents with $F\ge(1-\kappa)F_{\max}$; local / recombinant
  proposals; top-$B$ by $\mu$.
- **MCMC:** target $\pi\propto e^{\mu/T}$ (maximization form); collect $M$;
  rank by $\mu$; take top $B$.
- **BO:** enumerative UCB over the pool; top $B$.
- **Thompson:** one posterior draw per candidate; top $B$. Optional
  affinity / brightness constraints (`search.yaml` -> `thompson`); primary
  head defaults to selectivity.

**Campaigns stay paper-faithful.** `biosensor-stage4-campaign` forces
`kind=gp_zero_mean`, scalar fitness, Matern-5/2, and no version intercept
(BO-EVO SI) and does not require the physics join. Operational
`biosensor-stage4` uses the Stage-3 fused surrogate (Hamming, multi-output,
calibrated $\sigma$) after joining physics/confidence onto observed and
design rows. Missing physics prefilters as PASS.

Campaign metrics include success ratio and cumulative best fitness across
rounds/repeats.

---

### 7. Stage 5 - Prospective wet-lab loop

#### 7.1 Freeze (5A)

Before synthesis, write immutable

`round_{NN}_predictions.parquet`

with predicted fitness, 95% interval $(\hat\mu\pm z_{0.975}\hat\sigma)$
using the std attached to the Stage-4 batch (calibrated $\sigma_{\mathrm{cal}}$
when Stage 4 wrote it), physics/GP components, structural confidence,
selection algorithm, and rank. Hash the file (`*.sha256`) and refuse
rewrites (anti-hindsight leakage).

#### 7.2 Import (5B)

New plates pass through the **same** Stage-0 cleaning and fitness transform,
then append to `experiment_master` (no second pathway).

#### 7.3 Prospective validation (5C)

Join frozen predictions to observations and compute:

- Pearson / Spearman correlation
- RMSE / MAE
- ranking precision@$k$
- 95% interval coverage
- fitness improvement vs prior best; best fitness found
- metrics by selection algorithm
- physics-component vs observation correlation (re-check)

#### 7.4 Model update (5D) and Gate 4

Only after Gate 4 (freeze integrity + matched observations + finite metrics):

1. append new data
2. refit physics weights, $\alpha$, version intercept, and GP
3. re-run Stage-3 calibration gates
4. propose the next batch

Physics coefficients are logged by round:

| Round | $w_{\mathrm{RIF\,Ac}}$ | $w_{\mathrm{RIF\,Prop}}$ | $w_{\Delta\mathrm{RIF}}$ | $\alpha$ |

Weights trending toward zero as labeled data accumulate is a legitimate
scientific outcome and is recorded, not suppressed.

---

### 8. Stage 6 - Ablation and reporting

Ablation cells vary: physics on/off, GP on/off, confidence weighting,
structure source (consensus / Boltz2 / AF3), and prefilter on/off. Every cell
uses the **same** Stage-0 splits and random seed.

Statistics engine (default: each config vs a reference fused model):

- paired bootstrap CIs for $\Delta\mathrm{RMSE}$ / $\Delta\mathrm{MAE}$
- Wilcoxon signed-rank on absolute errors
- Holm adjustment across comparisons
- effect sizes (paired Cohen's $d$, Cliff's $\delta$)

Reporting writes metrics/comparison tables, optional figures, and
`ablation_report.md` under `outputs/stage6/`.

Stage 6 does **not** replace Gates 0-5; it supplies the scientific evidence
matrix those gates summarize.

---

### 9. End-to-end computational recipe

```text
1. Stage 0  -> experiment_master + frozen splits + Gate 0
2. Stage 1  -> structure_model_id ensemble + structural_confidence + ipSAE
3. Stage 2  -> ligand_conformers + RF3 mutation scan + physics summary + Gate 2
4. Stage 3  -> CV predictions + fused surrogate + uncertainty_calibration + Gate 3
5. Stage 4  -> design space -> prefilter -> propose batches (optional freeze)
6. Wet lab  -> synthesize / measure selected batch
7. Stage 5  -> validate vs freeze -> Gate 4 -> append/refit -> next batch
8. Stage 6  -> ablation matrix on shared splits -> statistics -> report
```

Runnable entry points (Python >=3.11):

```bash
biosensor-stage0
biosensor-stage1 --jobs-only   # then ingest after HPC
biosensor-stage2               # mock backend until HPC tools are set
biosensor-stage3
biosensor-stage4
biosensor-stage4-campaign
biosensor-stage5 freeze|ingest ...
biosensor-stage6
```

Configuration sources of truth: `configs/pipeline.yaml`, `fitness.yaml`,
`search.yaml`, `thresholds.yaml`, `structures.yaml`, `physics.yaml`,
`rf3_physics.yaml`, `ablation.yaml`.

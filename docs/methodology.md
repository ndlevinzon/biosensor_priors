# Methodology

## Summary

**biosensor-priors** is a modular pipeline for designing AcCoA-selective
biosensor variants with a physics-informed Gaussian process (GP) and an
active-learning loop. Wet-lab measurements are the only ground truth.
Structural predictors (AF2/AF3/RFAA/…) and physics scores (RIF/RPX) enter only
as **priors and uncertainty channels**. The code never treats a physics score
or a predicted structure as a substitute for experimental fitness.

The operational path is: clean and freeze experimental data and fitness
(Stage 0); assemble multi-model structures and confidence (Stage 1); generate
ligand conformers and RIF/RPX mutation landscapes with selectivity
$\Delta\mathrm{RIF}_{\mathrm{sel}}$ (Stage 2); fit a decomposed surrogate
$\hat F(x)=\hat\mu_0(x)+\hat f_{\mathrm{GP}}(x)$ under cross-validation
(Stage 3); propose batches with Random / AdaLead / MCMC / BO over a
constrained design space (Stage 4); freeze predictions before synthesis,
ingest new plates through the same cleaning path, validate prospectively, and
refit (Stage 5). Stage 6 runs an ablation matrix on shared splits and seeds and
reports paired statistics.

Stages communicate only through versioned tables, manifests, hashes, and
stable identifiers. Gates (0–5) block silent misuse of failed artifacts—for
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
   AlphaFold; changing UCB $\kappa$ must not force re-running RIF.
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

### 2. Stage 0 — Ground truth and fitness

#### 2.1 Cleaning and alignment

Raw workbook rows are normalized (affinity units, fold-change, brightness
ordinals, mutation parsing from Construct/Description). Constructs are
resolved onto version backgrounds and aligned to a **canonical numbering**
(reference version, typically `V1.0`) so position `324` means the same site
across V1–V2.4. Physicochemical residue tables are attached per site.

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

Censoring policies (frozen in `fitness.yaml`) include, among others:

- Affinity: exact used as $-\log_{10}(K_d/\mathrm{\mu M})$; left-censored
  `<x` uses $x$ conservatively; right-censored `>x` omitted.
- Selectivity: positive lower bound of $K_d(\mathrm{Prop})/K_d(\mathrm{Ac})$;
  non-informative bounds omitted.

#### 2.3 Splits and Gate 0

Frozen train/held-out splits (`split_*.json`) are written once and reused by
Stages 3 and 6. Default scientific CV preference is
**leave-one-construct-out**. Gate 0 checks unique IDs, required control
mutations (`Q324R`, `A355R`), fitness reproducibility, and train/test
non-overlap.

**Primary artifact:** `data/processed/experiment_master.{pkl,parquet}`.

---

### 3. Stage 1 — Structural ensemble and confidence

Stage 1 is HPC orchestration plus structural analysis (adapters partially
stubbed until predictors are deployed).

For each `(version, method, seed, state)` the pipeline intends to produce a
`structure_model_id` and per-residue confidence features (pLDDT, pocket PAE,
cross-model RMSD). These are combined into a scalar
**structural confidence** $C_{\mathrm{structure}}\in[0,1]$ and a reliability
flag, written to `structural_confidence.parquet`.

Thresholds (illustrative, from `thresholds.yaml`):

- pLDDT minimum for reliability
- maximum Cα RMSD across models
- maximum pocket PAE

Downstream stages **never** parse raw AF2/AF3 directory layouts; they consume
the standardized confidence table.

---

### 4. Stage 2 — Physics landscape

Score direction is frozen:

$$
\text{more negative RIF/RPX} \equiv \text{better interaction (Rosetta-like)}.
$$

Parsers and gates **do not infer** this convention.

#### 4.1 Ligand conformer pipeline (2A)

For AcCoA and PropCoA. CHPC has no OpenEye OMEGA: conformers use **RDKit
ETKDG** (`builtin:rdkit`); QM uses **Gaussian16**
(`module load gaussian16/SSE4.C01`) via written `.gjf` + SLURM scripts.

```text
starting structure → conformer generation → geometry cleanup
  → QM refinement → deduplication / clustering → approved ensemble
```

Each approved structure receives a permanent `conformer_id` derived from
content hash + schema version. Catalog:
`data/physics/ligand_conformers.parquet`.

#### 4.2 Rosetta interface / packing wrappers (2B)

Stage-2 priors are **PyRosetta** mutate→pack energies (CHPC
``pyrosetta/4.0.0``), written into legacy schema columns ``rif_ac``,
``rif_prop``, and ``rpx``. RifDock / rpxdock are not used.

For each `structure_model_id` and ligand ensemble the Python layer:

1. constructs the external command from templates in `physics.yaml`
2. writes shell / optional Slurm scripts
3. captures stdout/stderr and `job.json` provenance
4. verifies completion and parses score tables

Until HPC binaries are configured, `backend: mock` exercises the same path
with deterministic pseudo-scores.

#### 4.3 Twenty–amino-acid scan (2C)

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

(and likewise for RPX / $\Delta\mathrm{RIF}_{\mathrm{sel}}$), joined to
$C_{\mathrm{structure}}$ when available.

#### 4.5 Gate 2 (2E)

Controls `Q324R` and `A355R` must show $\Delta\mathrm{RIF}_{\mathrm{sel}}$ with
the expected sign for `favorable_AcCoA` under the frozen score direction
(for `more_negative_is_better`, expected $\Delta\mathrm{RIF}_{\mathrm{sel}}<0$).
If either fails: `physics_gate = FAIL` and Stage 3 must not use physics at
full weight (falls back to `gp_zero_mean`).

---

### 5. Stage 3 — Physics-informed GP

#### 5.1 Feature construction

For construct $x$, features may include:

- sequence encodings: one-hot, Georgiev (19-D physchem), hybrid, or mutation-bag
- physics block: $\mathrm{RIF}_{\mathrm{Ac}}$, $\mathrm{RIF}_{\mathrm{Prop}}$,
  $\Delta\mathrm{RIF}_{\mathrm{sel}}$, $\mathrm{RPX}$
- structural confidence $C_{\mathrm{structure}}$

Standardization statistics are fit **inside each training split only**.

#### 5.2 Confidence weighting of physics

Raw physics features $z$ are retained, and a weighted copy is formed:

$$
z^\* = C_{\mathrm{structure}}\cdot z
$$

for each physics column. Ablations (Stage 6) compare weighted vs unweighted.

#### 5.3 Decomposed surrogate

Physics mean (train-only linear / ridge, or intercept if physics absent):

$$
\mu_0(x)
  = w_{\mathrm{RIF\,Ac}}\,\mathrm{RIF}_{\mathrm{Ac}}(x)
  + w_{\mathrm{RPX}}\,\mathrm{RPX}(x)
  + w_{\Delta}\,\Delta\mathrm{RIF}_{\mathrm{sel}}(x)
  + b
$$

(or $\mu_0\equiv\bar y$ when no physics features). Residual target:

$$
r(x) = F(x) - \mu_0(x).
$$

GP residual (zero-mean GP on features):

$$
f_{\mathrm{residual}} \sim \mathrm{GP}(0, k),\qquad
\hat F(x)=\hat\mu_0(x)+\hat f_{\mathrm{GP}}(x),\quad
\hat\sigma(x)=\hat\sigma_{\mathrm{GP}}(x).
$$

Model kinds for CV / ablations:

| Kind | Prediction |
| --- | --- |
| `physics_only` | $\hat\mu_0$ |
| `gp_zero_mean` | $\hat f_{\mathrm{GP}}$ on $F$ with $\mu_0=0$ |
| `physics_gp` | $\hat\mu_0+\hat f_{\mathrm{GP}}$ |

#### 5.4 Gate 3

On identical Stage-0 splits, compare fused vs baselines with RMSE, MAE,
Pearson, Spearman, top-$k$ ranking; paired Wilcoxon on absolute errors with
Holm adjustment; bootstrap CIs on $\Delta\mathrm{RMSE}$. Gate 3 requires
evidence that the fused model improves on both baselines.

---

### 6. Stage 4 — Active learning / search

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

“Goodness” of a physics score respects `score_direction` (e.g. negate scores
when more-negative-is-better).

#### 6.3 Acquisition and policies

Standard UCB:

$$
\mathrm{UCB}(x)=\mu(x)+\kappa\,\sigma(x).
$$

Optional effective uncertainty:

$$
\sigma_{\mathrm{eff}}^2
  = \sigma_{\mathrm{GP}}^2
  + \lambda_s\,\sigma_{\mathrm{structure}}^2
  + \lambda_p\,\sigma_{\mathrm{physics}}^2.
$$

Paper-faithful solvers (shared `propose(observed, pool, surrogate, B)` API):

- **Random:** mutate $1/N$ sites → collect $M$ → sample batch $B$.
- **AdaLead:** parents with $F\ge(1-\kappa)F_{\max}$; local / recombinant
  proposals; top-$B$ by $\mu$.
- **MCMC:** target $\pi\propto e^{\mu/T}$ (maximization form); collect $M$;
  rank by $\mu$; take top $B$.
- **BO:** enumerative UCB over the pool; top $B$.

Campaign metrics include success ratio and cumulative best fitness across
rounds/repeats.

---

### 7. Stage 5 — Prospective wet-lab loop

#### 7.1 Freeze (5A)

Before synthesis, write immutable

`round_{NN}_predictions.parquet`

with predicted fitness, 95% interval $(\hat\mu\pm z_{0.975}\hat\sigma)$,
physics/GP components, structural confidence, selection algorithm, and rank.
Hash the file (`*.sha256`) and refuse rewrites (anti–hindsight leakage).

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
2. refit physics weights and GP  
3. re-run Stage-3 calibration gates  
4. propose the next batch  

Physics coefficients are logged by round:

| Round | $w_{\mathrm{RIF\,Ac}}$ | $w_{\mathrm{RPX}}$ | $w_{\Delta\mathrm{RIF}}$ |

Weights trending toward zero as labeled data accumulate is a legitimate
scientific outcome and is recorded, not suppressed.

---

### 8. Stage 6 — Ablation and reporting

Ablation cells vary: physics on/off, GP on/off, confidence weighting,
structure source (consensus / AF2 / AF3), and prefilter on/off. Every cell
uses the **same** Stage-0 splits and random seed.

Statistics engine (default: each config vs a reference fused model):

- paired bootstrap CIs for $\Delta\mathrm{RMSE}$ / $\Delta\mathrm{MAE}$
- Wilcoxon signed-rank on absolute errors
- Holm adjustment across comparisons
- effect sizes (paired Cohen's $d$, Cliff's $\delta$)

Reporting writes metrics/comparison tables, optional figures, and
`ablation_report.md` under `outputs/stage6/`.

Stage 6 does **not** replace Gates 0–5; it supplies the scientific evidence
matrix those gates summarize.

---

### 9. End-to-end computational recipe

```text
1. Stage 0  → experiment_master + frozen splits + Gate 0
2. Stage 1  → structure_model_id ensemble + structural_confidence
3. Stage 2  → ligand_conformers + mutation scan + physics summary + Gate 2
4. Stage 3  → CV predictions + fused surrogate (+ Gate 2 weight policy) + Gate 3
5. Stage 4  → design space → prefilter → propose batches (optional freeze)
6. Wet lab  → synthesize / measure selected batch
7. Stage 5  → validate vs freeze → Gate 4 → append/refit → next batch
8. Stage 6  → ablation matrix on shared splits → statistics → report
```

Runnable entry points (Python ≥3.11):

```bash
biosensor-stage0
biosensor-stage2   # mock backend until HPC tools are set
biosensor-stage3
biosensor-stage4
biosensor-stage5 freeze|ingest ...
biosensor-stage6
```

Configuration sources of truth: `configs/pipeline.yaml`, `fitness.yaml`,
`search.yaml`, `thresholds.yaml`, `physics.yaml`, `ablation.yaml`.

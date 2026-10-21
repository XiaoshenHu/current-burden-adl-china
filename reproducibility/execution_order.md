# Execution commands and data flow

Run commands below from the checkout root after
[environment installation](environment_requirements.md). Commands execute the
existing scripts with their original statistical definitions and random seeds.
They do not invent missing data or implementations.

## Start with availability checks

```bash
python reproducibility/run.py --init-workspace
python reproducibility/run.py --list
python reproducibility/run.py --check all
```

`--check` returns **1** when an input/source is missing and **0** when listed
prerequisites exist. This checks file availability, not data contents, schema,
labels, licensing, Stata archive contents or statistical correctness. On the
public checkout, most analytical stages are expected to be **BLOCKED**.

Every stage has an exact machine-readable contract in [pipeline.json](pipeline.json).
Inspect the executable command, resolved inputs, alternatives and output paths:

```bash
python reproducibility/run.py --describe stage1
python reproducibility/run.py --describe klosa-build
```

`--run STAGE` checks that stage and executes exactly one original entrypoint using
the active Python interpreter. It returns the original process exit code. It
does not automatically run dependencies or reduce iterations. Dependency order
and script locations are recorded in the contracts. Scripts can also run
directly, for example:

```bash
python analytic_code/model_validation_and_representation/stage2_level_change.py
```

For all explicit read/write call expressions, including dynamic helper paths and
reports, see [io_inventory.md](io_inventory.md). `output_files` lists resolvable
static filenames; dynamic/conditional outputs remain documented at their source
call sites. Outputs are regenerated under `ADL_WORK_ROOT`, not written over the
released `locked_result_assets/`. Some stages update workspace index notes.

## A. Legacy CHARLS processing branch

Place permitted inputs beneath `ADL_CHARLS_RAW_DIR`:

| Relative file | Required by |
| --- | --- |
| `H_CHARLS_D_Data.dta` | Preparation, optional inventory, both legacy analyses |
| `H_CHARLS_EOL_a.dta` | Preparation and optional inventory |
| `orig_life_history/Health_History.dta` | Preparation |
| `orig_life_history/Family_Information.dta` | Preparation |
| `H_CHARLS_LH_a.dta` | Optional variable inventory only |

```bash
python reproducibility/run.py --run charls-inventory
python reproducibility/run.py --run charls-prepare
python reproducibility/run.py --run charls-legacy-time-split
```

The inventory is optional. Preparation writes `derived/charls_analytic_base.csv`
and its summary; the legacy time-split analysis writes `derived/` trajectory
features, `outputs/` tables/figures and `docs/` methods notes.

The other legacy entrypoint is currently **blocked by missing original source**:

```bash
python reproducibility/run.py --check charls-legacy-master
# Only after restoring the original reinforcement module and required data:
python reproducibility/run.py --run charls-legacy-master
```

Its output tree is `reruns/2026-03-19_landmark_primary_rerun_v1/`, containing
`derived/`, `outputs/`, `docs/`, `analysis_manifest.json` and `output_registry.csv`.
It does **not** produce the expanded v2 input package used below. See
[the source audit](missing_source_audit.md) for all eight missing callables.

## B. Locked expanded v2 CHARLS branch

This branch requires the **original** expanded v2 input tree. Set
`ADL_EXPANDED_RERUN_ROOT` to its permitted local location. The producer is absent
from this release and its history. Raw survey files or legacy branch outputs
alone cannot recreate it using the published code.

| Stage | Principal input / producer | Principal outputs under `results/` |
| --- | --- | --- |
| `stage1` | Original expanded dataset, assignments and reference tables | `stage1_baseline_lock/`: sample flow, profiles, association and performance locks |
| `stage2` | Original dataset and phenotype assignments | `stage2_level_change/`: features, model comparisons, coefficients, OOF predictions, DCA |
| `stage3` | Original dataset plus Stage 2 features/comparisons | `stage3_nested_validation/`: nested performance, OOF predictions, tuning, DCA, comparison |
| `stage4` | Original dataset and assignments | `stage4_missing_selection/`: missingness, SMD, weights, CC/MI/IPW comparisons |
| `stage5` | Original dataset and assignments | `stage5_emerging_beyond_burden/`: models, adjusted risks, contrasts, within-risk-strata summary |
| `stage6` | Stages 2, 3, 5 plus original prediction-performance table | `stage6_representation/`: representation comparison and report |
| `stage7` | Original trajectory features, assignments and posterior probabilities | `stage7_cluster_robustness/`: GMM comparisons, stability, posterior summaries |
| `stage8` | Original dataset/features/assignments plus Stage 2 features | `stage8_domain_contribution/`: ablation and continuous contributions |
| `stage9` | Original dataset and assignments | `stage9_outcome_structure/`: competing-status multinomial models |
| `stage10` | Original dataset and assignments | `stage10_targeted_sensitivity/`: older-adult and measurement robustness |
| `m1-risk` | Rebuilds Stage 2 sample from original dataset/assignments | `stageM1_risk_strat/`: quintiles, deciles, report and plot |
| `m1-calibration` | Same sample and M1 prediction helper | `stageM1_calibration/`: calibration metrics, deciles, report and plot |

Only run after each stage's `--check` passes and the input release is verified:

```bash
python reproducibility/run.py --run stage1
python reproducibility/run.py --run stage2
python reproducibility/run.py --run stage3
python reproducibility/run.py --run stage4
python reproducibility/run.py --run stage5
python reproducibility/run.py --run stage6
python reproducibility/run.py --run stage7
python reproducibility/run.py --run stage8
python reproducibility/run.py --run stage9
python reproducibility/run.py --run stage10
python reproducibility/run.py --run m1-risk
python reproducibility/run.py --run m1-calibration
```

Optional re-rendering of existing Stage 7 CSVs:

```bash
python reproducibility/run.py --run stage7-finalize
```

This optional script regenerates plots/report and replaces Stage 7 metadata with
its original finalization metadata. It does not rerun clustering.

## C. Pragmatic-rule branch

Requires the expanded v2 branch outputs. All rule-stage outputs go to
`results/stageR_rule/`.

| Stage | Required inputs / producer | Principal output |
| --- | --- | --- |
| `r1` | Stage 2 coefficients; Stages 3, 4, 8 and 10 summaries | `r1_m2_core_features.csv` and feature-stability report |
| `r2` | Stage 2 features, R1 feature table, Stage 3 nested performance | `r2_candidate_models.csv` and `stageR2_run_metadata.json` thresholds |
| `r3` | Stage 2 features, R2 candidates/thresholds, Stage 3 performance | `r3_model_rule_comparison.csv`, high-risk characteristics |
| `r4` | Same R3 input loader (including R2 thresholds) | `r4_rule_vs_emerging_overlap.csv`, `r4_operational_high_risk_profile.csv` |
| `r5` | Stage 2 features plus original dataset/assignments via Stage 4 IPW functions | `r5_internal_validation.csv` and stability report |
| `r6` | R3 comparison, R4 profile, R5 validation and initialized notes | Three synthesis/recommendation Markdown files |

```bash
python reproducibility/run.py --run r1
python reproducibility/run.py --run r2
python reproducibility/run.py --run r3
python reproducibility/run.py --run r4
python reproducibility/run.py --run r5
python reproducibility/run.py --run r6
```

Metadata JSON and additional report/plot filenames are listed by `--describe` and
in the source IO inventory. Original fixed narrative metadata have not been
rewritten; review them against any actual rerun.

## D. KLoSA external applicability branches

The following are independent of unavailable CHARLS participant data.

```bash
# Optional resource inventory; reads all 28 wave files to count rows.
python reproducibility/run.py --check klosa-audit
python reproducibility/run.py --run klosa-audit
```

The original audit requires `w01_e.dta` through `w09_e.dta`, `str01_e.dta` through
`str09_e.dta`, `Lt01_e.dta` through `Lt09_e.dta`, and `w05_new_e.dta`. Its optional
resource detection does not make the row-counting stage data-free.

For the Gateway route, supply `bukpmcwp.do`, the raw wave files, extracted exit
files, exit ZIPs and nested imputation ZIP listed by:

```bash
python reproducibility/run.py --describe klosa-build
python reproducibility/run.py --check klosa-build
python reproducibility/run.py --run klosa-build
```

The exact exit filenames and case matter on Linux. The program expects members
`w07_exit_e.dta`, `w08_exit_e.dta`, `Exit09_e.dta` in the respective exit ZIPs and
`e_imp_w01.dta` through `e_imp_w07.dta`, plus `e_imp_w05_new.dta`, inside the nested
STATA imputation ZIP. Availability checks do not validate ZIP members.

The official do-file must contain `||KLoSA_data_folder||` and `||save to folder||`
placeholders. The wrapper parameterizes these without changing transformations.
The build stages files in `ADL_GATEWAY_BUILD_ROOT` and mirrors harmonized data,
the parameterized do-file and available logs into
`results/klosa_external_validation/gateway_harmonized_build/`. Inspect the Stata
log for completion and errors; a Python process exit alone does not validate the
harmonization. Use a fresh staging directory for a different source release.

Then run:

```bash
python reproducibility/run.py --run k2
python reproducibility/run.py --run k3
```

Alternatively set `ADL_KLOSA_HARMONIZED_PATH` to an existing licensed
`GH_KLoSA_f.dta` and run K2/K3 without the build. K2 writes cohort definitions and
sample flow. K3 writes M1 performance, calibration, quintile/decile risk groups
and a summary. Its original method is KLoSA-fitted out-of-fold **structural
external replication**, not exact transport of fixed CHARLS coefficients.

The adapted PTS6 route instead requires only `str01_e.dta` through `str04_e.dta`
in `ADL_KLOSA_RAW_DIR`:

```bash
python reproducibility/run.py --check k4
python reproducibility/run.py --run k4
```

It writes PTS6 performance, group characteristics and its validation summary.
Its original adapted predictor and threshold definitions are retained. All K2-K4
outputs go to `results/klosa_external_validation/`.

## Validation scope

Installation and data-free tests are reproducible now. Statistical reruns need
the licensed input data, and the CHARLS expanded branch also needs unreleased
source or its original permitted input package. No statistical outputs were
regenerated during the repair; the 28 released locked assets are unchanged.

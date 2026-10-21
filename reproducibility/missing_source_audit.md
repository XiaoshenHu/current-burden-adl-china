# Source completeness audit

Baseline: `73ba8f9caaaa9abe78904954fde031847ce551a2`.
Audit date: 2026-10-21 (Asia/Shanghai).

## Missing reinforcement module

`analytic_code/charls_main/run_landmark_primary_rerun_master.py` originally imports
eight functions from `run_submission_grade_reinforcement`. No implementation of
that module or any of these function definitions was found in:

- The current main tree (30 Python files).
- The four reachable commits: `893d62a`, `692503d`, `0b4d4a8`, `73ba8f9`.
- The sole remote branch (`main`) and remote tag listing (no tags).
- Historical `github_release_package.zip` in `692503d` (also present in
  `0b4d4a8`). Its 30 substantive Python members match released source files byte
  for byte; it does not contain the missing module.
- GitHub releases (none at audit time).

| Missing callable | Direct caller in legacy master | Existing usage |
| --- | --- | --- |
| `fit_cox` | `fit_full_followup_cox_supplements` | Main, subgroup, interaction, IPW and sensitivity Cox fits |
| `cox_table` | `fit_full_followup_cox_supplements` | Model effect tables |
| `make_ipw_weights` | `fit_full_followup_cox_supplements` | Complete-case selection weights and summary |
| `ph_diagnostics` | `fit_full_followup_cox_supplements` | Proportional hazards diagnostics |
| `plot_schoenfeld` | `fit_full_followup_cox_supplements` | Residual diagnostic figure |
| `spline_curve` | `fit_full_followup_cox_supplements` | Mortality spline curve data and figure |
| `draw_flowchart` | `main` | Sample-flow figure |
| `make_baseline_table` | `main` | Baseline descriptions by adversity and phenotype |

Call sites do **not** establish the original penalization, weight estimation,
truncation, missing-data handling, spline specification or diagnostics settings.
Plausible replacements would not establish methodological identity. No
replacement functions, placeholder results or skipped statistical steps were
added.

The module is now required at the start of `main()` before analysis outputs are
written, and at the start of the Cox helper if called independently. Absence
produces an actionable error while allowing import-only inspection. The runner
marks `charls-legacy-master` blocked. Restoring the same callable names establishes
availability only; source and transitive dependencies still need validation.

To unblock, restore the **original** source at
`analytic_code/charls_main/run_submission_grade_reinforcement.py`, port additional
original imports/paths, and verify with licensed data and the original output
registry. Do not substitute a new implementation solely to make imports pass.

## Separate missing expanded v2 producer

Stages 1-10 and the M1 helpers read the tree
`rerun_risk_stratification_expanded_v2_20260323/`, particularly:

| Required path within that tree | Consumers |
| --- | --- |
| `02_data_processing/analysis_dataset_landmark.csv` | Stages 1-5, 8-10; M1 helpers; R5 IPW |
| `03_phenotype_derivation/phenotype_assignments.csv` | Stages 1, 2, 4, 5, 7-10; M1 helpers; R5 IPW |
| `03_phenotype_derivation/trajectory_feature_matrix.csv` | Stages 7 and 8 |
| `03_phenotype_derivation/posterior_probabilities.csv` | Stage 7 |
| `03_phenotype_derivation/class_sizes.csv` | Stage 1 |
| `04_primary_models/primary_incident_adl_crude_risk.csv` | Stage 1 |
| Six `07_tables_figures/table_*.csv` reference tables | Stage 1; prediction performance also used by Stage 6 |

Exact reference names are in [pipeline.json](pipeline.json). No producer for this
tree was found in released write sites or the historical archive. The legacy
master writes `reruns/2026-03-19_landmark_primary_rerun_v1/`, with different paths,
structures and phenotype labels. The raw-data preparer writes
`derived/charls_analytic_base.csv`, not the expanded v2 dataset.

These outputs are not interchangeable. No renaming, fabricated mapping,
reclustering, or copying of aggregate locked assets into participant-level input
locations was performed. Raw survey data alone do not fill this source gap.
Obtain the original expanded v2 producer plus provenance, or the original
permitted local input package, before running that branch.

## Confirmed layout repairs

- Runtime developer-home paths replaced with configuration; incorrect
  `parents[1]` roots from the packaged relocation no longer separate preparation
  and analysis outputs into different directories.
- Cross-directory imports use repository package paths: stage7 finalization,
  K2/K3 harmonized helpers, M1 helpers, R4 and R5.
- Both identical Gateway build entrypoints retained with the same configuration
  and executable discovery. No Stata transformation code was invented.
- Output-directory creation added to both CHARLS preparation entrypoints.
- KLoSA feasibility preflight explicitly lists all 28 raw wave files, because
  the original audit reads them to count rows even when optional resources are
  absent. No audit counts have been fabricated for missing files.
- `manifests/manifest_github.*` and original release QC notes remain historical
  provenance. Their original source paths and counts do not describe this
  repaired checkout. They are not runtime configuration.

## Verification boundaries

Data-free checks cover imports, path overrides (including spaces), preflight,
missing-source errors, unchanged statistical function bodies/constants against
baseline, and hashes of all 28 locked assets. The AST comparison permits only
executable discovery, initialization and deferred dependency loading changes.
It does not prove numerical equivalence.

No individual-level CHARLS/KLoSA data were present. Sample construction,
coefficients, discrimination, calibration, uncertainty estimates and Stata
harmonization remain **unverified by reanalysis**. Manuscript values and result
definitions were retained, not independently endorsed or recalculated. Some
original reports include fixed narrative/sample-size metadata; review these
against actual reruns before use with another data release.

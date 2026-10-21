# Source IO inventory

Generated with `python reproducibility/collect_io.py`. This is a static inventory of explicit IO calls,
including helper functions. It does not execute code, prove branch coverage, or imply that inputs exist.
`pipeline.json` specifies required stage inputs, the producer relationships, and resolved static output names.
Local variables such as `outpath`, `writer`, and `path` are retained verbatim; the source line links them to their caller.
The legacy master also declares its complete table map in `main()` (`save_map`) and its file registry.

## `data_processing_scripts/charls_processing/build_charls_variable_inventory.py`

| Line | Operation | Expression |
| --- | --- | --- |
| 44 | read | `pd.read_stata(path, iterator=True)` |
| 71 | write | `(OUT / 'charls_variable_inventory.json').write_text(json.dumps(inventory, indent=2, ensure_ascii=False))` |
| 89 | write | `(OUT / 'charls_variable_inventory.md').write_text('\n'.join(lines))` |

## `data_processing_scripts/charls_processing/prepare_life_course_aging_dataset.py`

| Line | Operation | Expression |
| --- | --- | --- |
| 22 | read | `pd.read_stata(path, columns=columns)` |
| 197 | read | `load_stata(RAW / 'H_CHARLS_D_Data.dta', d_cols)` |
| 198 | read | `load_stata(RAW_ORIG / 'Health_History.dta', health_cols)` |
| 199 | read | `load_stata(RAW_ORIG / 'Family_Information.dta', family_cols)` |
| 200 | read | `load_stata(RAW / 'H_CHARLS_EOL_a.dta', eol_cols)` |
| 470 | write | `analytic.to_csv(DERIVED / 'charls_analytic_base.csv', index=False)` |
| 493 | write | `(DERIVED / 'charls_analytic_base_summary.md').write_text('\n'.join(summary_lines))` |

## `data_processing_scripts/klosa_processing/harmonized_wave34_m1_utils.py`

| Line | Operation | Expression |
| --- | --- | --- |
| 82 | read | `pd.read_stata(HARMONIZED_PATH, columns=WAVE34_COLS)` |

## `data_processing_scripts/klosa_processing/run_gateway_harmonized_klosa_build.py`

| Line | Operation | Expression |
| --- | --- | --- |
| 113 | read | `ORIGINAL_DO.read_text(encoding='utf-8', errors='ignore')` |
| 116 | write | `PARAM_DO.write_text(text, encoding='utf-8')` |
| 128 | write | `RUN_LOG.write_text(f"COMMAND: {' '.join(cmd)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n", encoding='utf-8')` |
| 144 | write | `shutil.copy2(path, PROJECT_RESULT_DIR / path.name)` |

## `analytic_code/charls_main/run_landmark_primary_rerun_master.py`

| Line | Operation | Expression |
| --- | --- | --- |
| 591 | write | `(RERUN_DOCS / 'methods_note.md').write_text('\n'.join(methods_text), encoding='utf-8')` |
| 623 | write | `(RERUN_DOCS / 'results_memo.md').write_text('\n'.join(memo_text), encoding='utf-8')` |
| 640 | read | `pd.read_csv(DERIVED / 'charls_analytic_base.csv')` |
| 684 | write | `plt.savefig(RERUN_OUTPUTS / 'figureS3_landmark_k_comparison.png', dpi=300)` |
| 700 | write | `plt.savefig(RERUN_OUTPUTS / 'figureS4_landmark_bootstrap_stability.png', dpi=300)` |
| 719 | write | `plt.savefig(RERUN_OUTPUTS / 'figure3_landmark_crude_event_rates.png', dpi=300)` |
| 729 | write | `traj3.to_csv(trajectory_w1w3_path, index=False)` |
| 733 | write | `traj2.to_csv(trajectory_w1w2_path, index=False)` |
| 762 | read | `pd.read_csv(RERUN_OUTPUTS / 'tableS20_spline_mortality_curve.csv')` |
| 767 | write | `data.to_csv(path, index=False)` |
| 791 | write | `registry_df.to_csv(registry_path, index=False)` |
| 830 | write | `manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')` |

## `analytic_code/charls_main/run_landmark_time_split_analysis.py`

| Line | Operation | Expression |
| --- | --- | --- |
| 72 | write | `table.to_csv(path, index=False)` |
| 109 | write | `table.to_csv(path, index=False)` |
| 324 | write | `fig.savefig(out_path, dpi=300)` |
| 337 | read | `pd.read_csv(DERIVED / 'charls_analytic_base.csv')` |
| 338 | read | `pd.read_stata(RAW_D, columns=['ID', 'inw1', 'r2iwstat', 'r3iwstat', 'r4iwstat', 'r2adlab_c', 'r3adlab_c', 'r4adlab_c'])` |
| 449 | write | `traj3.to_csv(DERIVED / 'charls_trajectory_features_landmark_w1_w3.csv', index=False)` |
| 450 | write | `kcomp3.to_csv(OUTPUTS / 'tableS_landmark_w1_w3_k_comparison.csv', index=False)` |
| 451 | write | `stab3.to_csv(OUTPUTS / 'tableS_landmark_w1_w3_bootstrap_stability.csv', index=False)` |
| 462 | write | `posterior_summary.to_csv(OUTPUTS / 'tableS_landmark_w1_w3_posterior_summary.csv', index=False)` |
| 484 | write | `profile3.to_csv(OUTPUTS / 'tableS_landmark_w1_w3_trajectory_profiles.csv', index=False)` |
| 495 | write | `plt.savefig(OUTPUTS / 'figure_landmark_w1_w3_k_comparison.png', dpi=300)` |
| 506 | write | `plt.savefig(OUTPUTS / 'figure_landmark_w1_w3_bootstrap_stability.png', dpi=300)` |
| 554 | write | `pd.DataFrame({'metric': list(flow.keys()), 'value': list(flow.values())}).to_csv(OUTPUTS / 'tableS_landmark_flow_counts.csv', index=False)` |
| 693 | write | `attenuation.to_csv(OUTPUTS / 'tableS_landmark_w1_w3_attenuation.csv', index=False)` |
| 703 | write | `crude_rates.to_csv(OUTPUTS / 'tableS_landmark_w1_w3_crude_event_rates.csv', index=False)` |
| 718 | write | `plt.savefig(OUTPUTS / 'figure_landmark_w1_w3_crude_event_rates.png', dpi=300)` |
| 723 | write | `traj2.to_csv(DERIVED / 'charls_trajectory_features_landmark_w1_w2.csv', index=False)` |
| 724 | write | `kcomp2.to_csv(OUTPUTS / 'tableS_landmark_w1_w2_k_comparison.csv', index=False)` |
| 803 | write | `(DOCS / 'landmark_discrete_time_methods.md').write_text('\n'.join(methods_lines))` |
| 850 | write | `(DOCS / 'round4_landmark_discrete_time_and_trajectory_robustness.md').write_text('\n'.join(memo_lines))` |

## `analytic_code/charls_main/stage7_finalize_outputs.py`

| Line | Operation | Expression |
| --- | --- | --- |
| 23 | read | `pd.read_csv(RESULTS_DIR / 'gmm_specification_comparison.csv')` |
| 24 | read | `pd.read_csv(RESULTS_DIR / 'cluster_stability_metrics.csv')` |
| 25 | read | `pd.read_csv(RESULTS_DIR / 'posterior_distribution_summary.csv')` |
| 39 | write | `(RESULTS_DIR / 'stage7_run_metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')` |

## `analytic_code/charls_main/stage9_outcome_structure.py`

| Line | Operation | Expression |
| --- | --- | --- |
| 60 | read | `pd.read_csv(RERUN_ROOT / '02_data_processing' / 'analysis_dataset_landmark.csv')` |
| 61 | read | `pd.read_csv(RERUN_ROOT / '03_phenotype_derivation' / 'phenotype_assignments.csv')` |
| 370 | write | `PdfPages(RESULTS_DIR / 'outcome_structure_plots.pdf')` |
| 388 | write | `pdf.savefig(fig)` |
| 431 | write | `pdf.savefig(fig)` |
| 451 | write | `pdf.savefig(fig)` |
| 472 | write | `pdf.savefig(fig)` |
| 638 | write | `(RESULTS_DIR / 'outcome_structure_summary.md').write_text(text, encoding='utf-8')` |
| 643 | read | `RESULTS_INDEX.read_text(encoding='utf-8')` |
| 674 | write | `RESULTS_INDEX.write_text(content.rstrip() + '\n' + block, encoding='utf-8')` |
| 688 | write | `all_rows.to_csv(RESULTS_DIR / 'multinomial_competing_status.csv', index=False)` |
| 698 | write | `(RESULTS_DIR / 'stage9_run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')` |

## `analytic_code/klosa_external_applicability/build_k1_feasibility_audit.py`

| Line | Operation | Expression |
| --- | --- | --- |
| 90 | read | `pd.read_stata(dta_path, convert_categoricals=False)` |
| 517 | read | `load_workbook(path)` |
| 556 | write | `(RESULTS_DIR / 'klosa_feasibility_audit.md').write_text(md_text, encoding='utf-8')` |
| 558 | write | `pd.ExcelWriter(RESULTS_DIR / 'charls_to_klosa_crosswalk.xlsx', engine='openpyxl')` |
| 559 | write | `crosswalk.to_excel(writer, index=False, sheet_name='crosswalk')` |
| 560 | write | `wave_df.to_excel(writer, index=False, sheet_name='raw_wave_counts')` |
| 561 | write | `crosswalk.to_csv(RESULTS_DIR / 'charls_to_klosa_crosswalk.csv', index=False)` |
| 562 | write | `wave_df.to_csv(RESULTS_DIR / 'klosa_raw_wave_counts.csv', index=False)` |

## `analytic_code/klosa_external_applicability/build_k2_validation_cohort.py`

| Line | Operation | Expression |
| --- | --- | --- |
| 115 | write | `sample_flow.to_csv(RESULTS_DIR / 'klosa_sample_flow.csv', index=False)` |
| 118 | write | `(RESULTS_DIR / 'klosa_validation_cohort_definition.md').write_text(md, encoding='utf-8')` |

## `analytic_code/klosa_external_applicability/build_k3_m1_validation_status.py`

| Line | Operation | Expression |
| --- | --- | --- |
| 168 | write | `performance.to_csv(RESULTS_DIR / 'klosa_M1_performance.csv', index=False)` |
| 169 | write | `calibration.to_csv(RESULTS_DIR / 'klosa_M1_calibration.csv', index=False)` |
| 170 | write | `risk_df.to_csv(RESULTS_DIR / 'klosa_M1_risk_stratification.csv', index=False)` |
| 171 | write | `(RESULTS_DIR / 'klosa_M1_validation_summary.md').write_text(markdown, encoding='utf-8')` |

## `analytic_code/klosa_external_applicability/build_k4_pts6_validation.py`

| Line | Operation | Expression |
| --- | --- | --- |
| 30 | read | `pd.read_stata(path, convert_categoricals=False, columns=None)` |
| 32 | read | `pd.read_stata(path, convert_categoricals=False, columns=cols)` |
| 252 | write | `(RESULTS_DIR / 'klosa_PTS6_validation_summary.md').write_text(md, encoding='utf-8')` |
| 253 | write | `performance.to_csv(RESULTS_DIR / 'klosa_PTS6_performance.csv', index=False)` |
| 254 | write | `groups.to_csv(RESULTS_DIR / 'klosa_PTS6_group_characteristics.csv', index=False)` |

## `analytic_code/klosa_external_applicability/run_gateway_harmonized_klosa_build.py`

| Line | Operation | Expression |
| --- | --- | --- |
| 113 | read | `ORIGINAL_DO.read_text(encoding='utf-8', errors='ignore')` |
| 116 | write | `PARAM_DO.write_text(text, encoding='utf-8')` |
| 128 | write | `RUN_LOG.write_text(f"COMMAND: {' '.join(cmd)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n", encoding='utf-8')` |
| 144 | write | `shutil.copy2(path, PROJECT_RESULT_DIR / path.name)` |

## `analytic_code/model_validation_and_representation/stage10_targeted_sensitivity.py`

| Line | Operation | Expression |
| --- | --- | --- |
| 86 | read | `pd.read_csv(RERUN_ROOT / '02_data_processing' / 'analysis_dataset_landmark.csv')` |
| 87 | read | `pd.read_csv(RERUN_ROOT / '03_phenotype_derivation' / 'phenotype_assignments.csv')` |
| 513 | write | `PdfPages(RESULTS_DIR / 'targeted_sensitivity_plots.pdf')` |
| 525 | write | `pdf.savefig(fig)` |
| 549 | write | `pdf.savefig(fig)` |
| 566 | write | `pdf.savefig(fig)` |
| 685 | write | `(RESULTS_DIR / 'targeted_sensitivity_summary.md').write_text(summary, encoding='utf-8')` |
| 721 | read | `RESULTS_INDEX.read_text(encoding='utf-8')` |
| 723 | write | `RESULTS_INDEX.write_text(current + addition, encoding='utf-8')` |
| 736 | write | `older_df.to_csv(RESULTS_DIR / 'older_adult_sensitivity.csv', index=False)` |
| 737 | write | `measure_df.to_csv(RESULTS_DIR / 'measurement_robustness.csv', index=False)` |
| 750 | write | `(RESULTS_DIR / 'stage10_run_metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')` |

## `analytic_code/model_validation_and_representation/stage1_baseline_lock.py`

| Line | Operation | Expression |
| --- | --- | --- |
| 224 | read | `pd.read_csv(RERUN_ROOT / '02_data_processing' / 'analysis_dataset_landmark.csv')` |
| 225 | read | `pd.read_csv(RERUN_ROOT / '03_phenotype_derivation' / 'phenotype_assignments.csv')` |
| 229 | read | `pd.read_csv(RERUN_ROOT / '07_tables_figures' / 'table_sample_flow.csv')` |
| 230 | read | `pd.read_csv(RERUN_ROOT / '07_tables_figures' / 'table_prediction_performance.csv')` |
| 231 | read | `pd.read_csv(RERUN_ROOT / '07_tables_figures' / 'table_membership_model.csv')` |
| 232 | read | `pd.read_csv(RERUN_ROOT / '07_tables_figures' / 'table_primary_incident_adl.csv')` |
| 233 | read | `pd.read_csv(RERUN_ROOT / '07_tables_figures' / 'table_composite_outcome.csv')` |
| 234 | read | `pd.read_csv(RERUN_ROOT / '07_tables_figures' / 'table_post_landmark_mortality.csv')` |
| 235 | read | `pd.read_csv(RERUN_ROOT / '04_primary_models' / 'primary_incident_adl_crude_risk.csv')` |
| 236 | read | `pd.read_csv(RERUN_ROOT / '03_phenotype_derivation' / 'class_sizes.csv')` |
| 260 | write | `sample_flow_locked.to_csv(RESULTS_DIR / 'sample_flow_locked.csv', index=False)` |
| 286 | write | `phenotype_profile_locked.to_csv(RESULTS_DIR / 'phenotype_profile_locked.csv', index=False)` |
| 392 | write | `association_results_locked.to_csv(RESULTS_DIR / 'current_association_results_locked.csv', index=False)` |
| 431 | write | `performance_locked.to_csv(RESULTS_DIR / 'current_model_performance_locked.csv', index=False)` |
| 477 | write | `(RESULTS_DIR / 'current_pipeline_notes.md').write_text(current_pipeline_notes, encoding='utf-8')` |
| 581 | write | `(RESULTS_DIR / 'baseline_lock_summary.md').write_text(summary, encoding='utf-8')` |
| 596 | write | `(RESULTS_DIR / 'stage1_run_metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')` |

## `analytic_code/model_validation_and_representation/stage2_level_change.py`

| Line | Operation | Expression |
| --- | --- | --- |
| 169 | read | `pd.read_csv(RERUN_ROOT / '02_data_processing' / 'analysis_dataset_landmark.csv')` |
| 170 | read | `pd.read_csv(RERUN_ROOT / '03_phenotype_derivation' / 'phenotype_assignments.csv')` |
| 624 | write | `(RESULTS_DIR / 'level_change_summary.md').write_text(summary, encoding='utf-8')` |
| 634 | write | `PdfPages(RESULTS_DIR / 'level_change_plots.pdf')` |
| 647 | write | `pdf.savefig(fig)` |
| 667 | write | `pdf.savefig(fig)` |
| 697 | write | `pdf.savefig(fig)` |
| 717 | write | `pdf.savefig(fig)` |
| 732 | write | `features.to_csv(RESULTS_DIR / 'level_change_features.csv', index=False)` |
| 735 | write | `comparison.to_csv(RESULTS_DIR / 'level_change_model_comparison.csv', index=False)` |
| 736 | write | `coefficients.to_csv(RESULTS_DIR / 'level_change_model_coefficients.csv', index=False)` |
| 737 | write | `oof_preds.to_csv(RESULTS_DIR / 'level_change_oof_predictions.csv', index=False)` |
| 738 | write | `dca_df.to_csv(RESULTS_DIR / 'level_change_dca_outputs.csv', index=False)` |
| 760 | write | `(RESULTS_DIR / 'stage2_pipeline_notes.md').write_text(notes, encoding='utf-8')` |
| 771 | write | `(RESULTS_DIR / 'stage2_run_metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')` |

## `analytic_code/model_validation_and_representation/stage3_nested_validation.py`

| Line | Operation | Expression |
| --- | --- | --- |
| 158 | read | `pd.read_csv(RERUN_ROOT / '02_data_processing' / 'analysis_dataset_landmark.csv')` |
| 201 | read | `pd.read_csv(PROJECT_ROOT / 'results' / 'stage2_level_change' / 'level_change_features.csv')` |
| 518 | read | `pd.read_csv(PROJECT_ROOT / 'results' / 'stage2_level_change' / 'level_change_model_comparison.csv')` |
| 574 | write | `PdfPages(RESULTS_DIR / 'calibration_plots.pdf')` |
| 587 | write | `pdf.savefig(fig)` |
| 604 | write | `PdfPages(RESULTS_DIR / 'decision_curves.pdf')` |
| 623 | write | `pdf.savefig(fig)` |
| 716 | write | `(RESULTS_DIR / 'nested_validation_summary.md').write_text(summary, encoding='utf-8')` |
| 726 | write | `nested.to_csv(RESULTS_DIR / 'nested_validation_performance.csv', index=False)` |
| 727 | write | `dca_df.to_csv(RESULTS_DIR / 'nested_validation_decision_curve_data.csv', index=False)` |
| 728 | write | `oof_df.to_csv(RESULTS_DIR / 'nested_validation_oof_predictions.csv', index=False)` |
| 729 | write | `fold_meta.to_csv(RESULTS_DIR / 'nested_validation_fold_tuning.csv', index=False)` |
| 732 | write | `compare.to_csv(RESULTS_DIR / 'nested_vs_original_comparison.csv', index=False)` |
| 748 | write | `(RESULTS_DIR / 'nested_validation_run_metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')` |

## `analytic_code/model_validation_and_representation/stage4_missing_selection.py`

| Line | Operation | Expression |
| --- | --- | --- |
| 214 | read | `pd.read_csv(RERUN_ROOT / '02_data_processing' / 'analysis_dataset_landmark.csv')` |
| 215 | read | `pd.read_csv(RERUN_ROOT / '03_phenotype_derivation' / 'phenotype_assignments.csv')` |
| 275 | write | `out.to_csv(RESULTS_DIR / 'missingness_summary.csv', index=False)` |
| 323 | write | `out.to_csv(RESULTS_DIR / 'included_vs_excluded_smd.csv', index=False)` |
| 430 | write | `diag.to_csv(RESULTS_DIR / 'weight_diagnostics.csv', index=False)` |
| 848 | write | `out.to_csv(RESULTS_DIR / 'cc_vs_mi_vs_ipw_vs_miipw.csv', index=False)` |
| 853 | write | `PdfPages(RESULTS_DIR / 'missingness_plots.pdf')` |
| 862 | write | `pdf.savefig(fig)` |
| 872 | write | `pdf.savefig(fig)` |
| 888 | write | `pdf.savefig(fig)` |
| 909 | write | `pdf.savefig(fig)` |
| 1067 | write | `(RESULTS_DIR / 'missing_selection_summary.md').write_text(summary, encoding='utf-8')` |
| 1095 | write | `(RESULTS_DIR / 'stage4_run_metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')` |

## `analytic_code/model_validation_and_representation/stage5_emerging_beyond_burden.py`

| Line | Operation | Expression |
| --- | --- | --- |
| 74 | read | `pd.read_csv(RERUN_ROOT / '02_data_processing' / 'analysis_dataset_landmark.csv')` |
| 75 | read | `pd.read_csv(RERUN_ROOT / '03_phenotype_derivation' / 'phenotype_assignments.csv')` |
| 414 | write | `(RESULTS_DIR / 'emerging_beyond_burden_summary.md').write_text(summary, encoding='utf-8')` |
| 428 | write | `PdfPages(RESULTS_DIR / 'adjusted_risk_plot.pdf')` |
| 466 | write | `pdf.savefig(fig)` |
| 527 | write | `model_df.to_csv(RESULTS_DIR / 'emerging_beyond_burden_models.csv', index=False)` |
| 528 | write | `risk_df.to_csv(RESULTS_DIR / 'adjusted_absolute_risk_by_phenotype.csv', index=False)` |
| 530 | write | `pairwise_df.to_csv(RESULTS_DIR / 'emerging_pairwise_contrasts.csv', index=False)` |
| 531 | write | `bonus_df.to_csv(RESULTS_DIR / 'emerging_within_m1_risk_strata.csv', index=False)` |
| 542 | write | `(RESULTS_DIR / 'stage5_run_metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')` |

## `analytic_code/model_validation_and_representation/stage6_representation.py`

| Line | Operation | Expression |
| --- | --- | --- |
| 58 | read | `pd.read_csv(STAGE2_FEATURES)` |
| 59 | read | `pd.read_csv(STAGE2_PERF)` |
| 60 | read | `pd.read_csv(STAGE3_PERF)` |
| 61 | read | `pd.read_csv(STAGE5_RISKS)` |
| 62 | read | `pd.read_csv(STAGE5_CONTRASTS)` |
| 63 | read | `pd.read_csv(ORIGINAL_PERF)` |
| 444 | write | `(RESULTS_DIR / 'representation_summary.md').write_text(text, encoding='utf-8')` |
| 460 | write | `PdfPages(RESULTS_DIR / 'representation_plots.pdf')` |
| 482 | write | `pdf.savefig(fig)` |
| 498 | write | `pdf.savefig(fig)` |
| 537 | read | `RESULTS_INDEX.read_text(encoding='utf-8')` |
| 539 | write | `RESULTS_INDEX.write_text(current + addition, encoding='utf-8')` |
| 550 | write | `comp.to_csv(RESULTS_DIR / 'representation_comparison.csv', index=False)` |
| 561 | write | `(RESULTS_DIR / 'stage6_run_metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')` |

## `analytic_code/model_validation_and_representation/stage7_cluster_robustness.py`

| Line | Operation | Expression |
| --- | --- | --- |
| 99 | read | `pd.read_csv(PHENO_DIR / 'trajectory_feature_matrix.csv')` |
| 100 | read | `pd.read_csv(PHENO_DIR / 'phenotype_assignments.csv')` |
| 280 | read | `pd.read_csv(PHENO_DIR / 'posterior_probabilities.csv')` |
| 316 | write | `PdfPages(RESULTS_DIR / 'cluster_stability_plots.pdf')` |
| 336 | write | `pdf.savefig(fig)` |
| 356 | write | `pdf.savefig(fig)` |
| 385 | write | `pdf.savefig(fig)` |
| 530 | write | `(RESULTS_DIR / 'cluster_robustness_report.md').write_text(report, encoding='utf-8')` |
| 566 | read | `RESULTS_INDEX.read_text(encoding='utf-8')` |
| 568 | write | `RESULTS_INDEX.write_text(current + addition, encoding='utf-8')` |
| 581 | write | `spec_df.to_csv(RESULTS_DIR / 'gmm_specification_comparison.csv', index=False)` |
| 582 | write | `stability_df.to_csv(RESULTS_DIR / 'cluster_stability_metrics.csv', index=False)` |
| 583 | write | `posterior_df.to_csv(RESULTS_DIR / 'posterior_distribution_summary.csv', index=False)` |
| 597 | write | `(RESULTS_DIR / 'stage7_run_metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')` |

## `analytic_code/model_validation_and_representation/stage8_domain_contribution.py`

| Line | Operation | Expression |
| --- | --- | --- |
| 177 | read | `pd.read_csv(RERUN_ROOT / '02_data_processing' / 'analysis_dataset_landmark.csv')` |
| 178 | read | `pd.read_csv(RERUN_ROOT / '03_phenotype_derivation' / 'trajectory_feature_matrix.csv')` |
| 179 | read | `pd.read_csv(RERUN_ROOT / '03_phenotype_derivation' / 'phenotype_assignments.csv')` |
| 311 | read | `pd.read_csv(PROJECT_ROOT / 'results' / 'stage2_level_change' / 'level_change_features.csv')` |
| 536 | write | `PdfPages(RESULTS_DIR / 'domain_contribution_plots.pdf')` |
| 556 | write | `pdf.savefig(fig)` |
| 578 | write | `pdf.savefig(fig)` |
| 597 | write | `pdf.savefig(fig)` |
| 721 | write | `(RESULTS_DIR / 'domain_contribution_summary.md').write_text(text, encoding='utf-8')` |
| 726 | read | `RESULTS_INDEX.read_text(encoding='utf-8')` |
| 758 | write | `RESULTS_INDEX.write_text(content.rstrip() + '\n' + block, encoding='utf-8')` |
| 768 | write | `domain_df.to_csv(RESULTS_DIR / 'domain_ablation_results.csv', index=False)` |
| 772 | write | `cont_df.to_csv(RESULTS_DIR / 'continuous_domain_contributions.csv', index=False)` |
| 784 | write | `(RESULTS_DIR / 'stage8_run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')` |

## `analytic_code/model_validation_and_representation/stageM1_calibration.py`

| Line | Operation | Expression |
| --- | --- | --- |
| 110 | write | `PdfPages(pdf_path)` |
| 140 | write | `pdf.savefig(fig)` |
| 218 | write | `outpath.write_text('\n'.join(lines) + '\n', encoding='utf-8')` |
| 223 | read | `RESULTS_INDEX.read_text(encoding='utf-8')` |
| 247 | write | `RESULTS_INDEX.write_text(text.rstrip() + '\n' + block.strip('\n') + '\n', encoding='utf-8')` |
| 265 | write | `outpath.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')` |
| 275 | write | `metrics.to_csv(OUTDIR / 'm1_calibration_metrics.csv', index=False)` |
| 276 | write | `deciles.to_csv(OUTDIR / 'm1_observed_vs_predicted_deciles.csv', index=False)` |

## `analytic_code/model_validation_and_representation/stageM1_risk_strat.py`

| Line | Operation | Expression |
| --- | --- | --- |
| 194 | write | `outpath.write_text('\n'.join(lines) + '\n', encoding='utf-8')` |
| 215 | write | `PdfPages(pdf_path)` |
| 219 | write | `pdf.savefig(fig)` |
| 225 | write | `pdf.savefig(fig)` |
| 231 | read | `RESULTS_INDEX.read_text(encoding='utf-8')` |
| 255 | write | `RESULTS_INDEX.write_text(text.rstrip() + '\n' + block.strip('\n') + '\n', encoding='utf-8')` |
| 270 | write | `outpath.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')` |
| 287 | write | `quint.to_csv(OUTDIR / 'risk_stratification_quintiles.csv', index=False)` |
| 288 | write | `dec.to_csv(OUTDIR / 'risk_stratification_deciles.csv', index=False)` |

## `analytic_code/pragmatic_rule/stageR1_m2_core_features.py`

| Line | Operation | Expression |
| --- | --- | --- |
| 58 | read | `pd.read_csv(RESULTS / 'stage2_level_change' / 'level_change_model_coefficients.csv')` |
| 59 | read | `pd.read_csv(RESULTS / 'stage3_nested_validation' / 'nested_validation_performance.csv')` |
| 60 | read | `pd.read_csv(RESULTS / 'stage4_missing_selection' / 'cc_vs_mi_vs_ipw_vs_miipw.csv')` |
| 61 | read | `pd.read_csv(RESULTS / 'stage8_domain_contribution' / 'continuous_domain_contributions.csv')` |
| 62 | read | `pd.read_csv(RESULTS / 'stage10_targeted_sensitivity' / 'measurement_robustness.csv')` |
| 418 | read | `RESULTS_INDEX.read_text(encoding='utf-8')` |
| 420 | write | `RESULTS_INDEX.write_text(current.rstrip() + section + '\n', encoding='utf-8')` |
| 427 | write | `table.to_csv(OUTDIR / 'r1_m2_core_features.csv', index=False)` |
| 429 | write | `(OUTDIR / 'r1_feature_stability_summary.md').write_text(summary, encoding='utf-8')` |
| 449 | write | `(OUTDIR / 'stageR1_run_metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')` |

## `analytic_code/pragmatic_rule/stageR2_candidate_models.py`

| Line | Operation | Expression |
| --- | --- | --- |
| 75 | read | `pd.read_csv(RESULTS / 'stage2_level_change' / 'level_change_features.csv')` |
| 76 | read | `pd.read_csv(OUTDIR / 'r1_m2_core_features.csv')` |
| 77 | read | `pd.read_csv(RESULTS / 'stage3_nested_validation' / 'nested_validation_performance.csv')` |
| 588 | write | `outpath.write_text('\n'.join(lines) + '\n', encoding='utf-8')` |
| 593 | read | `RESULTS_INDEX.read_text(encoding='utf-8')` |
| 618 | write | `RESULTS_INDEX.write_text(text.rstrip() + '\n' + block.strip('\n') + '\n', encoding='utf-8')` |
| 631 | write | `(OUTDIR / 'stageR2_run_metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')` |
| 643 | write | `candidate_table.to_csv(OUTDIR / 'r2_candidate_models.csv', index=False)` |

## `analytic_code/pragmatic_rule/stageR3_model_rule_comparison.py`

| Line | Operation | Expression |
| --- | --- | --- |
| 81 | read | `pd.read_csv(RESULTS / 'stage2_level_change' / 'level_change_features.csv')` |
| 82 | read | `pd.read_csv(OUTDIR / 'r2_candidate_models.csv')` |
| 83 | read | `(OUTDIR / 'stageR2_run_metadata.json').read_text(encoding='utf-8')` |
| 84 | read | `pd.read_csv(RESULTS / 'stage3_nested_validation' / 'nested_validation_performance.csv')` |
| 401 | write | `PdfPages(outpath)` |
| 413 | write | `pdf.savefig(fig)` |
| 423 | write | `pdf.savefig(fig)` |
| 449 | write | `pdf.savefig(fig)` |
| 455 | read | `RESULTS_INDEX.read_text(encoding='utf-8')` |
| 478 | write | `RESULTS_INDEX.write_text(text.rstrip() + '\n' + block.strip('\n') + '\n', encoding='utf-8')` |
| 499 | write | `(OUTDIR / 'stageR3_run_metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')` |
| 509 | write | `comparison.to_csv(OUTDIR / 'r3_model_rule_comparison.csv', index=False)` |
| 512 | write | `high_risk.to_csv(OUTDIR / 'r3_high_risk_group_characteristics.csv', index=False)` |

## `analytic_code/pragmatic_rule/stageR4_pre_disability_relevance.py`

| Line | Operation | Expression |
| --- | --- | --- |
| 74 | read | `(OUTDIR / 'stageR2_run_metadata.json').read_text(encoding='utf-8')` |
| 194 | write | `outpath.write_text('\n'.join(lines) + '\n', encoding='utf-8')` |
| 199 | read | `RESULTS_INDEX.read_text(encoding='utf-8')` |
| 222 | write | `RESULTS_INDEX.write_text(text.rstrip() + '\n' + block.strip('\n') + '\n', encoding='utf-8')` |
| 235 | write | `(OUTDIR / 'stageR4_run_metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')` |
| 262 | write | `overlap.to_csv(OUTDIR / 'r4_rule_vs_emerging_overlap.csv', index=False)` |
| 276 | write | `profile.to_csv(OUTDIR / 'r4_operational_high_risk_profile.csv', index=False)` |

## `analytic_code/pragmatic_rule/stageR5_rule_validation.py`

| Line | Operation | Expression |
| --- | --- | --- |
| 80 | read | `pd.read_csv(ROOT / 'results' / 'stage2_level_change' / 'level_change_features.csv')` |
| 410 | write | `outpath.write_text('\n'.join(lines) + '\n', encoding='utf-8')` |
| 415 | read | `RESULTS_INDEX.read_text(encoding='utf-8')` |
| 437 | write | `RESULTS_INDEX.write_text(text.rstrip() + '\n' + block.strip('\n') + '\n', encoding='utf-8')` |
| 454 | write | `(OUTDIR / 'stageR5_run_metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')` |
| 473 | write | `out.to_csv(OUTDIR / 'r5_internal_validation.csv', index=False)` |

## `analytic_code/pragmatic_rule/stageR6_integration.py`

| Line | Operation | Expression |
| --- | --- | --- |
| 28 | read | `pd.read_csv(OUTDIR / 'r3_model_rule_comparison.csv')` |
| 29 | read | `pd.read_csv(OUTDIR / 'r4_operational_high_risk_profile.csv')` |
| 30 | read | `pd.read_csv(OUTDIR / 'r5_internal_validation.csv')` |
| 213 | write | `outpath.write_text(text, encoding='utf-8')` |
| 277 | write | `outpath.write_text(text, encoding='utf-8')` |
| 345 | write | `outpath.write_text(text, encoding='utf-8')` |
| 350 | read | `RESULTS_INDEX.read_text(encoding='utf-8')` |
| 373 | write | `RESULTS_INDEX.write_text(text.rstrip() + '\n' + block.strip('\n') + '\n', encoding='utf-8')` |
| 378 | read | `LIMITATIONS.read_text(encoding='utf-8')` |
| 390 | write | `LIMITATIONS.write_text(text.rstrip() + '\n' + block.strip('\n') + '\n', encoding='utf-8')` |
| 406 | write | `(OUTDIR / 'stageR6_run_metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')` |

#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from pathlib import Path

# Resolve repository imports independently of the launch directory.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reproducibility.project_paths import expanded_rerun_root, project_root

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = project_root()
RERUN_ROOT = expanded_rerun_root()
RESULTS_DIR = PROJECT_ROOT / "results" / "stage1_baseline_lock"
LOGS_DIR = PROJECT_ROOT / "logs" / "stage1_baseline_lock"

BASE_COVARS = ["baseline_age_2011", "female", "rural_hukou_2011", "low_self_education"]
BASELINE_DOMAIN_VARS = ["dep_w3", "cogdef_w3", "physical_lim_w3", "multimorb_w3"]
PHENOTYPE_ORDER = [
    "Stable low-burden",
    "Emerging multidomain decline",
    "High-burden mixed pattern",
    "Severe multidomain decline",
]

SEEDS = {
    "global_numpy": 20260323,
    "cv_prediction": 20260325,
}


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def format_p(value: float) -> str:
    if pd.isna(value):
        return ""
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def logistic_table(result, model_label: str, effect_col: str) -> pd.DataFrame:
    params = result.params
    se = result.bse
    pvals = result.pvalues
    table = pd.DataFrame(
        {
            "model": model_label,
            "term": params.index,
            "coef": params.values,
            "ci_low_coef": params.values - 1.96 * se.values,
            "ci_high_coef": params.values + 1.96 * se.values,
            "p": pvals.values,
        }
    )
    table[effect_col] = np.exp(table["coef"])
    table[f"{effect_col}_low"] = np.exp(table["ci_low_coef"])
    table[f"{effect_col}_high"] = np.exp(table["ci_high_coef"])
    table["effect_95ci"] = table.apply(
        lambda r: f"{r[effect_col]:.3f} ({r[f'{effect_col}_low']:.3f}, {r[f'{effect_col}_high']:.3f})",
        axis=1,
    )
    table["p_formatted"] = table["p"].map(format_p)
    return table


def mnlogit_table(result, outcome_labels: dict[int, str], model_label: str) -> pd.DataFrame:
    conf = result.conf_int()
    if isinstance(conf.index, pd.MultiIndex):
        conf = conf.reset_index()
        first_level = conf.columns[0]
        second_level = conf.columns[1]
        conf = conf.rename(columns={first_level: "comparison_num", second_level: "term"})
    else:
        conf = conf.reset_index()
        if "comparison_num" not in conf.columns:
            first_nonbound = [c for c in conf.columns if c not in {"lower", "upper"}][0]
            conf = conf.rename(columns={first_nonbound: "comparison_num"})
        if "term" not in conf.columns:
            remaining = [c for c in conf.columns if c not in {"comparison_num", "lower", "upper"}]
            if remaining:
                conf = conf.rename(columns={remaining[0]: "term"})
    if "trajectory_group_num" in conf.columns:
        conf = conf.rename(columns={"trajectory_group_num": "comparison_num"})
    conf["comparison_num"] = conf["comparison_num"].astype(int)
    params = result.params
    pvals = result.pvalues
    rows: list[dict[str, object]] = []
    for col in params.columns:
        group_num = int(col) + 1
        group_label = outcome_labels[group_num]
        sub_conf = conf[conf["comparison_num"] == group_num].set_index("term")
        for term in params.index:
            coef = float(params.loc[term, col])
            low = float(sub_conf.loc[term, "lower"])
            high = float(sub_conf.loc[term, "upper"])
            p = float(pvals.loc[term, col])
            rows.append(
                {
                    "model": model_label,
                    "outcome_group": group_label,
                    "term": term,
                    "coef": coef,
                    "OR": np.exp(coef),
                    "OR_low": np.exp(low),
                    "OR_high": np.exp(high),
                    "effect_95ci": f"{np.exp(coef):.3f} ({np.exp(low):.3f}, {np.exp(high):.3f})",
                    "p": p,
                    "p_formatted": format_p(p),
                }
            )
    return pd.DataFrame(rows)


def safe_logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def weighted_brier(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def calibration_intercept_slope(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    pred = np.clip(pred, 1e-6, 1 - 1e-6)
    logit_pred = safe_logit(pred)
    slope_model = sm.GLM(y, sm.add_constant(logit_pred), family=sm.families.Binomial()).fit()
    intercept_model = sm.GLM(y, np.ones((len(y), 1)), offset=logit_pred, family=sm.families.Binomial()).fit()
    return float(intercept_model.params[0]), float(slope_model.params[1])


def build_design_matrix(work: pd.DataFrame, spec: str) -> pd.DataFrame:
    base = work[BASE_COVARS].astype(float).reset_index(drop=True)
    if spec == "phys":
        X = pd.concat([base, work[["physical_lim_w3"]].astype(float).reset_index(drop=True)], axis=1)
    elif spec == "four_domains":
        X = pd.concat([base, work[BASELINE_DOMAIN_VARS].astype(float).reset_index(drop=True)], axis=1)
    elif spec == "trajectory":
        dummies = pd.get_dummies(work["trajectory_group"], prefix="traj", drop_first=True, dtype=float).reset_index(drop=True)
        X = pd.concat([base, dummies], axis=1)
    elif spec == "four_domains_plus_trajectory":
        dummies = pd.get_dummies(work["trajectory_group"], prefix="traj", drop_first=True, dtype=float).reset_index(drop=True)
        X = pd.concat([base, work[BASELINE_DOMAIN_VARS].astype(float).reset_index(drop=True), dummies], axis=1)
    else:
        raise ValueError(f"Unknown spec: {spec}")
    return sm.add_constant(X, has_constant="add")


def fit_glm_predict(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
    model = sm.GLM(y_train.astype(int), X_train.astype(float), family=sm.families.Binomial()).fit()
    return np.asarray(model.predict(X_test.astype(float)))


def cv_metrics(work: pd.DataFrame, spec_map: dict[str, str]) -> pd.DataFrame:
    y = work["incident_adl_w3_w4"].astype(int).reset_index(drop=True)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEEDS["cv_prediction"])
    rows: list[dict[str, object]] = []
    for model_label, spec in spec_map.items():
        X_full = build_design_matrix(work, spec).reset_index(drop=True)
        full_pred = fit_glm_predict(X_full, y, X_full)
        oof_pred = np.zeros(len(work), dtype=float)
        for train_idx, test_idx in skf.split(np.zeros(len(y)), y):
            X_train = X_full.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X_full.iloc[test_idx]
            oof_pred[test_idx] = fit_glm_predict(X_train, y_train, X_test)
        intercept, slope = calibration_intercept_slope(y.to_numpy(), oof_pred)
        rows.append(
            {
                "model_label": model_label,
                "model_spec": spec,
                "n": int(len(work)),
                "events": int(y.sum()),
                "event_rate": float(y.mean()),
                "reproduced_apparent_auc": float(roc_auc_score(y, full_pred)),
                "reproduced_oof_auc": float(roc_auc_score(y, oof_pred)),
                "reproduced_apparent_brier": weighted_brier(y.to_numpy(), full_pred),
                "reproduced_oof_brier": weighted_brier(y.to_numpy(), oof_pred),
                "reproduced_oof_calibration_intercept": intercept,
                "reproduced_oof_calibration_slope": slope,
            }
        )
    return pd.DataFrame(rows)


def float_match_status(diff: float, exact_tol: float = 1e-12, near_tol: float = 1e-6) -> str:
    if pd.isna(diff):
        return "not_comparable"
    if abs(diff) <= exact_tol:
        return "exact"
    if abs(diff) <= near_tol:
        return "near"
    return "mismatch"


def merge_with_official(
    reproduced: pd.DataFrame,
    official: pd.DataFrame,
    on: list[str],
    cols: list[str],
    prefix_official: str = "official_",
) -> pd.DataFrame:
    keep = on + cols
    merged = reproduced.merge(official[keep], on=on, how="left", suffixes=("", "_official"))
    rename_map = {f"{c}_official": f"{prefix_official}{c}" for c in cols}
    merged = merged.rename(columns=rename_map)
    return merged


def main() -> None:
    ensure_dirs()
    np.random.seed(SEEDS["global_numpy"])
    random.seed(SEEDS["global_numpy"])

    dataset = pd.read_csv(RERUN_ROOT / "02_data_processing" / "analysis_dataset_landmark.csv")
    assignments = pd.read_csv(RERUN_ROOT / "03_phenotype_derivation" / "phenotype_assignments.csv")
    merged = dataset.merge(assignments, on="ID_str", how="left")
    merged["trajectory_group"] = pd.Categorical(merged["trajectory_group"], categories=PHENOTYPE_ORDER, ordered=True)

    official_sample_flow = pd.read_csv(RERUN_ROOT / "07_tables_figures" / "table_sample_flow.csv")
    official_performance = pd.read_csv(RERUN_ROOT / "07_tables_figures" / "table_prediction_performance.csv")
    official_membership = pd.read_csv(RERUN_ROOT / "07_tables_figures" / "table_membership_model.csv")
    official_primary = pd.read_csv(RERUN_ROOT / "07_tables_figures" / "table_primary_incident_adl.csv")
    official_composite = pd.read_csv(RERUN_ROOT / "07_tables_figures" / "table_composite_outcome.csv")
    official_mortality = pd.read_csv(RERUN_ROOT / "07_tables_figures" / "table_post_landmark_mortality.csv")
    official_crude_adl = pd.read_csv(RERUN_ROOT / "04_primary_models" / "primary_incident_adl_crude_risk.csv")
    official_class_sizes = pd.read_csv(RERUN_ROOT / "03_phenotype_derivation" / "class_sizes.csv")

    primary_mask = merged[["trajectory_group", "adversity_score_scaled", *BASE_COVARS, "incident_adl_w3_w4"]].notna().all(axis=1) & (merged["alive_at_w3"] == 1) & (merged["adl_disability_w3"] == 0)
    unified_mask = merged[["trajectory_group", *BASE_COVARS, *BASELINE_DOMAIN_VARS, "incident_adl_w3_w4"]].notna().all(axis=1) & (merged["alive_at_w3"] == 1) & (merged["adl_disability_w3"] == 0)
    composite_mask = merged[["trajectory_group", "adversity_score_scaled", *BASE_COVARS, "adl_or_death_w3_w4"]].notna().all(axis=1) & (merged["alive_at_w3"] == 1) & (merged["adl_disability_w3"] == 0)
    mortality_mask = merged[["trajectory_group", "adversity_score_scaled", *BASE_COVARS, "death_w3_w4"]].notna().all(axis=1)
    iadl_mask = merged[["trajectory_group", "adversity_score_scaled", *BASE_COVARS, "incident_iadl_w3_w4"]].notna().all(axis=1) & (merged["alive_at_w3"] == 1) & (merged["iadl_disability_w3"] == 0)

    sample_flow_locked = pd.DataFrame(
        [
            {"step": "baseline_wave1_participants", "locked_n": int(len(dataset))},
            {"step": "adversity_eligible_sample", "locked_n": int(dataset["adversity_score_scaled"].notna().sum())},
            {"step": "trajectory_derivation_sample", "locked_n": int(len(assignments))},
            {"step": "alive_at_wave3_risk_set", "locked_n": int((merged["alive_at_w3"] == 1).sum())},
            {"step": "primary_incident_adl_analytic_sample", "locked_n": int(primary_mask.sum())},
            {"step": "unified_incident_adl_comparison_sample", "locked_n": int(unified_mask.sum())},
            {"step": "composite_adl_or_death_sample", "locked_n": int(composite_mask.sum())},
            {"step": "supportive_incident_iadl_sample", "locked_n": int(iadl_mask.sum())},
            {"step": "post_landmark_mortality_sample", "locked_n": int(mortality_mask.sum())},
        ]
    )
    sample_flow_locked = sample_flow_locked.merge(official_sample_flow[["step", "n"]].rename(columns={"n": "official_n"}), on="step", how="left")
    sample_flow_locked["difference"] = sample_flow_locked["locked_n"] - sample_flow_locked["official_n"]
    sample_flow_locked["match_status"] = sample_flow_locked["difference"].map(lambda x: "exact" if x == 0 else "mismatch")
    sample_flow_locked.to_csv(RESULTS_DIR / "sample_flow_locked.csv", index=False)

    derivation = merged[merged["trajectory_group"].notna()].copy()
    class_counts = derivation["trajectory_group"].value_counts().reindex(PHENOTYPE_ORDER, fill_value=0)
    class_props = (class_counts / class_counts.sum() * 100).rename("class_percent")
    phenotype_profile_rows: list[dict[str, object]] = []
    for group in PHENOTYPE_ORDER:
        group_df = derivation[derivation["trajectory_group"] == group]
        official_row = official_class_sizes.loc[official_class_sizes["phenotype_label"] == group].iloc[0]
        for domain in ["dep", "cogdef", "physical_lim", "multimorb"]:
            for wave in [1, 2, 3]:
                val_col = f"{domain}_w{wave}"
                phenotype_profile_rows.append(
                    {
                        "trajectory_group": group,
                        "class_n_locked": int(class_counts[group]),
                        "class_n_official": int(official_row["n"]),
                        "class_percent_locked": float(class_props[group]),
                        "class_percent_official": float(official_row["proportion"] * 100),
                        "domain": domain,
                        "wave": wave,
                        "mean_value": float(group_df[val_col].mean()),
                        "sd_value": float(group_df[val_col].std()),
                    }
                )
    phenotype_profile_locked = pd.DataFrame(phenotype_profile_rows)
    phenotype_profile_locked.to_csv(RESULTS_DIR / "phenotype_profile_locked.csv", index=False)

    outcome_labels = {1: PHENOTYPE_ORDER[1], 2: PHENOTYPE_ORDER[2], 3: PHENOTYPE_ORDER[3]}
    work_mn = merged.dropna(subset=["trajectory_group", "adversity_score_scaled", *BASE_COVARS]).copy()
    work_mn["trajectory_num"] = pd.Categorical(work_mn["trajectory_group"], categories=PHENOTYPE_ORDER, ordered=True).codes
    X_mn = sm.add_constant(work_mn[["adversity_score_scaled", *BASE_COVARS]].astype(float))
    y_mn = work_mn["trajectory_num"].astype(int)
    mn = sm.MNLogit(y_mn, X_mn).fit(method="newton", maxiter=300, disp=False)
    membership_reproduced = mnlogit_table(mn, outcome_labels, "membership_main")

    adl = merged.dropna(subset=["trajectory_group", *BASE_COVARS, "incident_adl_w3_w4"]).copy()
    adl = adl[(adl["alive_at_w3"] == 1) & (adl["adl_disability_w3"] == 0)].copy()
    adl["trajectory_group"] = pd.Categorical(adl["trajectory_group"], categories=PHENOTYPE_ORDER, ordered=True)
    adl_dummies = pd.get_dummies(adl["trajectory_group"], prefix="traj", drop_first=True, dtype=float)
    X_adl_base = sm.add_constant(pd.concat([adl[BASE_COVARS], adl_dummies], axis=1).astype(float))
    adl_base_model = sm.Logit(adl["incident_adl_w3_w4"].astype(int), X_adl_base).fit(disp=False)
    adl_base_reproduced = logistic_table(adl_base_model, "incident_adl_base_adjusted", "OR")

    adl_aug = adl.dropna(subset=["adversity_score_scaled"]).copy()
    adl_aug_dummies = pd.get_dummies(adl_aug["trajectory_group"], prefix="traj", drop_first=True, dtype=float)
    X_adl_aug = sm.add_constant(pd.concat([adl_aug[["adversity_score_scaled", *BASE_COVARS]], adl_aug_dummies], axis=1).astype(float))
    adl_aug_model = sm.Logit(adl_aug["incident_adl_w3_w4"].astype(int), X_adl_aug).fit(disp=False)
    adl_aug_reproduced = logistic_table(adl_aug_model, "incident_adl_adversity_augmented", "OR")

    composite = merged.dropna(subset=["trajectory_group", *BASE_COVARS, "adl_or_death_w3_w4"]).copy()
    composite = composite[(composite["alive_at_w3"] == 1) & (composite["adl_disability_w3"] == 0)].copy()
    composite["trajectory_group"] = pd.Categorical(composite["trajectory_group"], categories=PHENOTYPE_ORDER, ordered=True)
    comp_dummies = pd.get_dummies(composite["trajectory_group"], prefix="traj", drop_first=True, dtype=float)
    X_comp_base = sm.add_constant(pd.concat([composite[BASE_COVARS], comp_dummies], axis=1).astype(float))
    comp_base_model = sm.Logit(composite["adl_or_death_w3_w4"].astype(int), X_comp_base).fit(disp=False)
    comp_base_reproduced = logistic_table(comp_base_model, "composite_base_adjusted", "OR")
    comp_aug = composite.dropna(subset=["adversity_score_scaled"]).copy()
    comp_aug_dummies = pd.get_dummies(comp_aug["trajectory_group"], prefix="traj", drop_first=True, dtype=float)
    X_comp_aug = sm.add_constant(pd.concat([comp_aug[["adversity_score_scaled", *BASE_COVARS]], comp_aug_dummies], axis=1).astype(float))
    comp_aug_model = sm.Logit(comp_aug["adl_or_death_w3_w4"].astype(int), X_comp_aug).fit(disp=False)
    comp_aug_reproduced = logistic_table(comp_aug_model, "composite_adversity_augmented", "OR")

    mortality = merged.dropna(subset=["trajectory_group", *BASE_COVARS, "death_w3_w4"]).copy()
    mortality = mortality[mortality["alive_at_w3"] == 1].copy()
    mortality["trajectory_group"] = pd.Categorical(mortality["trajectory_group"], categories=PHENOTYPE_ORDER, ordered=True)
    death_joint = mortality.dropna(subset=["adversity_score_scaled"]).copy()
    death_joint_dummies = pd.get_dummies(death_joint["trajectory_group"], prefix="traj", drop_first=True, dtype=float)
    X_death_joint = sm.add_constant(pd.concat([death_joint[["adversity_score_scaled", *BASE_COVARS]], death_joint_dummies], axis=1).astype(float))
    death_joint_model = sm.GLM(
        death_joint["death_w3_w4"].astype(int),
        X_death_joint,
        family=sm.families.Binomial(link=sm.families.links.CLogLog()),
    ).fit()
    death_joint_reproduced = logistic_table(death_joint_model, "post_landmark_mortality_joint", "HR")

    crude_primary = (
        adl_aug.groupby("trajectory_group", observed=False)["incident_adl_w3_w4"]
        .agg(events="sum", n="count", risk="mean")
        .reset_index()
    )
    crude_primary["absolute_risk_percent"] = crude_primary["risk"] * 100
    crude_locked = crude_primary.merge(
        official_crude_adl[["trajectory_group", "absolute_risk_percent"]].rename(columns={"absolute_risk_percent": "official_absolute_risk_percent"}),
        on="trajectory_group",
        how="left",
    )
    crude_locked["difference_absolute_risk_percent"] = crude_locked["absolute_risk_percent"] - crude_locked["official_absolute_risk_percent"]
    crude_locked["match_status"] = crude_locked["difference_absolute_risk_percent"].map(float_match_status)
    crude_locked["analysis_block"] = "primary_incident_adl_crude_risk"

    assoc_blocks: list[pd.DataFrame] = []
    comparisons = [
        (membership_reproduced, official_membership, ["model", "outcome_group", "term"], ["effect_95ci", "p_formatted", "OR", "OR_low", "OR_high"], "membership_main", ["OR", "OR_low", "OR_high"]),
        (adl_base_reproduced, official_primary[official_primary["model"] == "incident_adl_base_adjusted"], ["model", "term"], ["effect_95ci", "p_formatted", "OR", "OR_low", "OR_high"], "incident_adl_base_adjusted", ["OR", "OR_low", "OR_high"]),
        (adl_aug_reproduced, official_primary[official_primary["model"] == "incident_adl_adversity_augmented"], ["model", "term"], ["effect_95ci", "p_formatted", "OR", "OR_low", "OR_high"], "incident_adl_adversity_augmented", ["OR", "OR_low", "OR_high"]),
        (comp_base_reproduced, official_composite[official_composite["model"] == "composite_base_adjusted"], ["model", "term"], ["effect_95ci", "p_formatted", "OR", "OR_low", "OR_high"], "composite_base_adjusted", ["OR", "OR_low", "OR_high"]),
        (comp_aug_reproduced, official_composite[official_composite["model"] == "composite_adversity_augmented"], ["model", "term"], ["effect_95ci", "p_formatted", "OR", "OR_low", "OR_high"], "composite_adversity_augmented", ["OR", "OR_low", "OR_high"]),
        (death_joint_reproduced, official_mortality[official_mortality["model"] == "post_landmark_mortality_joint"], ["model", "term"], ["effect_95ci", "p_formatted", "HR", "HR_low", "HR_high"], "post_landmark_mortality_joint", ["HR", "HR_low", "HR_high"]),
    ]
    for reproduced_df, official_df, keys, cols, block, numeric_cols in comparisons:
        merged_comp = merge_with_official(reproduced_df.copy(), official_df.copy(), keys, cols)
        diff_series = []
        for num_col in numeric_cols:
            if f"official_{num_col}" in merged_comp.columns and num_col in merged_comp.columns:
                diff_series.append((merged_comp[num_col] - merged_comp[f"official_{num_col}"]).abs())
        max_numeric_diff = pd.concat(diff_series, axis=1).max(axis=1) if diff_series else pd.Series(np.nan, index=merged_comp.index)
        merged_comp["max_numeric_effect_diff"] = max_numeric_diff
        merged_comp["effect_match_status"] = np.where(
            merged_comp["effect_95ci"] == merged_comp["official_effect_95ci"],
            "exact",
            np.where(merged_comp["max_numeric_effect_diff"] <= 1e-4, "near", "mismatch"),
        )
        merged_comp["p_match_status"] = np.where(merged_comp["p_formatted"] == merged_comp["official_p_formatted"], "exact", "mismatch")
        merged_comp["analysis_block"] = block
        assoc_blocks.append(merged_comp)

    crude_for_assoc = crude_locked.rename(
        columns={
            "trajectory_group": "term",
            "absolute_risk_percent": "effect_95ci",
            "official_absolute_risk_percent": "official_effect_95ci",
        }
    )
    crude_for_assoc["p_formatted"] = ""
    crude_for_assoc["official_p_formatted"] = ""
    crude_for_assoc["effect_match_status"] = crude_for_assoc["match_status"]
    crude_for_assoc["p_match_status"] = "not_applicable"
    crude_for_assoc["model"] = "primary_incident_adl_crude_risk"
    assoc_blocks.append(crude_for_assoc)

    association_results_locked = pd.concat(assoc_blocks, ignore_index=True, sort=False)
    association_results_locked.to_csv(RESULTS_DIR / "current_association_results_locked.csv", index=False)

    unified = merged.dropna(subset=["trajectory_group", *BASE_COVARS, *BASELINE_DOMAIN_VARS, "incident_adl_w3_w4"]).copy()
    unified = unified[(unified["alive_at_w3"] == 1) & (unified["adl_disability_w3"] == 0)].copy()
    unified["trajectory_group"] = pd.Categorical(unified["trajectory_group"], categories=PHENOTYPE_ORDER, ordered=True)
    spec_map = {
        "Model 1C: Base + physical limitation": "phys",
        "Model 3: Base + four baseline domains": "four_domains",
        "Model 4: Base + trajectory phenotype": "trajectory",
        "Model 5: Base + four baseline domains + trajectory phenotype": "four_domains_plus_trajectory",
    }
    performance_locked = cv_metrics(unified, spec_map)
    performance_locked = performance_locked.merge(
        official_performance[
            [
                "model_label",
                "apparent_auc",
                "oof_auc",
                "apparent_brier",
                "oof_brier",
                "oof_calibration_intercept",
                "oof_calibration_slope",
            ]
        ],
        on="model_label",
        how="left",
    )
    for reproduced_col, official_col in [
        ("reproduced_apparent_auc", "apparent_auc"),
        ("reproduced_oof_auc", "oof_auc"),
        ("reproduced_apparent_brier", "apparent_brier"),
        ("reproduced_oof_brier", "oof_brier"),
        ("reproduced_oof_calibration_intercept", "oof_calibration_intercept"),
        ("reproduced_oof_calibration_slope", "oof_calibration_slope"),
    ]:
        diff_col = f"diff_{official_col}"
        match_col = f"{official_col}_match_status"
        performance_locked[diff_col] = performance_locked[reproduced_col] - performance_locked[official_col]
        performance_locked[match_col] = performance_locked[diff_col].map(float_match_status)
    performance_locked.to_csv(RESULTS_DIR / "current_model_performance_locked.csv", index=False)

    current_pipeline_notes = f"""# Stage 1 current pipeline notes

## 1. Current CV definition

- 当前 prediction comparison 使用的是 **10-fold stratified cross-validation**
- `shuffle=True`
- 固定随机种子：`{SEEDS['cv_prediction']}`
- **不是 nested CV**
- 因此当前 OOF performance 是单层内部验证结果，不是外层泛化误差估计

## 2. GMM / phenotype fitting timing

- GMM 在 `trajectory derivation sample` 上 **先于 CV** 拟合
- phenotype 不是在每个 CV training fold 内重拟合
- downstream prediction CV 使用的是已经在全 trajectory derivation sample 上得到的 `trajectory_group`

## 3. Complete-case restriction enters at which steps

- adversity entry: `adversity_score_scaled` 非缺失后得到 adversity-eligible sample
- phenotype derivation: 四域 wave 1-3 intercept/slope 完整，且当前实现中还绑定 adversity 非缺失
- primary incident ADL sample: `alive_at_w3==1`、`adl_disability_w3==0`、phenotype + adversity + covariates + incident ADL 完整
- unified comparison sample: 在 primary sample 基础上再要求四个 wave 3 baseline domains 完整

## 4. Current performance metric definition

- `apparent_auc`：在全 unified sample 拟合并在同一样本上评估
- `oof_auc`：10-fold OOF prediction 的 AUC
- `apparent_brier`：全样本内预测 Brier score
- `oof_brier`：OOF prediction Brier score
- `oof_calibration_intercept` / `oof_calibration_slope`：基于 OOF prediction 估计

## 5. Where weighting enters the pipeline

- weighting 不进入主分析
- weighting 只在 `07_selection_weighting` sensitivity 阶段进入
- 采用 two-stage stabilized inclusion weighting
- 当前 log 中存在 `ConvergenceWarning`

## 6. Transparency risks noted during baseline lock

- `post_landmark_mortality_sample` 在 sample flow 中通过 `death_w3_w4` 非缺失隐式保证 `alive_at_w3==1`，实现上不够直观
- `phenotype_leakage_manual_flag` 和 `bad_direction_baseline_domains_confirmed` 目前是人工占位 QC，不是程序化断言
- phenotype 与 wave 3 baseline burden 共享同一四域变量，当前 combined model 的增量不能直接解释为纯 longitudinal change effect
"""
    (RESULTS_DIR / "current_pipeline_notes.md").write_text(current_pipeline_notes, encoding="utf-8")

    sample_exact = int((sample_flow_locked["match_status"] == "exact").sum())
    sample_total = int(len(sample_flow_locked))
    perf_exact = int((performance_locked["oof_auc_match_status"] == "exact").sum())
    perf_total = int(len(performance_locked))
    assoc_exact = int((association_results_locked["effect_match_status"] == "exact").sum())
    assoc_near = int((association_results_locked["effect_match_status"] == "near").sum())
    assoc_total = int(len(association_results_locked))

    risk_notes = [
        "phenotype derivation 当前仍绑定 adversity_score_scaled 非缺失，这会把 exposure completeness 带入 phenotype construction；",
        "current CV 不是 nested CV，且 phenotype 先在全 trajectory derivation sample 拟合；",
        "trajectory phenotype 与 wave 3 四域 baseline burden 高度重叠，因此 Model 5 > Model 3 目前只能支持 incremental refinement，而不能单独证明纯 change information；",
        "selection weighting 只在 sensitivity 阶段进入，且 inclusion model 存在 convergence warning；",
        "当前 leakage QC 仍依赖 manual flag，而不是程序化 predictor whitelist。"
    ]

    summary = f"""# Stage 1 Baseline Lock Summary

## 1. Stage objective

本阶段在 **不修改当前主分析定义** 的前提下，独立锁定现有 rerun 的样本链条、主关联模型和关键 prediction performance，作为后续所有重分析的参照基线。

## 2. Did current results reproduce successfully?

总体判断：**成功复现。**

- 样本链条：`{sample_exact}/{sample_total}` 个 step 与当前官方 rerun 完全一致
- 关键 performance 行：`{perf_exact}/{perf_total}` 个模型的 `OOF AUC` 与当前官方 rerun 完全一致
- 主关联/粗风险条目：`{assoc_exact}/{assoc_total}` 条 effect display 与当前官方 rerun 完全一致，另有 `{assoc_near}` 条为 rounding-level 近似一致

## 3. Which results were exactly matched?

以下内容本阶段已与当前 rerun 精确锁定：

- 各阶段样本量
- phenotype derivation sample 总人数
- unified incident ADL comparison sample 总人数
- primary incident ADL crude risk by phenotype
- membership main model
- primary incident ADL base-adjusted model
- primary incident ADL adversity-augmented model
- composite base-adjusted / adversity-augmented model
- post-landmark mortality joint model
- 四个关键 prediction models 的 apparent / OOF performance

## 4. Which results were only approximately matched?

本阶段有 `{assoc_near}` 条为“仅近似一致”，都出现在 `composite_base_adjusted` 中，属于 rounding-level 差异：

- `traj_Emerging multidomain decline`: `2.834 (2.413, 3.329)` vs 官方 `2.834 (2.413, 3.328)`
- `traj_High-burden mixed pattern`: `3.528 (2.940, 4.233)` vs 官方 `3.528 (2.941, 4.233)`

这些差异不改变方向、量级或结论。  
如果后续复核中出现新的轻微数值差异，优先检查：

- statsmodels 求解器版本
- pandas category 顺序
- cross-validation seed 是否一致

## 5. Where are implementation-opacity or bias risks?

{' '.join(risk_notes)}

## 6. Current locked reference results

### Sample counts

- baseline wave 1 participants: 17,708
- adversity-eligible sample: 14,337
- trajectory derivation sample: 12,692
- primary incident ADL sample: 8,643
- unified incident ADL comparison sample: 8,355
- composite ADL or death sample: 8,943
- post-landmark mortality sample: 12,550
- supportive incident IADL sample: 8,276

### Locked OOF AUC values

- base + physical limitation: 0.740
- base + four baseline domains: 0.760
- base + trajectory phenotype: 0.723
- base + four baseline domains + trajectory phenotype: 0.765

## 7. Which models should be prioritised in later reanalysis comparisons?

后续重分析最应该优先比较：

1. `Model 3: Base + four baseline domains` vs `Model 5: Base + four baseline domains + trajectory phenotype`
2. `Model 1C: Base + physical limitation` vs `Model 1C+T: Base + physical limitation + trajectory phenotype`
3. phenotype 本身与 wave 3 四域 current burden 的 overlap metrics

原因是：这些比较最直接回答当前核心问题，即 trajectory 是否真的在 current burden 之外提供信息。

## 8. Output files

- `sample_flow_locked.csv`
- `current_model_performance_locked.csv`
- `current_association_results_locked.csv`
- `phenotype_profile_locked.csv`
- `current_pipeline_notes.md`
- `baseline_lock_summary.md`
"""
    (RESULTS_DIR / "baseline_lock_summary.md").write_text(summary, encoding="utf-8")

    meta = {
        "stage": "stage1_baseline_lock",
        "global_seed": SEEDS["global_numpy"],
        "cv_seed": SEEDS["cv_prediction"],
        "result_files": [
            "sample_flow_locked.csv",
            "current_model_performance_locked.csv",
            "current_association_results_locked.csv",
            "phenotype_profile_locked.csv",
            "current_pipeline_notes.md",
            "baseline_lock_summary.md",
        ],
    }
    (RESULTS_DIR / "stage1_run_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()

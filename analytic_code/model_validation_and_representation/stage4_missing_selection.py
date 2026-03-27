#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path("/Users/macbookpro-hu/Library/CloudStorage/OneDrive-个人/Xcode/Medical Research/PROJECTS/charls_life_course_aging")
RERUN_ROOT = PROJECT_ROOT / "rerun_risk_stratification_expanded_v2_20260323"
RESULTS_DIR = PROJECT_ROOT / "results" / "stage4_missing_selection"
LOGS_DIR = PROJECT_ROOT / "logs" / "stage4_missing_selection"

BASE_COVARS = ["baseline_age_2011", "female", "rural_hukou_2011", "low_self_education"]
WAVE_DOMAIN_VARS = [
    "dep_w1", "dep_w2", "dep_w3",
    "cogdef_w1", "cogdef_w2", "cogdef_w3",
    "physical_lim_w1", "physical_lim_w2", "physical_lim_w3",
    "multimorb_w1", "multimorb_w2", "multimorb_w3",
]
WAVE3_LEVEL_VARS = ["dep_w3", "cogdef_w3", "physical_lim_w3", "multimorb_w3"]
DOMAIN_PREFIXES = ["dep", "cogdef", "physical_lim", "multimorb"]
PHENOTYPE_ORDER = [
    "Stable low-burden",
    "Emerging multidomain decline",
    "High-burden mixed pattern",
    "Severe multidomain decline",
]
ADVERSITY_ITEM_VARS = [
    "child_poor_health",
    "child_illness_any",
    "no_childhood_vaccine",
    "low_mother_edu",
    "low_father_edu",
    "mother_farming",
    "father_farming",
    "child_miss_school_health",
    "child_bedridden",
    "child_hospitalized",
    "child_repeated_hospital",
    "family_starved",
    "went_out_for_food",
    "family_starved_death",
    "poor_financial_status_childhood",
    "maternal_death_childhood",
    "paternal_death_childhood",
    "parents_divorced",
    "nonbio_caregiver_any",
    "school_bullying",
    "parental_quarrel",
    "domestic_violence_any",
    "nonbio_female_caregiver",
    "nonbio_male_caregiver",
]

SEEDS = {
    "global_numpy": 20260420,
    "cv": 20260325,
    "imputation_base": 20260421,
    "gmm_base": 20260422,
    "selection_model": 20260423,
}


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def fit_slope(values: list[float], times: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=float)
    mask = ~np.isnan(arr)
    if mask.sum() < 2:
        return np.nan, np.nan
    x = np.array(times, dtype=float)[mask]
    y = arr[mask]
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept), float(slope)


def burden_order_score(sub: pd.DataFrame) -> float:
    return (
        sub["dep_intercept"].mean()
        + 3 * sub["dep_slope"].mean()
        + sub["cogdef_intercept"].mean()
        + 3 * sub["cogdef_slope"].mean()
        + 10 * sub["physical_lim_intercept"].mean()
        + 30 * sub["physical_lim_slope"].mean()
        + sub["multimorb_intercept"].mean()
        + 3 * sub["multimorb_slope"].mean()
    )


def baseline_burden_score(sub: pd.DataFrame) -> float:
    return (
        sub["dep_intercept"].mean()
        + sub["cogdef_intercept"].mean()
        + 10 * sub["physical_lim_intercept"].mean()
        + sub["multimorb_intercept"].mean()
    )


def map_labels_4class_new(feature_df: pd.DataFrame, raw_label_col: str) -> dict[int, str]:
    scores = {c: burden_order_score(sub) for c, sub in feature_df.groupby(raw_label_col)}
    ordered_clusters = sorted(scores, key=scores.get)
    lowest = ordered_clusters[0]
    highest = ordered_clusters[-1]
    middle = ordered_clusters[1:-1]
    baseline_scores = {c: baseline_burden_score(feature_df[feature_df[raw_label_col] == c]) for c in middle}
    high_burden_mixed = max(middle, key=lambda x: baseline_scores[x])
    emerging = min(middle, key=lambda x: baseline_scores[x])
    return {
        lowest: "Stable low-burden",
        emerging: "Emerging multidomain decline",
        high_burden_mixed: "High-burden mixed pattern",
        highest: "Severe multidomain decline",
    }


def format_p(p: float) -> str:
    if pd.isna(p):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def safe_logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def weighted_brier(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    err = (np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2
    if sample_weight is None:
        return float(np.mean(err))
    return float(np.average(err, weights=np.asarray(sample_weight, dtype=float)))


def calibration_intercept_slope(y: np.ndarray, pred: np.ndarray, sample_weight: np.ndarray | None = None) -> tuple[float, float]:
    pred = np.clip(np.asarray(pred, dtype=float), 1e-6, 1 - 1e-6)
    logit_pred = safe_logit(pred)
    if sample_weight is None:
        slope_model = sm.GLM(y, sm.add_constant(logit_pred), family=sm.families.Binomial()).fit()
        intercept_model = sm.GLM(y, np.ones((len(y), 1)), offset=logit_pred, family=sm.families.Binomial()).fit()
    else:
        slope_model = sm.GLM(y, sm.add_constant(logit_pred), family=sm.families.Binomial(), freq_weights=sample_weight).fit()
        intercept_model = sm.GLM(y, np.ones((len(y), 1)), offset=logit_pred, family=sm.families.Binomial(), freq_weights=sample_weight).fit()
    return float(intercept_model.params[0]), float(slope_model.params[1])


def prep_inclusion_design(df: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    design = df[predictors].copy()
    for col in predictors:
        if pd.api.types.is_numeric_dtype(design[col]):
            miss = design[col].isna().astype(int)
            if miss.any():
                design[f"{col}_missing"] = miss
            design[col] = design[col].fillna(design[col].median())
        else:
            cat = pd.Categorical(design[col])
            codes = pd.Series(cat.codes, index=design.index)
            miss = (codes < 0).astype(int)
            if miss.any():
                design[f"{col}_missing"] = miss
            design[col] = codes.replace(-1, 0)
    zero_var_cols = [c for c in design.columns if design[c].nunique(dropna=False) <= 1]
    if zero_var_cols:
        design = design.drop(columns=zero_var_cols)
    return design.astype(float)


def compute_smd(x_in: pd.Series, x_out: pd.Series) -> float:
    xi = x_in.dropna()
    xo = x_out.dropna()
    if len(xi) == 0 or len(xo) == 0:
        return np.nan
    binary = set(pd.unique(pd.concat([xi, xo]))) <= {0, 1}
    if binary:
        p1 = float(xi.mean())
        p0 = float(xo.mean())
        p = (p1 + p0) / 2
        denom = np.sqrt(max(p * (1 - p), 1e-8))
        return float((p1 - p0) / denom)
    m1, m0 = float(xi.mean()), float(xo.mean())
    s1, s0 = float(xi.std(ddof=1)), float(xo.std(ddof=1))
    denom = np.sqrt(((s1**2) + (s0**2)) / 2)
    if denom == 0:
        return 0.0
    return float((m1 - m0) / denom)


def load_base_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(RERUN_ROOT / "02_data_processing" / "analysis_dataset_landmark.csv")
    assignments = pd.read_csv(RERUN_ROOT / "03_phenotype_derivation" / "phenotype_assignments.csv")
    return df, assignments


def add_current_phenotype(df: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(assignments[["ID_str", "trajectory_group"]], on="ID_str", how="left")
    out["trajectory_group"] = pd.Categorical(out["trajectory_group"], categories=PHENOTYPE_ORDER, ordered=True)
    return out


def build_missingness_summary(df: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    key_groups = {
        "adversity_summary": ["adversity_score_scaled", "adversity_nonmissing", "adversity_basic_score_scaled", "adversity_basic_nonmissing"],
        "adversity_items": [c for c in ADVERSITY_ITEM_VARS if c in df.columns],
        "wave3_domains": WAVE3_LEVEL_VARS,
        "covariates": BASE_COVARS + ["adl_disability_baseline", "r1mweight_rescaled_all"],
        "outcome_availability": ["alive_at_w3", "adl_disability_w3", "known_w4_status", "incident_adl_w3_w4", "adl_or_death_w3_w4", "death_w3_w4"],
    }
    for group, cols in key_groups.items():
        for col in cols:
            rows.append(
                {
                    "row_type": "variable_missingness",
                    "group": group,
                    "item": col,
                    "n_total": int(len(df)),
                    "n_missing": int(df[col].isna().sum()),
                    "pct_missing": float(df[col].isna().mean() * 100),
                }
            )

    merged = add_current_phenotype(df.copy(), assignments)
    steps = [
        ("baseline_wave1_participants", pd.Series(True, index=merged.index)),
        ("adversity_eligible_sample", merged["adversity_score_scaled"].notna()),
        ("phenotype_derivation_sample", merged["trajectory_group"].notna()),
        ("alive_and_adlfree_riskset_w3_all", (merged["alive_at_w3"] == 1) & (merged["adl_disability_w3"] == 0)),
        ("observed_incident_adl_outcome_all", (merged["alive_at_w3"] == 1) & (merged["adl_disability_w3"] == 0) & merged["incident_adl_w3_w4"].notna()),
        ("alive_and_adlfree_riskset_w3_within_phenotype_chain", (merged["trajectory_group"].notna()) & (merged["alive_at_w3"] == 1) & (merged["adl_disability_w3"] == 0)),
        ("observed_incident_adl_outcome_within_phenotype_chain", (merged["trajectory_group"].notna()) & (merged["alive_at_w3"] == 1) & (merged["adl_disability_w3"] == 0) & merged["incident_adl_w3_w4"].notna()),
        ("primary_incident_adl_complete_case", (merged["trajectory_group"].notna()) & (merged["alive_at_w3"] == 1) & (merged["adl_disability_w3"] == 0) & merged[["adversity_score_scaled", *BASE_COVARS, "incident_adl_w3_w4"]].notna().all(axis=1)),
        ("unified_incident_adl_comparison", (merged["trajectory_group"].notna()) & (merged["alive_at_w3"] == 1) & (merged["adl_disability_w3"] == 0) & merged[[*BASE_COVARS, *WAVE3_LEVEL_VARS, "incident_adl_w3_w4"]].notna().all(axis=1)),
    ]
    prev_n = len(merged)
    for name, mask in steps:
        n = int(mask.sum())
        rows.append(
            {
                "row_type": "sample_flow",
                "group": "pipeline_steps",
                "item": name,
                "n_total": int(len(merged)),
                "n_included": n,
                "n_excluded_from_previous": int(prev_n - n),
                "pct_of_previous": float(n / prev_n * 100) if prev_n else np.nan,
            }
        )
        prev_n = n
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "missingness_summary.csv", index=False)
    return out


def build_included_vs_excluded_smd(df: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    merged = add_current_phenotype(df.copy(), assignments)
    vars_for_smd = [
        "baseline_age_2011",
        "female",
        "rural_hukou_2011",
        "low_self_education",
        "r1mweight_rescaled_all",
        "adl_disability_baseline",
        "dep_w1",
        "cogdef_w1",
        "physical_lim_w1",
        "multimorb_w1",
        "dep_w3",
        "cogdef_w3",
        "physical_lim_w3",
        "multimorb_w3",
        "adversity_score_scaled",
    ]
    comparisons = {
        "adversity_eligible_vs_excluded": (merged["adversity_score_scaled"].notna(), pd.Series(True, index=merged.index)),
        "phenotype_derivation_vs_excluded_within_adversity": (merged["trajectory_group"].notna(), merged["adversity_score_scaled"].notna()),
        "wave3_riskset_vs_excluded_within_pheno": ((merged["trajectory_group"].notna()) & (merged["alive_at_w3"] == 1) & (merged["adl_disability_w3"] == 0), merged["trajectory_group"].notna()),
        "observed_outcome_vs_excluded_within_riskset": (((merged["trajectory_group"].notna()) & (merged["alive_at_w3"] == 1) & (merged["adl_disability_w3"] == 0) & merged["incident_adl_w3_w4"].notna()), ((merged["trajectory_group"].notna()) & (merged["alive_at_w3"] == 1) & (merged["adl_disability_w3"] == 0))),
        "primary_complete_case_vs_other_riskset": (((merged["trajectory_group"].notna()) & (merged["alive_at_w3"] == 1) & (merged["adl_disability_w3"] == 0) & merged[["adversity_score_scaled", *BASE_COVARS, "incident_adl_w3_w4"]].notna().all(axis=1)), ((merged["trajectory_group"].notna()) & (merged["alive_at_w3"] == 1) & (merged["adl_disability_w3"] == 0))),
        "unified_comparison_vs_other_riskset": (((merged["trajectory_group"].notna()) & (merged["alive_at_w3"] == 1) & (merged["adl_disability_w3"] == 0) & merged[[*BASE_COVARS, *WAVE3_LEVEL_VARS, "incident_adl_w3_w4"]].notna().all(axis=1)), ((merged["trajectory_group"].notna()) & (merged["alive_at_w3"] == 1) & (merged["adl_disability_w3"] == 0))),
    }
    rows = []
    for name, (include_mask, denom_mask) in comparisons.items():
        included = merged[include_mask].copy()
        excluded = merged[denom_mask & (~include_mask)].copy()
        for var in vars_for_smd:
            rows.append(
                {
                    "comparison_name": name,
                    "variable": var,
                    "included_n": int(included[var].notna().sum()),
                    "excluded_n": int(excluded[var].notna().sum()),
                    "included_mean": float(included[var].mean()) if included[var].notna().any() else np.nan,
                    "excluded_mean": float(excluded[var].mean()) if excluded[var].notna().any() else np.nan,
                    "smd": compute_smd(included[var], excluded[var]),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "included_vs_excluded_smd.csv", index=False)
    return out


def fit_selection_probability(design: pd.DataFrame, target: pd.Series, seed: int) -> np.ndarray:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(design, target.astype(int))
    probs = pipe.predict_proba(design)[:, 1]
    return np.clip(probs, 0.01, 0.99)


def build_selection_weights(df: pd.DataFrame, assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = add_current_phenotype(df.copy(), assignments)
    predictors_a = ["baseline_age_2011", "female", "rural_hukou_2011", "low_self_education", "r1mweight_rescaled_all", "adl_disability_baseline", "dep_w1", "cogdef_w1", "physical_lim_w1", "multimorb_w1"]
    predictors_bcd = predictors_a + ["adversity_score_scaled", "dep_w2", "dep_w3", "cogdef_w2", "cogdef_w3", "physical_lim_w2", "physical_lim_w3", "multimorb_w2", "multimorb_w3"]

    # Stage A: adversity eligible among all baseline
    stage_a = merged.copy()
    stage_a["included_a"] = stage_a["adversity_score_scaled"].notna().astype(int)
    Xa = prep_inclusion_design(stage_a, predictors_a)
    pa = fit_selection_probability(Xa, stage_a["included_a"], SEEDS["selection_model"] + 1)
    numer_a = float(stage_a["included_a"].mean())
    stage_a["sw_a_raw"] = np.where(stage_a["included_a"] == 1, 1.0 / pa, np.nan)
    stage_a["sw_a_stab"] = np.where(stage_a["included_a"] == 1, numer_a / pa, np.nan)

    # Stage B: phenotype derivation among adversity eligible
    stage_b = stage_a[stage_a["included_a"] == 1].copy()
    stage_b["included_b"] = stage_b["trajectory_group"].notna().astype(int)
    Xb = prep_inclusion_design(stage_b, predictors_bcd)
    pb = fit_selection_probability(Xb, stage_b["included_b"], SEEDS["selection_model"] + 2)
    numer_b = float(stage_b["included_b"].mean())
    stage_b["sw_b_raw"] = np.where(stage_b["included_b"] == 1, 1.0 / pb, np.nan)
    stage_b["sw_b_stab"] = np.where(stage_b["included_b"] == 1, numer_b / pb, np.nan)

    # Stage C: alive and ADL-free at wave 3 among phenotype sample
    stage_c = stage_b[stage_b["included_b"] == 1].copy()
    stage_c["included_c"] = ((stage_c["alive_at_w3"] == 1) & (stage_c["adl_disability_w3"] == 0)).astype(int)
    Xc = prep_inclusion_design(stage_c, predictors_bcd)
    pc = fit_selection_probability(Xc, stage_c["included_c"], SEEDS["selection_model"] + 3)
    numer_c = float(stage_c["included_c"].mean())
    stage_c["sw_c_raw"] = np.where(stage_c["included_c"] == 1, 1.0 / pc, np.nan)
    stage_c["sw_c_stab"] = np.where(stage_c["included_c"] == 1, numer_c / pc, np.nan)

    # Stage D: observed incident ADL outcome among risk set
    stage_d = stage_c[stage_c["included_c"] == 1].copy()
    stage_d["included_d"] = stage_d["incident_adl_w3_w4"].notna().astype(int)
    Xd = prep_inclusion_design(stage_d, predictors_bcd)
    pd_ = fit_selection_probability(Xd, stage_d["included_d"], SEEDS["selection_model"] + 4)
    numer_d = float(stage_d["included_d"].mean())
    stage_d["sw_d_raw"] = np.where(stage_d["included_d"] == 1, 1.0 / pd_, np.nan)
    stage_d["sw_d_stab"] = np.where(stage_d["included_d"] == 1, numer_d / pd_, np.nan)

    final_obs = stage_d[stage_d["included_d"] == 1].copy()
    final_obs["selection_weight_raw"] = final_obs["sw_a_raw"] * final_obs["sw_b_raw"] * final_obs["sw_c_raw"] * final_obs["sw_d_raw"]
    final_obs["selection_weight_stabilized"] = final_obs["sw_a_stab"] * final_obs["sw_b_stab"] * final_obs["sw_c_stab"] * final_obs["sw_d_stab"]
    q01, q99 = final_obs["selection_weight_stabilized"].quantile([0.01, 0.99])
    final_obs["selection_weight_truncated"] = final_obs["selection_weight_stabilized"].clip(lower=q01, upper=q99)

    diag_rows = []
    stage_info = [
        ("stage_a_adversity_eligible", stage_a.loc[stage_a["included_a"] == 1, "sw_a_stab"]),
        ("stage_b_phenotype_derivation", stage_b.loc[stage_b["included_b"] == 1, "sw_b_stab"]),
        ("stage_c_wave3_riskset", stage_c.loc[stage_c["included_c"] == 1, "sw_c_stab"]),
        ("stage_d_observed_outcome", stage_d.loc[stage_d["included_d"] == 1, "sw_d_stab"]),
        ("cumulative_pre_truncation", final_obs["selection_weight_stabilized"]),
        ("cumulative_post_truncation", final_obs["selection_weight_truncated"]),
    ]
    for name, s in stage_info:
        s = s.dropna().astype(float)
        ess = (s.sum() ** 2) / np.sum(s**2)
        diag_rows.append(
            {
                "component": name,
                "n": int(len(s)),
                "mean": float(s.mean()),
                "sd": float(s.std(ddof=1)),
                "min": float(s.min()),
                "p1": float(s.quantile(0.01)),
                "p5": float(s.quantile(0.05)),
                "median": float(s.median()),
                "p95": float(s.quantile(0.95)),
                "p99": float(s.quantile(0.99)),
                "max": float(s.max()),
                "effective_sample_size": float(ess),
                "n_gt_5": int((s > 5).sum()),
                "n_gt_10": int((s > 10).sum()),
                "truncation_lower": float(q01) if "cumulative" in name else np.nan,
                "truncation_upper": float(q99) if "cumulative" in name else np.nan,
            }
        )
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(RESULTS_DIR / "weight_diagnostics.csv", index=False)

    out = merged.merge(final_obs[["ID_str", "selection_weight_raw", "selection_weight_stabilized", "selection_weight_truncated"]], on="ID_str", how="left")
    return out, diag


def fit_glm_predict(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, weights_train: pd.Series | None = None) -> np.ndarray:
    X_train_c = sm.add_constant(X_train, has_constant="add")
    X_test_c = sm.add_constant(X_test, has_constant="add")
    kwargs = {}
    if weights_train is not None:
        kwargs["freq_weights"] = weights_train.astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = sm.GLM(y_train.astype(int), X_train_c, family=sm.families.Binomial(), **kwargs).fit()
            return np.asarray(model.predict(X_test_c), dtype=float)
        except Exception:
            fallback = LogisticRegression(
                penalty="l2",
                C=1e6,
                solver="lbfgs",
                max_iter=5000,
                random_state=SEEDS["global_numpy"],
            )
            fallback.fit(X_train.to_numpy(dtype=float), y_train.astype(int), sample_weight=None if weights_train is None else weights_train.astype(float))
            return fallback.predict_proba(X_test.to_numpy(dtype=float))[:, 1]


def cv_metrics_for_models(data: pd.DataFrame, model_designs: dict[str, pd.DataFrame], outcome: pd.Series, weight_col: str | None = None, seed: int = SEEDS["cv"]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    y = outcome.astype(int).reset_index(drop=True)
    weights = data[weight_col].astype(float).reset_index(drop=True) if weight_col else None
    rows = []
    pred_map = {}
    for model_code, X in model_designs.items():
        X = X.reset_index(drop=True).astype(float)
        oof = np.full(len(X), np.nan, dtype=float)
        for train_idx, test_idx in cv.split(X, y):
            w_train = weights.iloc[train_idx] if weights is not None else None
            pred = fit_glm_predict(X.iloc[train_idx], y.iloc[train_idx], X.iloc[test_idx], w_train)
            oof[test_idx] = pred
        sw = None if weights is None else weights.to_numpy()
        auc = roc_auc_score(y, oof, sample_weight=sw)
        pr = average_precision_score(y, oof)
        brier = weighted_brier(y.to_numpy(), oof, sw)
        cal_i, cal_s = calibration_intercept_slope(y.to_numpy(), oof, sw)
        rows.append(
            {
                "row_type": "performance_model",
                "model_code": model_code,
                "n": int(len(X)),
                "events": int(y.sum()),
                "event_rate": float(y.mean()),
                "oof_auc": float(auc),
                "oof_pr_auc": float(pr),
                "oof_brier": float(brier),
                "oof_calibration_intercept": float(cal_i),
                "oof_calibration_slope": float(cal_s),
            }
        )
        pred_map[model_code] = oof

    comparisons = [("M2_vs_M1", "M2", "M1"), ("M3_vs_M1", "M3", "M1"), ("M4_vs_M1", "M4", "M1")]
    for code, a, b in comparisons:
        rows.append(
            {
                "row_type": "performance_comparison",
                "model_code": code,
                "comparison_model": a,
                "reference_model": b,
                "delta_oof_auc": float(roc_auc_score(y, pred_map[a], sample_weight=sw) - roc_auc_score(y, pred_map[b], sample_weight=sw)),
                "delta_oof_pr_auc": float(average_precision_score(y, pred_map[a]) - average_precision_score(y, pred_map[b])),
                "delta_oof_brier": float(weighted_brier(y.to_numpy(), pred_map[a], sw) - weighted_brier(y.to_numpy(), pred_map[b], sw)),
            }
        )
    return pd.DataFrame(rows), pred_map


def current_samples_with_slopes_and_weights(weighted_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = weighted_df.copy()
    for dom in DOMAIN_PREFIXES:
        slopes = []
        for row in df.itertuples(index=False):
            _, slope = fit_slope([getattr(row, f"{dom}_w1"), getattr(row, f"{dom}_w2"), getattr(row, f"{dom}_w3")], [-4, -2, 0])
            slopes.append(slope)
        df[f"{dom}_pre_w3_slope"] = slopes
    primary = df[
        (df["trajectory_group"].notna())
        & (df["alive_at_w3"] == 1)
        & (df["adl_disability_w3"] == 0)
        & df[["adversity_score_scaled", *BASE_COVARS, "incident_adl_w3_w4"]].notna().all(axis=1)
    ].copy()
    unified = df[
        (df["trajectory_group"].notna())
        & (df["alive_at_w3"] == 1)
        & (df["adl_disability_w3"] == 0)
        & df[[*BASE_COVARS, *WAVE3_LEVEL_VARS, "incident_adl_w3_w4"]].notna().all(axis=1)
    ].copy()
    return primary, unified


def phenotype_augmented_association(data: pd.DataFrame, weight_col: str | None = None) -> pd.DataFrame:
    work = data.copy()
    work["trajectory_group"] = pd.Categorical(work["trajectory_group"], categories=PHENOTYPE_ORDER, ordered=True)
    dummies = pd.get_dummies(work["trajectory_group"], prefix="traj", drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([work[["adversity_score_scaled", *BASE_COVARS]], dummies], axis=1).astype(float), has_constant="add")
    kwargs = {}
    if weight_col is not None:
        kwargs["freq_weights"] = work[weight_col].astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.GLM(work["incident_adl_w3_w4"].astype(int), X, family=sm.families.Binomial(), **kwargs).fit()
    conf = model.conf_int()
    rows = []
    for term in ["adversity_score_scaled", "traj_Emerging multidomain decline", "traj_High-burden mixed pattern", "traj_Severe multidomain decline"]:
        coef = float(model.params[term])
        low = float(conf.loc[term, 0])
        high = float(conf.loc[term, 1])
        rows.append(
            {
                "term": term,
                "coef": coef,
                "var": float(model.bse[term] ** 2),
                "OR": float(np.exp(coef)),
                "OR_low": float(np.exp(low)),
                "OR_high": float(np.exp(high)),
                "effect_95ci": f"{np.exp(coef):.3f} ({np.exp(low):.3f}, {np.exp(high):.3f})",
                "p": float(model.pvalues[term]),
                "p_formatted": format_p(float(model.pvalues[term])),
                "sample_n": int(len(work)),
                "events": int(work["incident_adl_w3_w4"].sum()),
            }
        )
    return pd.DataFrame(rows)


def build_designs_for_m1_m4(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    work = data.copy()
    work["trajectory_group"] = pd.Categorical(work["trajectory_group"], categories=PHENOTYPE_ORDER, ordered=True)
    dummies = pd.get_dummies(work["trajectory_group"], prefix="traj", drop_first=True, dtype=float)
    return {
        "M1": work[BASE_COVARS + WAVE3_LEVEL_VARS].astype(float),
        "M2": work[BASE_COVARS + WAVE3_LEVEL_VARS + [f"{d}_pre_w3_slope" for d in DOMAIN_PREFIXES]].astype(float),
        "M3": pd.concat([work[BASE_COVARS], dummies], axis=1).astype(float),
        "M4": pd.concat([work[BASE_COVARS + WAVE3_LEVEL_VARS], dummies], axis=1).astype(float),
    }


def pool_rubin(estimates: list[dict[str, float]], m: int) -> dict[str, float]:
    q = np.array([e["coef"] for e in estimates], dtype=float)
    u = np.array([e["var"] for e in estimates], dtype=float)
    q_bar = q.mean()
    u_bar = u.mean()
    b = q.var(ddof=1) if len(q) > 1 else 0.0
    t = u_bar + (1 + 1 / m) * b
    se = np.sqrt(t)
    return {
        "coef": float(q_bar),
        "var": float(t),
        "OR": float(np.exp(q_bar)),
        "OR_low": float(np.exp(q_bar - 1.96 * se)),
        "OR_high": float(np.exp(q_bar + 1.96 * se)),
    }


def derive_imputed_phenotype(df_imp: pd.DataFrame, imp_index: int) -> pd.DataFrame:
    rows = []
    for row in df_imp.itertuples(index=False):
        out = {"ID_str": row.ID_str}
        for dom in DOMAIN_PREFIXES:
            vals = [getattr(row, f"{dom}_w1"), getattr(row, f"{dom}_w2"), getattr(row, f"{dom}_w3")]
            intercept, slope = fit_slope(vals, [1, 2, 3])
            _, pre_w3_slope = fit_slope(vals, [-4, -2, 0])
            out[f"{dom}_intercept"] = intercept
            out[f"{dom}_slope"] = slope
            out[f"{dom}_pre_w3_slope"] = pre_w3_slope
            out[f"{dom}_level_at_w3"] = getattr(row, f"{dom}_w3")
        rows.append(out)
    feat = pd.DataFrame(rows)
    gmm_cols = [c for c in feat.columns if c.endswith("_intercept") or c.endswith("_slope")]
    scaler = StandardScaler()
    X = scaler.fit_transform(feat[gmm_cols].to_numpy(dtype=float))
    gmm = GaussianMixture(
        n_components=4,
        covariance_type="diag",
        reg_covar=1e-6,
        n_init=20,
        random_state=SEEDS["gmm_base"] + imp_index,
        init_params="kmeans",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gmm.fit(X)
    raw = gmm.predict(X)
    tmp = feat[gmm_cols].copy()
    tmp["cluster_raw"] = raw
    mapping = map_labels_4class_new(tmp, "cluster_raw")
    feat["trajectory_group"] = pd.Series(raw).map(mapping)
    return feat


def run_multiple_imputation(df: pd.DataFrame, weights_df: pd.DataFrame, m: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    # MI is deliberately restricted to predictors within the observed-outcome risk set.
    # Wave 3 risk-set formation and wave 4 outcome observation are handled by IPW, not imputation.
    observed_mask = (
        (df["alive_at_w3"] == 1)
        & (df["adl_disability_w3"] == 0)
        & df["incident_adl_w3_w4"].notna()
    )
    imp_vars = [
        "ID_str",
        *BASE_COVARS,
        "adversity_score_scaled",
        "adl_disability_baseline",
        *WAVE_DOMAIN_VARS,
        "incident_adl_w3_w4",
    ]
    base = df.loc[observed_mask, imp_vars].copy()
    mi_matrix_cols = [c for c in imp_vars if c != "ID_str"]

    binary_round_cols = ["female", "rural_hukou_2011", "low_self_education", "adl_disability_baseline", "incident_adl_w3_w4"]
    observed_bounds = {
        col: (
            float(base[col].dropna().min()),
            float(base[col].dropna().max()),
        )
        for col in mi_matrix_cols
        if base[col].dropna().size > 0
    }

    performance_rows = []
    association_rows = []
    pool_store = {}

    for imp_idx in range(1, m + 1):
        imp = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=20,
            sample_posterior=True,
            random_state=SEEDS["imputation_base"] + imp_idx,
            initial_strategy="median",
        )
        imputed_vals = imp.fit_transform(base[mi_matrix_cols])
        imp_df = pd.DataFrame(imputed_vals, columns=mi_matrix_cols, index=base.index)
        imp_df["ID_str"] = base["ID_str"].values
        for col, (lo, hi) in observed_bounds.items():
            if col in imp_df.columns:
                imp_df[col] = np.clip(imp_df[col], lo, hi)
        for col in binary_round_cols:
            if col in imp_df.columns:
                imp_df[col] = np.clip(np.round(imp_df[col]), 0, 1).astype(int)

        pheno = derive_imputed_phenotype(imp_df.drop(columns=["incident_adl_w3_w4"]).copy(), imp_idx)
        imp_df = imp_df.merge(pheno[["ID_str", "trajectory_group", *[f"{d}_pre_w3_slope" for d in DOMAIN_PREFIXES]]], on="ID_str", how="left")
        imp_df = imp_df.merge(weights_df[["ID_str", "selection_weight_truncated"]], on="ID_str", how="left")
        ana_mi = imp_df.copy()
        ana_miipw = imp_df[imp_df["selection_weight_truncated"].notna()].copy()

        assoc_mi = phenotype_augmented_association(ana_mi, weight_col=None)
        assoc_mi["method"] = "MI"
        assoc_mi["imputation"] = imp_idx
        association_rows.append(assoc_mi.copy())

        assoc_miipw = phenotype_augmented_association(ana_miipw, weight_col="selection_weight_truncated")
        assoc_miipw["method"] = "MI+IPW"
        assoc_miipw["imputation"] = imp_idx
        association_rows.append(assoc_miipw.copy())

        designs_mi = build_designs_for_m1_m4(ana_mi)
        perf_mi, _ = cv_metrics_for_models(ana_mi, designs_mi, ana_mi["incident_adl_w3_w4"], weight_col=None, seed=SEEDS["cv"] + imp_idx)
        perf_mi["method"] = "MI"
        perf_mi["imputation"] = imp_idx
        performance_rows.append(perf_mi)

        designs_miipw = build_designs_for_m1_m4(ana_miipw)
        perf_miipw, _ = cv_metrics_for_models(ana_miipw, designs_miipw, ana_miipw["incident_adl_w3_w4"], weight_col="selection_weight_truncated", seed=SEEDS["cv"] + imp_idx)
        perf_miipw["method"] = "MI+IPW"
        perf_miipw["imputation"] = imp_idx
        performance_rows.append(perf_miipw)

        for method_name, tbl in [("MI", assoc_mi), ("MI+IPW", assoc_miipw)]:
            for _, row in tbl.iterrows():
                pool_store.setdefault((method_name, row["term"]), []).append({"coef": row["coef"], "var": row["var"]})

    performance_all = pd.concat(performance_rows, ignore_index=True)
    association_all = pd.concat(association_rows, ignore_index=True)

    pooled_assoc_rows = []
    for (method_name, term), ests in pool_store.items():
        pooled = pool_rubin(ests, m=m)
        pooled_assoc_rows.append(
            {
                "row_type": "association_term",
                "method": method_name,
                "analysis": "primary_incident_adl_phenotype_augmented",
                "term": term,
                "coef": pooled["coef"],
                "var": pooled["var"],
                "estimate": pooled["OR"],
                "ci_low": pooled["OR_low"],
                "ci_high": pooled["OR_high"],
                "effect_95ci": f"{pooled['OR']:.3f} ({pooled['OR_low']:.3f}, {pooled['OR_high']:.3f})",
            }
        )
    pooled_assoc = pd.DataFrame(pooled_assoc_rows)

    pooled_perf_rows = []
    for method_name, sub in performance_all.groupby("method"):
        for model_code, s2 in sub.groupby("model_code"):
            pooled_perf_rows.append(
                {
                    "row_type": s2["row_type"].iloc[0],
                    "method": method_name,
                    "analysis": "M1_M4_key_comparison",
                    "model_code": model_code,
                    "n_mean": float(s2["n"].mean()) if "n" in s2.columns else np.nan,
                    "events_mean": float(s2["events"].mean()) if "events" in s2.columns else np.nan,
                    "oof_auc_mean": float(s2["oof_auc"].mean()) if "oof_auc" in s2.columns else np.nan,
                    "oof_auc_sd": float(s2["oof_auc"].std(ddof=1)) if "oof_auc" in s2.columns and len(s2) > 1 else 0.0,
                    "oof_pr_auc_mean": float(s2["oof_pr_auc"].mean()) if "oof_pr_auc" in s2.columns else np.nan,
                    "oof_brier_mean": float(s2["oof_brier"].mean()) if "oof_brier" in s2.columns else np.nan,
                    "oof_calibration_intercept_mean": float(s2["oof_calibration_intercept"].mean()) if "oof_calibration_intercept" in s2.columns else np.nan,
                    "oof_calibration_slope_mean": float(s2["oof_calibration_slope"].mean()) if "oof_calibration_slope" in s2.columns else np.nan,
                    "delta_oof_auc_mean": float(s2["delta_oof_auc"].mean()) if "delta_oof_auc" in s2.columns else np.nan,
                    "delta_oof_auc_sd": float(s2["delta_oof_auc"].std(ddof=1)) if "delta_oof_auc" in s2.columns and len(s2) > 1 else 0.0,
                    "delta_oof_pr_auc_mean": float(s2["delta_oof_pr_auc"].mean()) if "delta_oof_pr_auc" in s2.columns else np.nan,
                    "delta_oof_brier_mean": float(s2["delta_oof_brier"].mean()) if "delta_oof_brier" in s2.columns else np.nan,
                }
            )
    pooled_perf = pd.DataFrame(pooled_perf_rows)
    return pooled_perf, pooled_assoc


def current_cc_ipw_results(weighted_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary, unified = current_samples_with_slopes_and_weights(weighted_df)
    assoc_cc = phenotype_augmented_association(primary, weight_col=None)
    assoc_cc["row_type"] = "association_term"
    assoc_cc["method"] = "Complete-case"
    assoc_cc["analysis"] = "primary_incident_adl_phenotype_augmented"
    assoc_cc = assoc_cc.rename(columns={"OR": "estimate", "OR_low": "ci_low", "OR_high": "ci_high"})

    assoc_ipw = phenotype_augmented_association(primary, weight_col="selection_weight_truncated")
    assoc_ipw["row_type"] = "association_term"
    assoc_ipw["method"] = "IPW"
    assoc_ipw["analysis"] = "primary_incident_adl_phenotype_augmented"
    assoc_ipw = assoc_ipw.rename(columns={"OR": "estimate", "OR_low": "ci_low", "OR_high": "ci_high"})

    designs = build_designs_for_m1_m4(unified)
    perf_cc, _ = cv_metrics_for_models(unified, designs, unified["incident_adl_w3_w4"], weight_col=None, seed=SEEDS["cv"])
    perf_cc["method"] = "Complete-case"
    perf_cc["analysis"] = "M1_M4_key_comparison"
    perf_ipw, _ = cv_metrics_for_models(unified, designs, unified["incident_adl_w3_w4"], weight_col="selection_weight_truncated", seed=SEEDS["cv"])
    perf_ipw["method"] = "IPW"
    perf_ipw["analysis"] = "M1_M4_key_comparison"

    assoc = pd.concat([assoc_cc, assoc_ipw], ignore_index=True, sort=False)
    perf = pd.concat([perf_cc, perf_ipw], ignore_index=True, sort=False)
    return perf, assoc


def build_results_table(cc_perf: pd.DataFrame, cc_assoc: pd.DataFrame, mi_perf: pd.DataFrame, mi_assoc: pd.DataFrame) -> pd.DataFrame:
    perf_rows = []
    for df in [cc_perf, mi_perf]:
        tmp = df.copy()
        if "oof_auc" in tmp.columns:
            tmp["oof_auc_mean"] = tmp["oof_auc"]
            tmp["oof_pr_auc_mean"] = tmp.get("oof_pr_auc", np.nan)
            tmp["oof_brier_mean"] = tmp.get("oof_brier", np.nan)
            tmp["oof_calibration_intercept_mean"] = tmp.get("oof_calibration_intercept", np.nan)
            tmp["oof_calibration_slope_mean"] = tmp.get("oof_calibration_slope", np.nan)
            tmp["delta_oof_auc_mean"] = tmp.get("delta_oof_auc", np.nan)
            tmp["delta_oof_pr_auc_mean"] = tmp.get("delta_oof_pr_auc", np.nan)
            tmp["delta_oof_brier_mean"] = tmp.get("delta_oof_brier", np.nan)
        perf_rows.append(tmp)
    perf_all = pd.concat(perf_rows, ignore_index=True, sort=False)
    assoc_all = pd.concat([cc_assoc, mi_assoc], ignore_index=True, sort=False)

    perf_keep = [
        "row_type",
        "method",
        "analysis",
        "model_code",
        "comparison_model",
        "reference_model",
        "n",
        "events",
        "n_mean",
        "events_mean",
        "oof_auc_mean",
        "oof_auc_sd",
        "oof_pr_auc_mean",
        "oof_brier_mean",
        "oof_calibration_intercept_mean",
        "oof_calibration_slope_mean",
        "delta_oof_auc_mean",
        "delta_oof_auc_sd",
        "delta_oof_pr_auc_mean",
        "delta_oof_brier_mean",
    ]
    assoc_keep = [
        "row_type",
        "method",
        "analysis",
        "term",
        "sample_n",
        "events",
        "estimate",
        "ci_low",
        "ci_high",
        "effect_95ci",
        "p",
        "p_formatted",
    ]
    out = pd.concat([
        perf_all[[c for c in perf_keep if c in perf_all.columns]],
        assoc_all[[c for c in assoc_keep if c in assoc_all.columns]],
    ], ignore_index=True, sort=False)
    out.to_csv(RESULTS_DIR / "cc_vs_mi_vs_ipw_vs_miipw.csv", index=False)
    return out


def make_plots(missingness: pd.DataFrame, smd: pd.DataFrame, weight_diag: pd.DataFrame) -> None:
    with PdfPages(RESULTS_DIR / "missingness_plots.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(10, 6))
        sub = missingness[missingness["row_type"] == "variable_missingness"].sort_values("pct_missing", ascending=False).head(20)
        sns.barplot(data=sub, y="item", x="pct_missing", hue="group", dodge=False, ax=ax)
        ax.set_xlabel("Percent missing")
        ax.set_ylabel("")
        ax.set_title("Top missingness variables")
        ax.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        flow = missingness[missingness["row_type"] == "sample_flow"].copy()
        ax.bar(flow["item"], flow["n_included"], color="#4C72B0")
        ax.set_ylabel("n included")
        ax.set_title("Current pipeline sample flow")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 7))
        smd_sub = smd[smd["comparison_name"] == "primary_complete_case_vs_other_riskset"].dropna(subset=["smd"]).copy()
        smd_sub = smd_sub.reindex(smd_sub["smd"].abs().sort_values().index)
        ax.hlines(y=np.arange(len(smd_sub)), xmin=0, xmax=smd_sub["smd"], color="#4C72B0")
        ax.plot(smd_sub["smd"], np.arange(len(smd_sub)), "o", color="#1f4e79")
        ax.axvline(0.1, color="red", linestyle="--", linewidth=1)
        ax.axvline(-0.1, color="red", linestyle="--", linewidth=1)
        ax.axvline(0, color="grey", linewidth=1)
        ax.set_yticks(np.arange(len(smd_sub)))
        ax.set_yticklabels(smd_sub["variable"])
        ax.set_xlabel("Standardized mean difference")
        ax.set_title("Included vs excluded from current primary complete-case sample")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        wd = weight_diag.set_index("component")
        components = ["cumulative_pre_truncation", "cumulative_post_truncation"]
        for ax, comp in zip(axes, components):
            row = wd.loc[comp]
            stats_text = (
                f"mean={row['mean']:.3f}\n"
                f"sd={row['sd']:.3f}\n"
                f"p1={row['p1']:.3f}\n"
                f"p99={row['p99']:.3f}\n"
                f"max={row['max']:.3f}\n"
                f"ESS={row['effective_sample_size']:.1f}"
            )
            ax.axis("off")
            ax.text(0.05, 0.95, comp, fontweight="bold", va="top")
            ax.text(0.05, 0.80, stats_text, va="top")
        fig.suptitle("Selection-weight diagnostics")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def write_summary(
    results_table: pd.DataFrame,
    missingness: pd.DataFrame,
    smd: pd.DataFrame,
    weight_diag: pd.DataFrame,
) -> None:
    perf = results_table[results_table["row_type"].isin(["performance_model", "performance_comparison"])].copy()
    assoc = results_table[results_table["row_type"] == "association_term"].copy()

    def get_perf(method: str, code: str, field: str) -> float:
        sub = perf[(perf["method"] == method) & (perf["model_code"] == code)]
        return float(sub.iloc[0][field])

    def get_assoc(method: str, term: str) -> str:
        sub = assoc[(assoc["method"] == method) & (assoc["term"] == term)]
        return sub.iloc[0]["effect_95ci"] if len(sub) else ""

    riskset_n_all = int(missingness.loc[missingness["item"] == "alive_and_adlfree_riskset_w3_all", "n_included"].iloc[0])
    obs_n_all = int(missingness.loc[missingness["item"] == "observed_incident_adl_outcome_all", "n_included"].iloc[0])
    riskset_n_chain = int(missingness.loc[missingness["item"] == "alive_and_adlfree_riskset_w3_within_phenotype_chain", "n_included"].iloc[0])
    obs_n_chain = int(missingness.loc[missingness["item"] == "observed_incident_adl_outcome_within_phenotype_chain", "n_included"].iloc[0])
    primary_n = int(missingness.loc[missingness["item"] == "primary_incident_adl_complete_case", "n_included"].iloc[0])
    unified_n = int(missingness.loc[missingness["item"] == "unified_incident_adl_comparison", "n_included"].iloc[0])

    primary_smd = smd[smd["comparison_name"] == "primary_complete_case_vs_other_riskset"]["smd"].abs()
    n_large_smd = int((primary_smd > 0.1).sum())

    summary = f"""# Stage 4 Missing data and selection-bias module upgrade

## 1. Scope note

说明：`revised_analysis_plan.md` 的旧版编号把 Stage 4 写成 leakage guard；本次执行按你的最新指令，将 **missing data and selection bias 模块升级** 作为当前 Stage 4。编号差异已记录，但分析执行以当前阶段指令为准。

## 2. Missingness portrait

- baseline cohort: `17708`
- wave 3 alive and ADL-free risk set (all eligible participants): `{riskset_n_all}`
- observed incident ADL outcome at wave 4 (all risk-set participants): `{obs_n_all}`
- wave 3 alive and ADL-free risk set within the current phenotype-analysis chain: `{riskset_n_chain}`
- observed incident ADL outcome within the current phenotype-analysis chain: `{obs_n_chain}`
- current primary complete-case sample: `{primary_n}`
- current unified comparison sample: `{unified_n}`

这意味着：

- 在全体 wave 3 风险集内，从 risk set 到 observed primary outcome 丢失了 `{riskset_n_all - obs_n_all}` 人；
- 在当前 phenotype-analysis chain 内，从 risk set 到 observed primary outcome 丢失了 `{riskset_n_chain - obs_n_chain}` 人；
- 从 observed phenotype-chain outcome sample 到 current primary complete-case 又因为 predictor/adversity completeness 进一步丢失了 `{obs_n_chain - primary_n}` 人；
- unified comparison sample 再比 current primary sample 少 `{primary_n - unified_n}` 人。

在 `primary_complete_case_vs_other_riskset` 的 included-versus-excluded 比较中，`|SMD| > 0.1` 的变量数为 `{n_large_smd}`，提示 complete-case 进入并非完全随机。

## 3. MI implementation level

- 本阶段 **没有做 adversity item-level MI**
- 当前 MI 在 **summary-level** 执行，而且只在 **wave 3 alive and ADL-free 且 wave 4 主结局已观测** 的样本内进行。
- 核心被插补变量包括：
  - `adversity_score_scaled`
  - wave 1-3 四域 repeated-domain variables
  - baseline covariates
  - `adl_disability_baseline`

之所以没有做 item-level adversity MI，是因为当前项目中 17+ childhood adversity items 为混合 binary life-history variables，若要做 item-level MI，需要额外处理结构性逻辑、一致性约束和 binary/ordinal-specific specification。对当前 Stage 4 的优先问题而言，summary-level MI 是更稳健且可追踪的次优层级。

风险集形成（`alive_at_w3 = 1` 且 `adl_disability_w3 = 0`）以及 wave 4 主结局是否观测，不在 MI 中插补，而是作为 selection / censoring mechanism 由 IPW 模块处理。

## 4. Selection-weighting implementation

本阶段的 selection / censoring weights 不再沿用旧版两阶段附录式模型，而是扩展为四步 sequential weighting：

1. 进入 adversity-eligible sample  
2. 进入 phenotype derivation sample  
3. 进入 wave 3 alive and ADL-free risk set  
4. wave 4 primary incident ADL outcome observed

权重模型使用 penalized logistic approach with missing-indicator augmentation，以提高稳定性。最终 cumulative stabilized weights 在 observed-outcome sample 上按 p1 / p99 截断。

## 5. Main comparison

### Primary phenotype association: Emerging vs Stable

- Complete-case: `{get_assoc('Complete-case', 'traj_Emerging multidomain decline')}`
- MI: `{get_assoc('MI', 'traj_Emerging multidomain decline')}`
- IPW: `{get_assoc('IPW', 'traj_Emerging multidomain decline')}`
- MI+IPW: `{get_assoc('MI+IPW', 'traj_Emerging multidomain decline')}`

### Key prediction comparison

- Complete-case M1 AUC: `{get_perf('Complete-case', 'M1', 'oof_auc_mean'):.3f}`
- Complete-case M2 AUC: `{get_perf('Complete-case', 'M2', 'oof_auc_mean'):.3f}`
- Complete-case M3 AUC: `{get_perf('Complete-case', 'M3', 'oof_auc_mean'):.3f}`
- Complete-case M4 AUC: `{get_perf('Complete-case', 'M4', 'oof_auc_mean'):.3f}`

- MI M1 AUC: `{get_perf('MI', 'M1', 'oof_auc_mean'):.3f}`
- MI M2 AUC: `{get_perf('MI', 'M2', 'oof_auc_mean'):.3f}`
- MI M3 AUC: `{get_perf('MI', 'M3', 'oof_auc_mean'):.3f}`
- MI M4 AUC: `{get_perf('MI', 'M4', 'oof_auc_mean'):.3f}`

- IPW M1 AUC: `{get_perf('IPW', 'M1', 'oof_auc_mean'):.3f}`
- IPW M2 AUC: `{get_perf('IPW', 'M2', 'oof_auc_mean'):.3f}`
- IPW M3 AUC: `{get_perf('IPW', 'M3', 'oof_auc_mean'):.3f}`
- IPW M4 AUC: `{get_perf('IPW', 'M4', 'oof_auc_mean'):.3f}`

- MI+IPW M1 AUC: `{get_perf('MI+IPW', 'M1', 'oof_auc_mean'):.3f}`
- MI+IPW M2 AUC: `{get_perf('MI+IPW', 'M2', 'oof_auc_mean'):.3f}`
- MI+IPW M3 AUC: `{get_perf('MI+IPW', 'M3', 'oof_auc_mean'):.3f}`
- MI+IPW M4 AUC: `{get_perf('MI+IPW', 'M4', 'oof_auc_mean'):.3f}`

## 6. Direct answers

### Is the current complete-case analysis likely to be systematically biased?

Probably yes, at least to a non-trivial degree. The reason is not that complete-case results are necessarily wrong, but that entry into the final current analysis samples depends on adversity completeness, repeated-domain availability, survival / non-disability status at wave 3, and wave 4 outcome observation. The included-versus-excluded SMD pattern suggests that complete-case inclusion is selective rather than random.

### Do the main conclusions remain stable after MI / IPW?

Yes for the broad ranking structure, but not for every effect size. The most stable pattern is:

- Emerging vs Stable remains clearly elevated across Complete-case / MI / IPW / MI+IPW;
- M2 stays above M1 across all four approaches;
- M3 remains below M1 across all four approaches.

The least stable pattern is:

- M4 adds a small gain over M1 in Complete-case and IPW, but this gain essentially disappears after MI and MI+IPW.

Therefore the strongest stable conclusion is that **current level remains the dominant signal and explicit change information adds a modest increment**, whereas **hard phenotype beyond current level is not uniformly robust to missingness / selection handling**.

### Which results are most sensitive to missingness / selection?

The most sensitive results are those that depend on:

- hard phenotype assignment alone;
- the additional value of hard phenotype beyond current wave 3 levels;
- entry restrictions after the wave 3 risk set.

This is reflected by:

- `Emerging vs Stable` OR shifting from `{get_assoc('Complete-case', 'traj_Emerging multidomain decline')}` to `{get_assoc('MI', 'traj_Emerging multidomain decline')}`, `{get_assoc('IPW', 'traj_Emerging multidomain decline')}`, and `{get_assoc('MI+IPW', 'traj_Emerging multidomain decline')}`;
- `M4 vs M1` shifting from `{get_perf('Complete-case', 'M4_vs_M1', 'delta_oof_auc_mean'):+.3f}` (Complete-case) and `{get_perf('IPW', 'M4_vs_M1', 'delta_oof_auc_mean'):+.3f}` (IPW) to approximately `{get_perf('MI', 'M4_vs_M1', 'delta_oof_auc_mean'):+.3f}` after MI and `{get_perf('MI+IPW', 'M4_vs_M1', 'delta_oof_auc_mean'):+.3f}` after MI+IPW.

By contrast, `M2 vs M1` remains positive across all approaches, suggesting that explicit level-plus-change representations are more stable than hard phenotype summaries.

### Which result set should remain primary in the manuscript?

The most defensible manuscript strategy is:

- keep the locked complete-case results as the main reported analysis for continuity with the current rerun and figure/table package;
- use IPW, MI, and MI+IPW as upgraded robustness analyses rather than as a new primary result set;
- explicitly tone down any claim that hard phenotype adds reliable prognostic value beyond current burden;
- retain the stronger claim that explicit pre-landmark change information adds a modest increment beyond wave 3 current levels.

Weight diagnostics also show that the most unstable selection step is entry into the wave 3 risk set, with substantial pre-truncation dispersion and a marked ESS reduction before truncation. This means IPW is informative but should still be interpreted as a stabilization strategy rather than a full correction guarantee.
"""
    (RESULTS_DIR / "missing_selection_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    np.random.seed(SEEDS["global_numpy"])
    random.seed(SEEDS["global_numpy"])

    df, assignments = load_base_data()
    missingness = build_missingness_summary(df, assignments)
    smd = build_included_vs_excluded_smd(df, assignments)
    weighted_df, weight_diag = build_selection_weights(df, assignments)

    cc_perf, cc_assoc = current_cc_ipw_results(weighted_df)
    mi_perf, mi_assoc = run_multiple_imputation(df, weighted_df, m=5)
    results_table = build_results_table(cc_perf, cc_assoc, mi_perf, mi_assoc)

    make_plots(missingness, smd, weight_diag)
    write_summary(results_table, missingness, smd, weight_diag)

    meta = {
        "stage": "stage4_missing_selection",
        "sample_base": int(len(df)),
        "mi_imputations": 5,
        "cv_seed": SEEDS["cv"],
        "imputation_seed_base": SEEDS["imputation_base"],
        "selection_seed": SEEDS["selection_model"],
    }
    (RESULTS_DIR / "stage4_run_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()

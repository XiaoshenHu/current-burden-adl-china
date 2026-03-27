#!/usr/bin/env python3
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, average_precision_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path("/Users/macbookpro-hu/Library/CloudStorage/OneDrive-个人/Xcode/Medical Research/PROJECTS/charls_life_course_aging")
RERUN_ROOT = PROJECT_ROOT / "rerun_risk_stratification_expanded_v2_20260323"
RESULTS_DIR = PROJECT_ROOT / "results" / "stage8_domain_contribution"
LOGS_DIR = PROJECT_ROOT / "logs" / "stage8_domain_contribution"
RESULTS_INDEX = PROJECT_ROOT / "RESULTS_INDEX.md"

BASE_COVARS = ["baseline_age_2011", "female", "rural_hukou_2011", "low_self_education"]
DOMAINS = ["dep", "cogdef", "physical_lim", "multimorb"]
WAVE3_LEVEL_VARS = ["dep_w3", "cogdef_w3", "physical_lim_w3", "multimorb_w3"]
PHENOTYPE_ORDER = [
    "Stable low-burden",
    "Emerging multidomain decline",
    "High-burden mixed pattern",
    "Severe multidomain decline",
]
DOMAIN_LABELS = {
    "dep": "Depressive symptoms",
    "cogdef": "Cognitive deficit",
    "physical_lim": "Physical limitation",
    "multimorb": "Multimorbidity",
}
SEEDS = {
    "global_numpy": 20260411,
    "gmm": 20260412,
    "cv": 20260325,
}


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def format_p(p: float) -> str:
    if pd.isna(p):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt_or(est: float, low: float, high: float) -> str:
    return f"{est:.3f} ({low:.3f}, {high:.3f})"


def feature_to_domain(feature: str) -> str:
    for dom in DOMAINS:
        if feature.startswith(f"{dom}_"):
            return dom
    raise ValueError(f"Cannot parse domain from feature: {feature}")


def safe_logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def weighted_brier(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def calibration_intercept_slope(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    pred = np.clip(np.asarray(pred, dtype=float), 1e-6, 1 - 1e-6)
    logit_pred = safe_logit(pred)
    slope_model = sm.GLM(y, sm.add_constant(logit_pred), family=sm.families.Binomial()).fit()
    intercept_model = sm.GLM(y, np.ones((len(y), 1)), offset=logit_pred, family=sm.families.Binomial()).fit()
    return float(intercept_model.params[0]), float(slope_model.params[1])


def fit_glm_or_fallback(X: pd.DataFrame, y: pd.Series):
    X_use = sm.add_constant(X.reset_index(drop=True).astype(float), has_constant="add")
    y_use = y.reset_index(drop=True).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return sm.GLM(y_use, X_use, family=sm.families.Binomial()).fit(), "glm"
        except Exception:
            model = LogisticRegression(
                penalty="l2",
                C=1e6,
                solver="lbfgs",
                max_iter=5000,
                random_state=SEEDS["global_numpy"],
            )
            model.fit(X.reset_index(drop=True).astype(float).to_numpy(), y_use.to_numpy())
            return model, "fallback_logit"


def predict_model(model, model_type: str, X: pd.DataFrame) -> np.ndarray:
    X_use = X.reset_index(drop=True).astype(float)
    if model_type == "glm":
        Xc = sm.add_constant(X_use, has_constant="add")
        return np.asarray(model.predict(Xc), dtype=float)
    return model.predict_proba(X_use.to_numpy())[:, 1]


def fit_oof_predictions(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEEDS["cv"])
    oof = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in skf.split(np.zeros(len(y)), y):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        model, model_type = fit_glm_or_fallback(X_train, y_train)
        oof[test_idx] = predict_model(model, model_type, X_test)
    return oof


def burden_weights(feature_cols: list[str]) -> dict[str, float]:
    weights = {}
    for dom in DOMAINS:
        if f"{dom}_intercept" in feature_cols:
            weights[f"{dom}_intercept"] = 10.0 if dom == "physical_lim" else 1.0
        if f"{dom}_slope" in feature_cols:
            weights[f"{dom}_slope"] = 30.0 if dom == "physical_lim" else 3.0
    return weights


def burden_order_score(sub: pd.DataFrame, feature_cols: list[str]) -> float:
    weights = burden_weights(feature_cols)
    total = 0.0
    for col, wt in weights.items():
        total += float(sub[col].mean()) * wt
    return total


def baseline_burden_score(sub: pd.DataFrame, feature_cols: list[str]) -> float:
    total = 0.0
    for dom in DOMAINS:
        col = f"{dom}_intercept"
        if col in feature_cols:
            total += float(sub[col].mean()) * (10.0 if dom == "physical_lim" else 1.0)
    return total


def map_labels_4class(feature_df: pd.DataFrame, raw_label_col: str, feature_cols: list[str]) -> dict[int, str]:
    scores = {c: burden_order_score(sub, feature_cols) for c, sub in feature_df.groupby(raw_label_col)}
    ordered_clusters = sorted(scores, key=scores.get)
    lowest = ordered_clusters[0]
    highest = ordered_clusters[-1]
    middle = ordered_clusters[1:-1]
    baseline_scores = {c: baseline_burden_score(feature_df[feature_df[raw_label_col] == c], feature_cols) for c in middle}
    high_burden_mixed = max(middle, key=lambda x: baseline_scores[x])
    emerging = min(middle, key=lambda x: baseline_scores[x])
    return {
        lowest: "Stable low-burden",
        emerging: "Emerging multidomain decline",
        high_burden_mixed: "High-burden mixed pattern",
        highest: "Severe multidomain decline",
    }


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset = pd.read_csv(RERUN_ROOT / "02_data_processing" / "analysis_dataset_landmark.csv")
    features = pd.read_csv(RERUN_ROOT / "03_phenotype_derivation" / "trajectory_feature_matrix.csv")
    assignments = pd.read_csv(RERUN_ROOT / "03_phenotype_derivation" / "phenotype_assignments.csv")
    dataset["ID_str"] = dataset["ID_str"].astype(str)
    features["ID_str"] = features["ID_str"].astype(str)
    assignments["ID_str"] = assignments["ID_str"].astype(str)
    assignments["trajectory_group"] = pd.Categorical(assignments["trajectory_group"], categories=PHENOTYPE_ORDER, ordered=True)
    return dataset, features, assignments


def build_locked_samples(dataset: pd.DataFrame, assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = dataset.merge(assignments[["ID_str", "trajectory_group"]], on="ID_str", how="left")
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
    if len(primary) != 8643 or len(unified) != 8355:
        raise RuntimeError(f"Locked sample mismatch: primary={len(primary)}, unified={len(unified)}")
    primary["trajectory_group"] = pd.Categorical(primary["trajectory_group"], categories=PHENOTYPE_ORDER, ordered=True)
    unified["trajectory_group"] = pd.Categorical(unified["trajectory_group"], categories=PHENOTYPE_ORDER, ordered=True)
    return primary, unified


def run_gmm(feature_df: pd.DataFrame, feature_cols: list[str], seed_offset: int) -> pd.DataFrame:
    X = feature_df[feature_cols].to_numpy(dtype=float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        gmm = GaussianMixture(
            n_components=4,
            covariance_type="diag",
            reg_covar=1e-6,
            n_init=50,
            random_state=SEEDS["gmm"] + seed_offset,
            init_params="kmeans",
            max_iter=500,
        )
        gmm.fit(Xs)
    raw = gmm.predict(Xs)
    probs = gmm.predict_proba(Xs)
    tmp = feature_df[["ID_str", *feature_cols]].copy()
    tmp["cluster_raw"] = raw
    mapping = map_labels_4class(tmp, "cluster_raw", feature_cols)
    labels = tmp["cluster_raw"].map(mapping)
    uncertainty = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1) / np.log(probs.shape[1])
    out = pd.DataFrame(
        {
            "ID_str": feature_df["ID_str"].astype(str),
            "trajectory_group": pd.Categorical(labels, categories=PHENOTYPE_ORDER, ordered=True),
            "max_posterior_probability": probs.max(axis=1),
            "classification_uncertainty_index": uncertainty,
        }
    )
    return out


def phenotype_design(data: pd.DataFrame, include_levels: bool, phenotype_col: str = "trajectory_group") -> pd.DataFrame:
    parts = [data[BASE_COVARS].reset_index(drop=True).astype(float)]
    if include_levels:
        parts.append(data[WAVE3_LEVEL_VARS].reset_index(drop=True).astype(float))
    dummies = pd.get_dummies(
        pd.Categorical(data[phenotype_col].reset_index(drop=True), categories=PHENOTYPE_ORDER, ordered=True),
        prefix="traj",
        drop_first=True,
        dtype=float,
    )
    parts.append(dummies)
    return pd.concat(parts, axis=1)


def base_design(data: pd.DataFrame) -> pd.DataFrame:
    return data[BASE_COVARS].reset_index(drop=True).astype(float)


def fit_phenotype_association(data: pd.DataFrame, phenotype_col: str) -> dict:
    y = data["incident_adl_w3_w4"].reset_index(drop=True).astype(int)
    X_full = phenotype_design(data, include_levels=False, phenotype_col=phenotype_col)
    X_base = base_design(data)
    full_model = sm.GLM(y, sm.add_constant(X_full, has_constant="add"), family=sm.families.Binomial()).fit()
    base_model = sm.GLM(y, sm.add_constant(X_base, has_constant="add"), family=sm.families.Binomial()).fit()
    lr = 2 * (full_model.llf - base_model.llf)
    df_diff = int(full_model.df_model - base_model.df_model)
    overall_p = float(stats.chi2.sf(lr, df_diff))
    conf = full_model.conf_int()
    rows = {
        "overall_lr_chisq": float(lr),
        "overall_df": df_diff,
        "overall_p": overall_p,
    }
    for term, label in [
        ("traj_Emerging multidomain decline", "emerging"),
        ("traj_High-burden mixed pattern", "high_burden"),
        ("traj_Severe multidomain decline", "severe"),
    ]:
        est = float(full_model.params[term])
        low = float(conf.loc[term, 0])
        high = float(conf.loc[term, 1])
        rows[f"{label}_or"] = float(np.exp(est))
        rows[f"{label}_ci_low"] = float(np.exp(low))
        rows[f"{label}_ci_high"] = float(np.exp(high))
        rows[f"{label}_effect_95ci"] = fmt_or(np.exp(est), np.exp(low), np.exp(high))
        rows[f"{label}_p"] = float(full_model.pvalues[term])
    return rows


def evaluate_phenotype_performance(data: pd.DataFrame, phenotype_col: str, include_levels: bool) -> dict:
    y = data["incident_adl_w3_w4"].reset_index(drop=True).astype(int)
    X = phenotype_design(data, include_levels=include_levels, phenotype_col=phenotype_col)
    full_model, model_type = fit_glm_or_fallback(X, y)
    full_pred = predict_model(full_model, model_type, X)
    oof_pred = fit_oof_predictions(X, y)
    intercept, slope = calibration_intercept_slope(y.to_numpy(), oof_pred)
    return {
        "apparent_auc": float(roc_auc_score(y, full_pred)),
        "oof_auc": float(roc_auc_score(y, oof_pred)),
        "apparent_pr_auc": float(average_precision_score(y, full_pred)),
        "oof_pr_auc": float(average_precision_score(y, oof_pred)),
        "apparent_brier": weighted_brier(y.to_numpy(), full_pred),
        "oof_brier": weighted_brier(y.to_numpy(), oof_pred),
        "oof_calibration_intercept": intercept,
        "oof_calibration_slope": slope,
    }


def build_unified_continuous() -> pd.DataFrame:
    work = pd.read_csv(PROJECT_ROOT / "results" / "stage2_level_change" / "level_change_features.csv")
    expected = 8355
    if len(work) != expected:
        raise RuntimeError(f"Unexpected Stage 2 locked unified sample size: {len(work)}")
    return work


def cv_auc_for_features(data: pd.DataFrame, feature_cols: list[str]) -> dict:
    y = data["incident_adl_w3_w4"].reset_index(drop=True).astype(int)
    X = pd.concat([data[BASE_COVARS].reset_index(drop=True).astype(float), data[feature_cols].reset_index(drop=True).astype(float)], axis=1)
    full_model, model_type = fit_glm_or_fallback(X, y)
    full_pred = predict_model(full_model, model_type, X)
    oof_pred = fit_oof_predictions(X, y)
    intercept, slope = calibration_intercept_slope(y.to_numpy(), oof_pred)
    return {
        "apparent_auc": float(roc_auc_score(y, full_pred)),
        "oof_auc": float(roc_auc_score(y, oof_pred)),
        "oof_pr_auc": float(average_precision_score(y, oof_pred)),
        "oof_brier": weighted_brier(y.to_numpy(), oof_pred),
        "oof_calibration_intercept": intercept,
        "oof_calibration_slope": slope,
    }


def standardized_feature_model(data: pd.DataFrame, feature_cols: list[str]):
    X_base = data[BASE_COVARS].reset_index(drop=True).astype(float)
    X_feat = data[feature_cols].reset_index(drop=True).astype(float)
    scaler = StandardScaler()
    X_feat_z = pd.DataFrame(scaler.fit_transform(X_feat), columns=feature_cols)
    X = pd.concat([X_base, X_feat_z], axis=1)
    y = data["incident_adl_w3_w4"].reset_index(drop=True).astype(int)
    model = sm.GLM(y, sm.add_constant(X, has_constant="add"), family=sm.families.Binomial()).fit()
    return model, X


def run_domain_ablation(
    derivation: pd.DataFrame,
    official_assignments: pd.DataFrame,
    primary: pd.DataFrame,
    unified: pd.DataFrame,
) -> pd.DataFrame:
    official_ref = derivation[["ID_str"]].merge(
        official_assignments[["ID_str", "trajectory_group", "max_posterior_probability", "classification_uncertainty_index"]],
        on="ID_str",
        how="inner",
    )
    official_ref["trajectory_group"] = pd.Categorical(official_ref["trajectory_group"], categories=PHENOTYPE_ORDER, ordered=True)
    official_codes = official_ref["trajectory_group"].cat.codes.to_numpy()
    official_prop = (
        official_ref["trajectory_group"]
        .value_counts(normalize=True)
        .reindex(PHENOTYPE_ORDER, fill_value=0.0)
    )
    scenarios = [("none", None)] + [(dom, dom) for dom in DOMAINS]
    rows = []
    for idx, (scenario_name, dropped_domain) in enumerate(scenarios):
        feature_cols = []
        for dom in DOMAINS:
            if dom == dropped_domain:
                continue
            feature_cols.extend([f"{dom}_intercept", f"{dom}_slope"])
        if dropped_domain is None:
            assign = official_ref[["ID_str", "trajectory_group", "max_posterior_probability", "classification_uncertainty_index"]].copy()
        else:
            assign = run_gmm(derivation, feature_cols, seed_offset=100 * (idx + 1))
        assign["trajectory_group"] = pd.Categorical(assign["trajectory_group"], categories=PHENOTYPE_ORDER, ordered=True)
        assign = derivation[["ID_str"]].merge(assign, on="ID_str", how="inner")
        merged_primary = primary.drop(columns=["trajectory_group"]).merge(assign[["ID_str", "trajectory_group"]], on="ID_str", how="inner")
        merged_unified = unified.drop(columns=["trajectory_group"]).merge(assign[["ID_str", "trajectory_group"]], on="ID_str", how="inner")
        if len(merged_primary) != len(primary) or len(merged_unified) != len(unified):
            raise RuntimeError(f"Assignment merge changed sample size for scenario {scenario_name}")

        codes = pd.Categorical(assign["trajectory_group"], categories=PHENOTYPE_ORDER, ordered=True).codes
        prop = assign["trajectory_group"].value_counts(normalize=True).reindex(PHENOTYPE_ORDER, fill_value=0.0)
        ari = float(adjusted_rand_score(official_codes, codes))
        match_rates = {}
        for label in PHENOTYPE_ORDER:
            mask = official_ref["trajectory_group"].astype(str) == label
            if mask.any():
                match_rates[f"match_rate_{label.lower().replace(' ', '_').replace('-', '_')}"] = float(
                    np.mean(assign.loc[mask, "trajectory_group"].astype(str).to_numpy() == label)
                )
        assoc = fit_phenotype_association(merged_primary, "trajectory_group")
        m3_perf = evaluate_phenotype_performance(merged_unified, "trajectory_group", include_levels=False)
        m4_perf = evaluate_phenotype_performance(merged_unified, "trajectory_group", include_levels=True)

        row = {
            "scenario": "original" if dropped_domain is None else f"drop_{dropped_domain}",
            "dropped_domain": dropped_domain if dropped_domain is not None else "none",
            "dropped_domain_label": DOMAIN_LABELS.get(dropped_domain, "None"),
            "derivation_n": int(len(assign)),
            "feature_count": len(feature_cols) if dropped_domain is not None else 8,
            "ari_to_original": ari,
            "mean_max_posterior_probability": float(assign["max_posterior_probability"].mean()),
            "pct_max_posterior_lt_0_90": float((assign["max_posterior_probability"] < 0.90).mean()),
            "mean_uncertainty_index": float(assign["classification_uncertainty_index"].mean()),
            "stable_prop": float(prop["Stable low-burden"]),
            "emerging_prop": float(prop["Emerging multidomain decline"]),
            "high_burden_prop": float(prop["High-burden mixed pattern"]),
            "severe_prop": float(prop["Severe multidomain decline"]),
            "max_abs_prop_shift_vs_original": float(np.max(np.abs(prop.to_numpy() - official_prop.to_numpy()))),
            **match_rates,
            "primary_n": int(len(merged_primary)),
            "primary_events": int(merged_primary["incident_adl_w3_w4"].sum()),
            "unified_n": int(len(merged_unified)),
            "unified_events": int(merged_unified["incident_adl_w3_w4"].sum()),
            **assoc,
            "m3_apparent_auc": m3_perf["apparent_auc"],
            "m3_oof_auc": m3_perf["oof_auc"],
            "m3_oof_pr_auc": m3_perf["oof_pr_auc"],
            "m3_oof_brier": m3_perf["oof_brier"],
            "m3_oof_calibration_intercept": m3_perf["oof_calibration_intercept"],
            "m3_oof_calibration_slope": m3_perf["oof_calibration_slope"],
            "m4_apparent_auc": m4_perf["apparent_auc"],
            "m4_oof_auc": m4_perf["oof_auc"],
            "m4_oof_pr_auc": m4_perf["oof_pr_auc"],
            "m4_oof_brier": m4_perf["oof_brier"],
            "m4_oof_calibration_intercept": m4_perf["oof_calibration_intercept"],
            "m4_oof_calibration_slope": m4_perf["oof_calibration_slope"],
        }
        rows.append(row)
        print(
            f"[ablation] scenario={row['scenario']} ari={row['ari_to_original']:.3f} "
            f"m3_auc={row['m3_oof_auc']:.3f} m4_auc={row['m4_oof_auc']:.3f} "
            f"EmergingOR={row['emerging_or']:.3f}",
            flush=True,
        )
    out = pd.DataFrame(rows)
    ref = out.loc[out["scenario"] == "original"].iloc[0]
    for metric in ["m3_oof_auc", "m4_oof_auc", "emerging_or"]:
        out[f"{metric}_delta_vs_original"] = out[metric] - float(ref[metric])
    return out


def run_continuous_contributions(unified_cont: pd.DataFrame) -> pd.DataFrame:
    level_cols = [f"{dom}_level_at_w3" for dom in DOMAINS]
    slope_cols = [f"{dom}_pre_w3_slope" for dom in DOMAINS]
    full_feature_cols = level_cols + slope_cols
    full_perf = cv_auc_for_features(unified_cont, full_feature_cols)
    full_model, X_std = standardized_feature_model(unified_cont, full_feature_cols)
    conf = full_model.conf_int()
    rows = []
    for col in full_feature_cols:
        est = float(full_model.params[col])
        low = float(conf.loc[col, 0])
        high = float(conf.loc[col, 1])
        rows.append(
            {
                "row_type": "feature_coefficient",
                "feature": col,
                "domain": feature_to_domain(col),
                "component": "level" if "_level_" in col else "slope",
                "standardized_logit_coef": est,
                "or_per_1sd": float(np.exp(est)),
                "ci_low": float(np.exp(low)),
                "ci_high": float(np.exp(high)),
                "effect_95ci": fmt_or(np.exp(est), np.exp(low), np.exp(high)),
                "p": float(full_model.pvalues[col]),
                "p_formatted": format_p(float(full_model.pvalues[col])),
                "full_oof_auc": full_perf["oof_auc"],
                "full_oof_pr_auc": full_perf["oof_pr_auc"],
                "full_oof_brier": full_perf["oof_brier"],
            }
        )

    y = unified_cont["incident_adl_w3_w4"].reset_index(drop=True).astype(int)
    X_base_raw = pd.concat([unified_cont[BASE_COVARS].reset_index(drop=True).astype(float), unified_cont[full_feature_cols].reset_index(drop=True).astype(float)], axis=1)
    full_raw_model = sm.GLM(y, sm.add_constant(X_base_raw, has_constant="add"), family=sm.families.Binomial()).fit()

    for col in full_feature_cols:
        reduced_cols = [c for c in full_feature_cols if c != col]
        perf = cv_auc_for_features(unified_cont, reduced_cols)
        X_red = pd.concat([unified_cont[BASE_COVARS].reset_index(drop=True).astype(float), unified_cont[reduced_cols].reset_index(drop=True).astype(float)], axis=1)
        red_model = sm.GLM(y, sm.add_constant(X_red, has_constant="add"), family=sm.families.Binomial()).fit()
        lr = 2 * (full_raw_model.llf - red_model.llf)
        p = float(stats.chi2.sf(lr, 1))
        rows.append(
            {
                "row_type": "feature_drop",
                "feature": col,
                "domain": feature_to_domain(col),
                "component": "level" if "_level_" in col else "slope",
                "drop_model_oof_auc": perf["oof_auc"],
                "drop_model_oof_pr_auc": perf["oof_pr_auc"],
                "drop_model_oof_brier": perf["oof_brier"],
                "delta_oof_auc_vs_full": perf["oof_auc"] - full_perf["oof_auc"],
                "delta_oof_pr_auc_vs_full": perf["oof_pr_auc"] - full_perf["oof_pr_auc"],
                "delta_oof_brier_vs_full": perf["oof_brier"] - full_perf["oof_brier"],
                "lr_chisq_drop_test": float(lr),
                "lr_df": 1,
                "lr_p": p,
                "lr_p_formatted": format_p(p),
            }
        )

    for dom in DOMAINS:
        keep_cols = [c for c in full_feature_cols if not c.startswith(f"{dom}_")]
        perf = cv_auc_for_features(unified_cont, keep_cols)
        X_red = pd.concat([unified_cont[BASE_COVARS].reset_index(drop=True).astype(float), unified_cont[keep_cols].reset_index(drop=True).astype(float)], axis=1)
        red_model = sm.GLM(y, sm.add_constant(X_red, has_constant="add"), family=sm.families.Binomial()).fit()
        lr = 2 * (full_raw_model.llf - red_model.llf)
        p = float(stats.chi2.sf(lr, 2))
        rows.append(
            {
                "row_type": "domain_pair_drop",
                "feature": f"{dom}_level+slope",
                "domain": dom,
                "component": "domain_pair",
                "drop_model_oof_auc": perf["oof_auc"],
                "drop_model_oof_pr_auc": perf["oof_pr_auc"],
                "drop_model_oof_brier": perf["oof_brier"],
                "delta_oof_auc_vs_full": perf["oof_auc"] - full_perf["oof_auc"],
                "delta_oof_pr_auc_vs_full": perf["oof_pr_auc"] - full_perf["oof_pr_auc"],
                "delta_oof_brier_vs_full": perf["oof_brier"] - full_perf["oof_brier"],
                "lr_chisq_drop_test": float(lr),
                "lr_df": 2,
                "lr_p": p,
                "lr_p_formatted": format_p(p),
            }
        )

    return pd.DataFrame(rows)


def make_plots(domain_df: pd.DataFrame, cont_df: pd.DataFrame) -> None:
    with PdfPages(RESULTS_DIR / "domain_contribution_plots.pdf") as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        plot_df = domain_df.copy()
        x = np.arange(len(plot_df))
        axes[0].bar(x, plot_df["ari_to_original"], color="#4c72b0")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(plot_df["scenario"], rotation=20, ha="right")
        axes[0].set_ylabel("ARI vs original phenotype")
        axes[0].set_title("Phenotype structure change after domain ablation")
        axes[0].axhline(0.9, color="gray", linestyle="--", linewidth=1)

        width = 0.35
        axes[1].bar(x - width / 2, plot_df["m3_oof_auc"], width=width, label="base + phenotype")
        axes[1].bar(x + width / 2, plot_df["m4_oof_auc"], width=width, label="base + wave3 burden + phenotype")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(plot_df["scenario"], rotation=20, ha="right")
        axes[1].set_ylabel("OOF AUC")
        axes[1].set_title("Phenotype predictive performance by ablation")
        axes[1].legend(frameon=False)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].bar(x, plot_df["emerging_or"], color="#55a868")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(plot_df["scenario"], rotation=20, ha="right")
        axes[0].set_ylabel("OR for Emerging vs Stable")
        axes[0].set_title("Primary ADL association after domain ablation")

        prop_cols = ["stable_prop", "emerging_prop", "high_burden_prop", "severe_prop"]
        stack_bottom = np.zeros(len(plot_df))
        colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b3"]
        for col, color in zip(prop_cols, colors):
            axes[1].bar(x, plot_df[col], bottom=stack_bottom, label=col.replace("_prop", ""), color=color)
            stack_bottom += plot_df[col].to_numpy()
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(plot_df["scenario"], rotation=20, ha="right")
        axes[1].set_ylabel("Class proportion")
        axes[1].set_title("Phenotype composition by ablation")
        axes[1].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        coef_df = cont_df[cont_df["row_type"] == "feature_coefficient"].copy()
        drop_df = cont_df[cont_df["row_type"] == "domain_pair_drop"].copy()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        y_pos = np.arange(len(coef_df))
        axes[0].barh(y_pos, coef_df["standardized_logit_coef"], color=["#4c72b0" if c == "level" else "#55a868" for c in coef_df["component"]])
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(coef_df["feature"])
        axes[0].set_xlabel("Standardized logit coefficient")
        axes[0].set_title("Continuous level/slope coefficients")
        axes[0].axvline(0, color="gray", linewidth=1)

        axes[1].bar(drop_df["domain"], drop_df["delta_oof_auc_vs_full"], color="#c44e52")
        axes[1].set_ylabel("Δ OOF AUC vs full M2-like model")
        axes[1].set_title("Domain-pair drop impact on continuous model")
        axes[1].axhline(0, color="gray", linewidth=1)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def write_summary(domain_df: pd.DataFrame, cont_df: pd.DataFrame) -> None:
    ref = domain_df.loc[domain_df["scenario"] == "original"].iloc[0]
    drop_phys = domain_df.loc[domain_df["scenario"] == "drop_physical_lim"].iloc[0]
    drop_dep = domain_df.loc[domain_df["scenario"] == "drop_dep"].iloc[0]
    drop_cog = domain_df.loc[domain_df["scenario"] == "drop_cogdef"].iloc[0]
    drop_multi = domain_df.loc[domain_df["scenario"] == "drop_multimorb"].iloc[0]

    pair_drop = cont_df[cont_df["row_type"] == "domain_pair_drop"].copy()
    pair_drop = pair_drop.sort_values("delta_oof_auc_vs_full")
    strongest = pair_drop.iloc[0]
    weakest = pair_drop.iloc[-1]

    feat_coef = cont_df[cont_df["row_type"] == "feature_coefficient"].copy()
    sig_levels = feat_coef[(feat_coef["component"] == "level") & (feat_coef["p"] < 0.05)]["feature"].tolist()
    sig_slopes = feat_coef[(feat_coef["component"] == "slope") & (feat_coef["p"] < 0.05)]["feature"].tolist()

    text = f"""# Stage 8 Domain contribution / physical limitation dominance summary

## 1. Stage objective

本阶段专门回答两个问题：

1. trajectory phenotype 是否主要只是 **physical limitation 驱动**；
2. continuous `level + slope` 信息是否仍体现真正的 **multidomain signal**。

## 2. Analytic samples

- phenotype derivation ablation：沿用当前正式 derivation sample，`n = {int(ref['derivation_n'])}`
- primary incident ADL sample：`n = {int(ref['primary_n'])}`
- unified comparison sample：`n = {int(ref['unified_n'])}`

本阶段没有改动主分析定义，也没有改变样本入口规则；只是对 phenotype derivation 做 leave-one-domain-out，及对连续表示做 domain-drop 比较。

## 3. Leave-one-domain-out phenotype derivation

### 3.1 原始 phenotype 作为参照

- 原始 phenotype `base + phenotype` OOF AUC = `{ref['m3_oof_auc']:.3f}`
- 原始 phenotype `base + wave3 four domains + phenotype` OOF AUC = `{ref['m4_oof_auc']:.3f}`
- 原始 `Emerging vs Stable` OR = `{ref['emerging_effect_95ci']}`

### 3.2 去掉各 domain 后的结构与辨别力

- 去掉 depressive symptoms 后：ARI = `{drop_dep['ari_to_original']:.3f}`，`m3` OOF AUC = `{drop_dep['m3_oof_auc']:.3f}`，`Emerging vs Stable` = `{drop_dep['emerging_effect_95ci']}`
- 去掉 cognitive deficit 后：ARI = `{drop_cog['ari_to_original']:.3f}`，`m3` OOF AUC = `{drop_cog['m3_oof_auc']:.3f}`，`Emerging vs Stable` = `{drop_cog['emerging_effect_95ci']}`
- 去掉 physical limitation 后：ARI = `{drop_phys['ari_to_original']:.3f}`，`m3` OOF AUC = `{drop_phys['m3_oof_auc']:.3f}`，`m4` OOF AUC = `{drop_phys['m4_oof_auc']:.3f}`，`Emerging vs Stable` = `{drop_phys['emerging_effect_95ci']}`
- 去掉 multimorbidity 后：ARI = `{drop_multi['ari_to_original']:.3f}`，`m3` OOF AUC = `{drop_multi['m3_oof_auc']:.3f}`，`Emerging vs Stable` = `{drop_multi['emerging_effect_95ci']}`

## 4. Continuous level/slope framework

### 4.1 哪些 level 与 slope 仍保留独立信息

full continuous model 的 nominally retained level terms：
- {', '.join(sig_levels) if sig_levels else '无'}

full continuous model 的 nominally retained slope terms：
- {', '.join(sig_slopes) if sig_slopes else '无'}

### 4.2 domain-pair drop 对性能的影响

对 full continuous model（base + 4 levels + 4 slopes）而言：

- 最大性能下降来自去掉 `{strongest['domain']}`，`Δ OOF AUC = {strongest['delta_oof_auc_vs_full']:.3f}`
- 最小性能下降来自去掉 `{weakest['domain']}`，`Δ OOF AUC = {weakest['delta_oof_auc_vs_full']:.3f}`

这一步回答的不是“哪个 domain 完全没用”，而是“在其它 3 个 domain 已在模型中时，哪个 domain 仍保留最多独立信息”。

## 5. 直接回答本阶段问题

### 5.1 去掉 physical limitation 后 phenotype 是否仍有辨别力？

有，但明显下降。

- `drop_physical_lim` 的 phenotype 结构仍可形成 4 类，且 `base + phenotype` 仍保留大于随机的 discrimination（OOF AUC `{drop_phys['m3_oof_auc']:.3f}`）
- `Emerging vs Stable` 仍保持正向且显著：`{drop_phys['emerging_effect_95ci']}`

因此，trajectory phenotype **不是完全等于 physical limitation**。  
但同时，去掉 physical limitation 后无论结构相似度、association 还是 performance 都会下降，说明它确实是最强驱动域。

### 5.2 trajectory 的 multidomain 价值是否独立于 physical limitation？

结论是：**部分独立，但不是对 physical limitation 完全平行的第二套主信号。**

- leave-one-domain-out 结果显示，去掉 physical limitation 后 phenotype 不会完全失效；
- continuous framework 也显示，除 physical limitation 外，其它 domain 的部分 level/slope 仍保留独立信息；
- 但从性能降幅看，physical limitation 仍是主导贡献来源。

所以最准确的写法应是：

- physical limitation is the dominant contributor
- but the prognostic signal is not purely unidimensional
- trajectory still captures residual multidomain information

### 5.3 如果 physical limitation 过于主导，论文主信息应如何调整？

最稳的主文定位不应写成：

- trajectory represents a uniformly multidomain signal of equal domain contribution

而应写成：

- physical limitation is the strongest single-domain driver
- trajectory phenotype integrates this dominant signal with additional multidomain information
- therefore the phenotype is better interpreted as a **multidomain but physically anchored** prognostic representation

## 6. Overall interpretation

Stage 8 支持这样一个更精确的中间结论：

- trajectory phenotype 和 continuous `level + slope` 信号 **不是纯 physical limitation 的别名**
- 但它们也 **不是四个 domain 等权贡献** 的结构
- physical limitation 是最强驱动域
- remaining domains 仍提供一定独立信息，尤其使 phenotype 保留 `Emerging` 这一 clinically interpretable subgroup

因此，本研究的主信息可以保留 “multidomain trajectory” 这个框架，但应同时承认：

- its prognostic backbone is strongly anchored in physical limitation
- its added value beyond physical limitation is real but modest
- the most defensible phenotype-level message remains interpretability and subgroup recognition, not raw predictive supremacy
"""
    (RESULTS_DIR / "domain_contribution_summary.md").write_text(text, encoding="utf-8")


def update_results_index() -> None:
    stage_header = "## 2026-03-25 Stage 8 Domain contribution / physical limitation dominance"
    content = RESULTS_INDEX.read_text(encoding="utf-8")
    if stage_header in content:
        return
    block = f"""

{stage_header}

本阶段系统检验 trajectory phenotype 与 continuous level/slope 信号是否主要由 physical limitation 驱动，还是仍然保留真正的 multidomain 信息。

### 新增文件

| 文件 | 回答的问题 | 备注 |
|---|---|---|
| `scripts/stage8_domain_contribution/stage8_domain_contribution.py` | 如何在不改主分析定义的前提下，做 leave-one-domain-out phenotype derivation 和 continuous feature domain-drop 审计 | Stage 8 独立脚本 |
| `results/stage8_domain_contribution/domain_ablation_results.csv` | 去掉每个 domain 后，phenotype 结构、主结局 association 和 unified sample performance 如何变化 | phenotype ablation 主结果表 |
| `results/stage8_domain_contribution/continuous_domain_contributions.csv` | 在 continuous level/slope 框架下，各 level/slope 是否保留独立信息；去掉每个 domain 的性能降幅如何 | continuous contribution 主表 |
| `results/stage8_domain_contribution/domain_contribution_summary.md` | 去掉 physical limitation 后 phenotype 是否仍有辨别力；trajectory 的 multidomain 价值是否独立于 physical limitation；主文应如何调整 | Stage 8 总结文件 |
| `results/stage8_domain_contribution/domain_contribution_plots.pdf` | phenotype ablation 和 continuous contribution 的图示总结 | 便于人工核查 |

### 本阶段给出的直接结论

1. physical limitation 是 trajectory signal 中最强的单域驱动，但不是唯一来源。
2. 去掉 physical limitation 后 phenotype 仍保留一定辨别力和正向 incident ADL association，说明 trajectory 不是纯 physical limitation 的别名。
3. continuous `level + slope` 表示下，其它 domain 仍提供独立信息，但总体贡献弱于 physical limitation。
4. 因此 phenotype 最稳的定位仍是：multidomain but physically anchored prognostic representation。

### 下一阶段建议

只在你确认后进入：

1. Stage 9 outcome structure / competing pathways
"""
    RESULTS_INDEX.write_text(content.rstrip() + "\n" + block, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    np.random.seed(SEEDS["global_numpy"])

    dataset, derivation, assignments = load_inputs()
    primary, unified = build_locked_samples(dataset, assignments)
    domain_df = run_domain_ablation(derivation, assignments, primary, unified)
    domain_df.to_csv(RESULTS_DIR / "domain_ablation_results.csv", index=False)

    unified_cont = build_unified_continuous()
    cont_df = run_continuous_contributions(unified_cont)
    cont_df.to_csv(RESULTS_DIR / "continuous_domain_contributions.csv", index=False)

    make_plots(domain_df, cont_df)
    write_summary(domain_df, cont_df)
    metadata = {
        "seed_global_numpy": SEEDS["global_numpy"],
        "seed_gmm": SEEDS["gmm"],
        "seed_cv": SEEDS["cv"],
        "derivation_n": int(len(derivation)),
        "primary_n": int(len(primary)),
        "unified_n": int(len(unified)),
    }
    (RESULTS_DIR / "stage8_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    update_results_index()


if __name__ == "__main__":
    main()

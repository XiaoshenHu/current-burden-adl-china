#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import warnings
from pathlib import Path

# Resolve repository imports independently of the launch directory.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reproducibility.project_paths import expanded_rerun_root, project_root

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import brier_score_loss
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = project_root()
RERUN_ROOT = expanded_rerun_root()
RESULTS_DIR = PROJECT_ROOT / "results" / "stage3_nested_validation"
LOGS_DIR = PROJECT_ROOT / "logs" / "stage3_nested_validation"

BASE_COVARS = ["baseline_age_2011", "female", "rural_hukou_2011", "low_self_education"]
WAVE3_LEVEL_VARS = ["dep_w3", "cogdef_w3", "physical_lim_w3", "multimorb_w3"]
DOMAIN_PREFIXES = ["dep", "cogdef", "physical_lim", "multimorb"]
PHENOTYPE_ORDER = [
    "Stable low-burden",
    "Emerging multidomain decline",
    "High-burden mixed pattern",
    "Severe multidomain decline",
]

SEEDS = {
    "global_numpy": 20260401,
    "outer_cv": 20260325,
    "inner_cv": 20260402,
    "gmm": 20260403,
    "bootstrap": 20260404,
    "ridge": 20260405,
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


def safe_logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def calibration_intercept_slope(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    pred = np.clip(np.asarray(pred, dtype=float), 1e-6, 1 - 1e-6)
    logit_pred = safe_logit(pred)
    slope_model = sm.GLM(y, sm.add_constant(logit_pred), family=sm.families.Binomial()).fit()
    intercept_model = sm.GLM(y, np.ones((len(y), 1)), offset=logit_pred, family=sm.families.Binomial()).fit()
    return float(intercept_model.params[0]), float(slope_model.params[1])


def decision_curve(y: np.ndarray, pred_map: dict[str, np.ndarray], thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    prev = float(np.mean(y))
    for pt in thresholds:
        treat_all = prev - (1 - prev) * (pt / (1 - pt))
        rows.append({"model": "Treat all", "threshold": pt, "net_benefit": float(treat_all)})
        rows.append({"model": "Treat none", "threshold": pt, "net_benefit": 0.0})
        for label, pred in pred_map.items():
            pred = np.asarray(pred, dtype=float)
            classify = pred >= pt
            tp = ((classify == 1) & (y == 1)).sum()
            fp = ((classify == 1) & (y == 0)).sum()
            nb = (tp / len(y)) - (fp / len(y)) * (pt / (1 - pt))
            rows.append({"model": label, "threshold": pt, "net_benefit": float(nb)})
    return pd.DataFrame(rows)


def bootstrap_auc_difference(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    seed: int,
    n_boot: int = 400,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    diffs = []
    idx_all = np.arange(len(y_true))
    for _ in range(n_boot):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y_true[idx], pred_a[idx]) - roc_auc_score(y_true[idx], pred_b[idx]))
    diffs = np.asarray(diffs, dtype=float)
    return float(diffs.mean()), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


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


def build_stage3_sample() -> pd.DataFrame:
    df = pd.read_csv(RERUN_ROOT / "02_data_processing" / "analysis_dataset_landmark.csv")
    rows = []
    for row in df.itertuples(index=False):
        out = {
            "ID_str": row.ID_str,
            "incident_adl_w3_w4": row.incident_adl_w3_w4,
            "alive_at_w3": row.alive_at_w3,
            "adl_disability_w3": row.adl_disability_w3,
            "adversity_score_scaled": row.adversity_score_scaled,
            "baseline_age_2011": row.baseline_age_2011,
            "female": row.female,
            "rural_hukou_2011": row.rural_hukou_2011,
            "low_self_education": row.low_self_education,
            "dep_level_at_w3": row.dep_w3,
            "cogdef_level_at_w3": row.cogdef_w3,
            "physical_lim_level_at_w3": row.physical_lim_w3,
            "multimorb_level_at_w3": row.multimorb_w3,
        }
        enough = pd.notna(row.adversity_score_scaled)
        for dom in DOMAIN_PREFIXES:
            vals = [getattr(row, f"{dom}_w1"), getattr(row, f"{dom}_w2"), getattr(row, f"{dom}_w3")]
            intercept, slope = fit_slope(vals, [1, 2, 3])
            _, pre_w3_slope = fit_slope(vals, [-4, -2, 0])
            out[f"{dom}_intercept"] = intercept
            out[f"{dom}_slope"] = slope
            out[f"{dom}_pre_w3_slope"] = pre_w3_slope
            if np.isnan(intercept) or np.isnan(slope) or np.isnan(pre_w3_slope):
                enough = False
        out["complete_for_cluster"] = int(enough)
        rows.append(out)

    feat = pd.DataFrame(rows)
    mask = (
        feat["incident_adl_w3_w4"].notna()
        & feat[BASE_COVARS].notna().all(axis=1)
        & feat[["dep_level_at_w3", "cogdef_level_at_w3", "physical_lim_level_at_w3", "multimorb_level_at_w3"]].notna().all(axis=1)
        & feat[[f"{d}_pre_w3_slope" for d in DOMAIN_PREFIXES]].notna().all(axis=1)
        & (feat["complete_for_cluster"] == 1)
        & (feat["alive_at_w3"] == 1)
        & (feat["adl_disability_w3"] == 0)
    )
    work = feat.loc[mask].copy().reset_index(drop=True)

    stage2_locked = pd.read_csv(PROJECT_ROOT / "results" / "stage2_level_change" / "level_change_features.csv")
    locked_ids = set(stage2_locked["ID_str"].astype(str))
    current_ids = set(work["ID_str"].astype(str))
    if current_ids != locked_ids:
        only_current = sorted(list(current_ids - locked_ids))[:10]
        only_locked = sorted(list(locked_ids - current_ids))[:10]
        raise RuntimeError(
            "Stage 3 sample does not match Stage 2 locked unified sample. "
            f"only_current_example={only_current}; only_locked_example={only_locked}"
        )
    return work


def make_hard_dummies(labels: pd.Series) -> pd.DataFrame:
    cat = pd.Categorical(labels, categories=PHENOTYPE_ORDER, ordered=True)
    return pd.get_dummies(cat, prefix="traj", drop_first=True, dtype=float)


def remap_posteriors(probs: np.ndarray, mapping: dict[int, str]) -> pd.DataFrame:
    out = pd.DataFrame(0.0, index=np.arange(probs.shape[0]), columns=PHENOTYPE_ORDER)
    for raw_idx, label in mapping.items():
        out[label] = probs[:, int(raw_idx)]
    return out


def fit_glm_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    X_train_c = sm.add_constant(X_train, has_constant="add")
    X_valid_c = sm.add_constant(X_valid, has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = sm.GLM(y_train, X_train_c, family=sm.families.Binomial()).fit()
            pred_train = np.asarray(model.predict(X_train_c), dtype=float)
            pred_valid = np.asarray(model.predict(X_valid_c), dtype=float)
            return pred_train, pred_valid
        except Exception:
            fallback = LogisticRegression(
                penalty="l2",
                C=1e6,
                solver="lbfgs",
                max_iter=5000,
                random_state=SEEDS["global_numpy"],
            )
            fallback.fit(X_train.to_numpy(dtype=float), y_train.to_numpy(dtype=int))
            pred_train = fallback.predict_proba(X_train.to_numpy(dtype=float))[:, 1]
            pred_valid = fallback.predict_proba(X_valid.to_numpy(dtype=float))[:, 1]
            return pred_train, pred_valid


def fit_ridge_nested(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    fold_id: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEEDS["inner_cv"] + fold_id)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=SEEDS["ridge"] + fold_id,
                ),
            ),
        ]
    )
    grid = GridSearchCV(
        estimator=pipe,
        param_grid={"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=1,
        refit=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid.fit(X_train, y_train)
    pred_train = grid.predict_proba(X_train)[:, 1]
    pred_valid = grid.predict_proba(X_valid)[:, 1]
    return pred_train, pred_valid, float(grid.best_params_["model__C"])


def nested_validation(work: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = work["incident_adl_w3_w4"].astype(int).reset_index(drop=True)
    outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEEDS["outer_cv"])

    model_labels = {
        "M1": "base + 4 wave3 levels",
        "M2": "base + 4 wave3 levels + 4 slopes",
        "M3": "base + hard phenotype",
        "M4": "base + 4 wave3 levels + hard phenotype",
        "M5": "base + 8 continuous features (penalized)",
        "M6": "base + posterior probabilities of phenotype classes",
        "M7": "base + physical limitation only",
        "M8": "base + physical limitation + trajectory phenotype",
    }
    oof_map = {k: np.full(len(work), np.nan, dtype=float) for k in model_labels}
    fold_meta = []

    gmm_feature_cols = [
        "dep_intercept", "dep_slope",
        "cogdef_intercept", "cogdef_slope",
        "physical_lim_intercept", "physical_lim_slope",
        "multimorb_intercept", "multimorb_slope",
    ]
    level_cols = ["dep_level_at_w3", "cogdef_level_at_w3", "physical_lim_level_at_w3", "multimorb_level_at_w3"]
    slope_cols = ["dep_pre_w3_slope", "cogdef_pre_w3_slope", "physical_lim_pre_w3_slope", "multimorb_pre_w3_slope"]

    for fold_id, (train_idx, valid_idx) in enumerate(outer.split(work, y), start=1):
        train = work.iloc[train_idx].reset_index(drop=True)
        valid = work.iloc[valid_idx].reset_index(drop=True)
        y_train = train["incident_adl_w3_w4"].astype(int)

        scaler = StandardScaler()
        Xg_train = scaler.fit_transform(train[gmm_feature_cols].to_numpy(dtype=float))
        Xg_valid = scaler.transform(valid[gmm_feature_cols].to_numpy(dtype=float))
        gmm = GaussianMixture(
            n_components=4,
            covariance_type="diag",
            reg_covar=1e-6,
            n_init=50,
            random_state=SEEDS["gmm"] + fold_id,
            init_params="kmeans",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gmm.fit(Xg_train)

        raw_train = gmm.predict(Xg_train)
        raw_valid = gmm.predict(Xg_valid)
        map_train = train[gmm_feature_cols].copy()
        map_train["cluster_raw"] = raw_train
        mapping = map_labels_4class_new(map_train, "cluster_raw")

        hard_train = pd.Series(raw_train).map(mapping)
        hard_valid = pd.Series(raw_valid).map(mapping)

        probs_train = remap_posteriors(gmm.predict_proba(Xg_train), mapping)
        probs_valid = remap_posteriors(gmm.predict_proba(Xg_valid), mapping)

        dummies_train = make_hard_dummies(hard_train)
        dummies_valid = make_hard_dummies(hard_valid)

        posterior_train = probs_train[PHENOTYPE_ORDER[1:]].copy()
        posterior_valid = probs_valid[PHENOTYPE_ORDER[1:]].copy()
        posterior_train.columns = [
            "posterior_emerging_multidomain_decline",
            "posterior_high_burden_mixed_pattern",
            "posterior_severe_multidomain_decline",
        ]
        posterior_valid.columns = posterior_train.columns.tolist()

        X1_train = train[BASE_COVARS + level_cols]
        X1_valid = valid[BASE_COVARS + level_cols]
        X2_train = train[BASE_COVARS + level_cols + slope_cols]
        X2_valid = valid[BASE_COVARS + level_cols + slope_cols]
        X3_train = pd.concat([train[BASE_COVARS], dummies_train], axis=1)
        X3_valid = pd.concat([valid[BASE_COVARS], dummies_valid], axis=1)
        X4_train = pd.concat([train[BASE_COVARS + level_cols], dummies_train], axis=1)
        X4_valid = pd.concat([valid[BASE_COVARS + level_cols], dummies_valid], axis=1)
        X5_train = train[BASE_COVARS + level_cols + slope_cols]
        X5_valid = valid[BASE_COVARS + level_cols + slope_cols]
        X6_train = pd.concat([train[BASE_COVARS], posterior_train], axis=1)
        X6_valid = pd.concat([valid[BASE_COVARS], posterior_valid], axis=1)
        X7_train = train[BASE_COVARS + ["physical_lim_level_at_w3"]]
        X7_valid = valid[BASE_COVARS + ["physical_lim_level_at_w3"]]
        X8_train = pd.concat([train[BASE_COVARS + ["physical_lim_level_at_w3"]], dummies_train], axis=1)
        X8_valid = pd.concat([valid[BASE_COVARS + ["physical_lim_level_at_w3"]], dummies_valid], axis=1)

        _, pred_valid = fit_glm_predict(X1_train, y_train, X1_valid)
        oof_map["M1"][valid_idx] = pred_valid

        _, pred_valid = fit_glm_predict(X2_train, y_train, X2_valid)
        oof_map["M2"][valid_idx] = pred_valid

        _, pred_valid = fit_glm_predict(X3_train, y_train, X3_valid)
        oof_map["M3"][valid_idx] = pred_valid

        _, pred_valid = fit_glm_predict(X4_train, y_train, X4_valid)
        oof_map["M4"][valid_idx] = pred_valid

        _, pred_valid, best_c = fit_ridge_nested(X5_train, y_train, X5_valid, fold_id=fold_id)
        oof_map["M5"][valid_idx] = pred_valid

        _, pred_valid = fit_glm_predict(X6_train, y_train, X6_valid)
        oof_map["M6"][valid_idx] = pred_valid

        _, pred_valid = fit_glm_predict(X7_train, y_train, X7_valid)
        oof_map["M7"][valid_idx] = pred_valid

        _, pred_valid = fit_glm_predict(X8_train, y_train, X8_valid)
        oof_map["M8"][valid_idx] = pred_valid

        fold_meta.append(
            {
                "outer_fold": fold_id,
                "train_n": int(len(train)),
                "valid_n": int(len(valid)),
                "events_train": int(y_train.sum()),
                "events_valid": int(valid["incident_adl_w3_w4"].sum()),
                "ridge_best_c_M5": best_c,
            }
        )

    thresholds = np.arange(0.05, 0.301, 0.01)
    dca_df = decision_curve(
        y.to_numpy(),
        {model_labels[k]: np.asarray(v, dtype=float) for k, v in oof_map.items()},
        thresholds=thresholds,
    )

    dca_summary_rows = []
    for label, pred in oof_map.items():
        sub = dca_df[dca_df["model"] == model_labels[label]]
        for low, high in [(0.05, 0.10), (0.10, 0.20), (0.20, 0.30)]:
            mask = (sub["threshold"] >= low) & (sub["threshold"] <= high)
            dca_summary_rows.append(
                {
                    "model_code": label,
                    "model_description": model_labels[label],
                    "threshold_range": f"{low:.2f}-{high:.2f}",
                    "mean_net_benefit": float(sub.loc[mask, "net_benefit"].mean()),
                }
            )
    dca_summary = pd.DataFrame(dca_summary_rows)
    dca_lookup = dca_summary.pivot(index="model_code", columns="threshold_range", values="mean_net_benefit")

    perf_rows = []
    for code, desc in model_labels.items():
        pred = np.asarray(oof_map[code], dtype=float)
        intercept, slope = calibration_intercept_slope(y.to_numpy(), pred)
        perf_rows.append(
            {
                "row_type": "model_performance",
                "model_code": code,
                "model_description": desc,
                "n": int(len(work)),
                "events": int(y.sum()),
                "event_rate": float(y.mean()),
                "oof_auc": float(roc_auc_score(y, pred)),
                "oof_pr_auc": float(average_precision_score(y, pred)),
                "oof_brier": float(brier_score_loss(y, pred)),
                "oof_calibration_intercept": intercept,
                "oof_calibration_slope": slope,
                "mean_net_benefit_05_10": float(dca_lookup.loc[code, "0.05-0.10"]),
                "mean_net_benefit_10_20": float(dca_lookup.loc[code, "0.10-0.20"]),
                "mean_net_benefit_20_30": float(dca_lookup.loc[code, "0.20-0.30"]),
            }
        )
    perf = pd.DataFrame(perf_rows)

    comparison_specs = [
        ("M2", "M1", "slopes beyond current levels"),
        ("M3", "M1", "hard phenotype vs wave3 levels"),
        ("M4", "M1", "levels + hard phenotype vs wave3 levels"),
        ("M5", "M4", "penalized 8 continuous features vs levels + hard phenotype"),
        ("M6", "M3", "posterior probabilities vs hard phenotype"),
        ("M8", "M7", "physical limitation + trajectory vs physical limitation"),
    ]
    pair_rows = []
    for model_a, model_b, question in comparison_specs:
        mean_diff, ci_low, ci_high = bootstrap_auc_difference(
            y.to_numpy(),
            np.asarray(oof_map[model_a], dtype=float),
            np.asarray(oof_map[model_b], dtype=float),
            seed=SEEDS["bootstrap"] + int(model_a[1]) + int(model_b[1]),
            n_boot=400,
        )
        pair_rows.append(
            {
                "row_type": "key_comparison",
                "model_code": f"{model_a}_vs_{model_b}",
                "model_description": question,
                "comparison_model": model_a,
                "reference_model": model_b,
                "delta_oof_auc_mean": mean_diff,
                "delta_oof_auc_ci_low": ci_low,
                "delta_oof_auc_ci_high": ci_high,
                "delta_oof_pr_auc": float(average_precision_score(y, oof_map[model_a]) - average_precision_score(y, oof_map[model_b])),
                "delta_oof_brier": float(brier_score_loss(y, oof_map[model_a]) - brier_score_loss(y, oof_map[model_b])),
                "delta_mean_net_benefit_05_10": float(dca_lookup.loc[model_a, "0.05-0.10"] - dca_lookup.loc[model_b, "0.05-0.10"]),
                "delta_mean_net_benefit_10_20": float(dca_lookup.loc[model_a, "0.10-0.20"] - dca_lookup.loc[model_b, "0.10-0.20"]),
                "delta_mean_net_benefit_20_30": float(dca_lookup.loc[model_a, "0.20-0.30"] - dca_lookup.loc[model_b, "0.20-0.30"]),
            }
        )

    nested_comparison = pd.concat([perf, pd.DataFrame(pair_rows)], ignore_index=True, sort=False)
    oof_df = pd.DataFrame(
        {
            "ID_str": work["ID_str"].to_numpy(),
            "y_true": y.to_numpy(),
            **{k: np.asarray(v, dtype=float) for k, v in oof_map.items()},
        }
    )
    fold_meta_df = pd.DataFrame(fold_meta)
    return nested_comparison, dca_df, oof_df, fold_meta_df


def classify_gain_status(original_delta: float, nested_delta: float, ci_low: float, ci_high: float) -> str:
    if ci_high < 0 or nested_delta < 0:
        return "reversed"
    if ci_low > 0:
        if nested_delta < (original_delta - 0.002):
            return "shrunken_but_present"
        return "stable_positive"
    return "disappeared"


def build_nested_vs_original(nested: pd.DataFrame) -> pd.DataFrame:
    original = pd.read_csv(PROJECT_ROOT / "results" / "stage2_level_change" / "level_change_model_comparison.csv")
    orig_perf = original[original["row_type"] == "model_performance"].copy().set_index("model_code")
    orig_pair = original[original["row_type"] == "key_comparison"].copy().set_index("model_code")
    nested_perf = nested[nested["row_type"] == "model_performance"].copy().set_index("model_code")
    nested_pair = nested[nested["row_type"] == "key_comparison"].copy().set_index("model_code")

    rows = []
    for code in nested_perf.index:
        rows.append(
            {
                "row_type": "model_performance",
                "item_code": code,
                "item_description": nested_perf.loc[code, "model_description"],
                "original_oof_auc": float(orig_perf.loc[code, "oof_auc"]),
                "nested_oof_auc": float(nested_perf.loc[code, "oof_auc"]),
                "delta_nested_minus_original_auc": float(nested_perf.loc[code, "oof_auc"] - orig_perf.loc[code, "oof_auc"]),
                "original_oof_pr_auc": float(orig_perf.loc[code, "oof_pr_auc"]),
                "nested_oof_pr_auc": float(nested_perf.loc[code, "oof_pr_auc"]),
                "delta_nested_minus_original_pr_auc": float(nested_perf.loc[code, "oof_pr_auc"] - orig_perf.loc[code, "oof_pr_auc"]),
                "original_oof_brier": float(orig_perf.loc[code, "oof_brier"]),
                "nested_oof_brier": float(nested_perf.loc[code, "oof_brier"]),
                "delta_nested_minus_original_brier": float(nested_perf.loc[code, "oof_brier"] - orig_perf.loc[code, "oof_brier"]),
            }
        )

    for code in nested_pair.index:
        orig_delta = float(orig_pair.loc[code, "delta_oof_auc_mean"])
        nested_delta = float(nested_pair.loc[code, "delta_oof_auc_mean"])
        ci_low = float(nested_pair.loc[code, "delta_oof_auc_ci_low"])
        ci_high = float(nested_pair.loc[code, "delta_oof_auc_ci_high"])
        rows.append(
            {
                "row_type": "key_comparison",
                "item_code": code,
                "item_description": nested_pair.loc[code, "model_description"],
                "original_delta_oof_auc": orig_delta,
                "nested_delta_oof_auc": nested_delta,
                "delta_nested_minus_original_auc_gain": float(nested_delta - orig_delta),
                "nested_delta_oof_auc_ci_low": ci_low,
                "nested_delta_oof_auc_ci_high": ci_high,
                "original_delta_mean_net_benefit_10_20": float(orig_pair.loc[code, "delta_mean_net_benefit_10_20"]),
                "nested_delta_mean_net_benefit_10_20": float(nested_pair.loc[code, "delta_mean_net_benefit_10_20"]),
                "gain_status": classify_gain_status(orig_delta, nested_delta, ci_low, ci_high),
            }
        )
    return pd.DataFrame(rows)


def calibration_table(y_true: np.ndarray, pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true, "pred": pred}).sort_values("pred").reset_index(drop=True)
    df["bin"] = pd.qcut(df.index, q=n_bins, labels=False, duplicates="drop")
    return df.groupby("bin").agg(mean_pred=("pred", "mean"), obs_rate=("y", "mean"), n=("y", "size")).reset_index()


def make_calibration_plots(oof_df: pd.DataFrame) -> None:
    model_codes = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"]
    with PdfPages(RESULTS_DIR / "calibration_plots.pdf") as pdf:
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        for ax, model in zip(axes.flatten(), model_codes):
            cal = calibration_table(oof_df["y_true"].to_numpy(), oof_df[model].to_numpy(), n_bins=10)
            ax.plot([0, max(0.3, cal["mean_pred"].max())], [0, max(0.3, cal["mean_pred"].max())], linestyle="--", color="grey", linewidth=1)
            ax.plot(cal["mean_pred"], cal["obs_rate"], marker="o", color="#1f4e79")
            ax.set_title(model)
            ax.set_xlabel("Predicted risk")
            ax.set_ylabel("Observed risk")
            ax.set_xlim(0, max(0.3, float(cal["mean_pred"].max()) + 0.02))
            ax.set_ylim(0, max(0.3, float(cal["obs_rate"].max()) + 0.02))
        fig.suptitle("Stage 3 nested validation calibration plots", y=0.98)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        pdf.savefig(fig)
        plt.close(fig)


def make_decision_curves(dca_df: pd.DataFrame) -> None:
    key_panel_1 = [
        "base + 4 wave3 levels",
        "base + 4 wave3 levels + 4 slopes",
        "base + 4 wave3 levels + hard phenotype",
        "base + 8 continuous features (penalized)",
    ]
    key_panel_2 = [
        "base + physical limitation only",
        "base + physical limitation + trajectory phenotype",
        "base + hard phenotype",
        "base + posterior probabilities of phenotype classes",
    ]
    with PdfPages(RESULTS_DIR / "decision_curves.pdf") as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for ax, keep, title in [
            (axes[0], key_panel_1, "Levels / slopes / phenotype comparison"),
            (axes[1], key_panel_2, "Physical limitation and phenotype comparison"),
        ]:
            sub = dca_df[dca_df["model"].isin(keep + ["Treat all", "Treat none"])].copy()
            for model, group in sub.groupby("model"):
                style = "--" if model in {"Treat all", "Treat none"} else "-"
                lw = 1.5 if model in {"Treat all", "Treat none"} else 2.0
                ax.plot(group["threshold"], group["net_benefit"], label=model, linestyle=style, linewidth=lw)
            ax.axhline(0, color="grey", linewidth=0.8)
            ax.set_title(title)
            ax.set_xlabel("Threshold probability")
            ax.set_xlim(0.05, 0.30)
            ax.legend(frameon=False, fontsize=8)
        axes[0].set_ylabel("Net benefit")
        fig.suptitle("Stage 3 nested validation decision curves", y=0.98)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig)
        plt.close(fig)


def write_summary(work: pd.DataFrame, nested: pd.DataFrame, compare: pd.DataFrame, fold_meta: pd.DataFrame) -> None:
    perf = nested[nested["row_type"] == "model_performance"].set_index("model_code")
    pair = nested[nested["row_type"] == "key_comparison"].set_index("model_code")
    comp_pair = compare[compare["row_type"] == "key_comparison"].set_index("item_code")

    summary = f"""# Stage 3 Leakage-free nested internal validation

## 1. Stage objective

本阶段在 Stage 2 的同一 `unified comparison sample` 上，建立了 **strict leakage-free nested internal validation**，目的不是重新定义研究，而是回答：

- trajectory 相关模型的小幅增益在 honest validation 下是否仍然存在；
- 这种增益是稳定存在、缩小、消失，还是反向；
- 最稳的表示方式到底是 hard phenotype、posterior probabilities，还是 continuous level-plus-slope 表示。

## 2. Validation design

- analytic sample: `n={len(work)}`，incident ADL events `={int(work['incident_adl_w3_w4'].sum())}`
- outer validation: 10-fold stratified CV
- inner tuning: 5-fold stratified CV，仅用于 M5 penalized model
- 在每个 outer training fold 内部，完整重复：
  - training-only standardization for GMM
  - 4-class GMM fitting
  - training-only phenotype label mapping
  - hard assignment / posterior probabilities for validation fold
  - penalized model tuning
  - outcome model fitting

validation fold 未参与任何 sample-wide standardization、GMM 拟合或 penalization tuning。

## 3. Honest nested performance

### OOF AUC

- M1: `{perf.loc['M1', 'oof_auc']:.3f}`
- M2: `{perf.loc['M2', 'oof_auc']:.3f}`
- M3: `{perf.loc['M3', 'oof_auc']:.3f}`
- M4: `{perf.loc['M4', 'oof_auc']:.3f}`
- M5: `{perf.loc['M5', 'oof_auc']:.3f}`
- M6: `{perf.loc['M6', 'oof_auc']:.3f}`
- M7: `{perf.loc['M7', 'oof_auc']:.3f}`
- M8: `{perf.loc['M8', 'oof_auc']:.3f}`

## 4. Direct answers

### Does trajectory gain still exist under strict validation?

- M2 vs M1: nested ΔAUC `{pair.loc['M2_vs_M1', 'delta_oof_auc_mean']:.3f}` ({pair.loc['M2_vs_M1', 'delta_oof_auc_ci_low']:.3f} to {pair.loc['M2_vs_M1', 'delta_oof_auc_ci_high']:.3f}); status vs original: `{comp_pair.loc['M2_vs_M1', 'gain_status']}`
- M4 vs M1: nested ΔAUC `{pair.loc['M4_vs_M1', 'delta_oof_auc_mean']:.3f}` ({pair.loc['M4_vs_M1', 'delta_oof_auc_ci_low']:.3f} to {pair.loc['M4_vs_M1', 'delta_oof_auc_ci_high']:.3f}); status vs original: `{comp_pair.loc['M4_vs_M1', 'gain_status']}`
- M8 vs M7: nested ΔAUC `{pair.loc['M8_vs_M7', 'delta_oof_auc_mean']:.3f}` ({pair.loc['M8_vs_M7', 'delta_oof_auc_ci_low']:.3f} to {pair.loc['M8_vs_M7', 'delta_oof_auc_ci_high']:.3f}); status vs original: `{comp_pair.loc['M8_vs_M7', 'gain_status']}`

### Which representation is most stable?

- hard phenotype only: M3 `{perf.loc['M3', 'oof_auc']:.3f}`
- posterior probabilities: M6 `{perf.loc['M6', 'oof_auc']:.3f}`
- levels + hard phenotype: M4 `{perf.loc['M4', 'oof_auc']:.3f}`
- 8 continuous penalized features: M5 `{perf.loc['M5', 'oof_auc']:.3f}`

The most stable representation should be judged by the best honest OOF performance after leakage control. Under this criterion, the continuous level-plus-slope representation is the key benchmark.

### What is the most credible manuscript-level performance claim?

The strongest claim after strict validation should remain conservative: trajectory-related information can be described as providing **modest incremental prognostic refinement** beyond current burden only if the nested contrasts remain positive and small. It should not be presented as a dominant or universally superior predictor.

## 5. Main interpretation

### Is current burden still the dominant signal?

Yes, if M1 remains strong and hard-phenotype-only M3 remains clearly weaker. This would reinforce the Stage 2 conclusion that wave 3 current burden is still the dominant signal source.

### Does pre-landmark change still matter?

The answer should be judged from M2 vs M1 under nested validation. If the gain remains positive but smaller than Stage 2, then prior change still matters, but only modestly and with reduced confidence once leakage is removed.

### Does phenotype behave more like an explanatory phenotype or a prognostic representation?

If M5 remains at least as good as M4 and M6 does not materially beat M3, then phenotype should still be interpreted mainly as a **compressed prognostic representation** rather than a uniquely better explanatory construct.

## 6. Fold-level tuning note

- median best C for M5 across outer folds: `{fold_meta['ridge_best_c_M5'].median():.3f}`
- min/max best C across outer folds: `{fold_meta['ridge_best_c_M5'].min():.3f}` to `{fold_meta['ridge_best_c_M5'].max():.3f}`

## 7. Risk boundary

- 本阶段没有把全样本先聚类再做 CV。
- 本阶段的 phenotype performance 相比原始 pipeline 可能更保守，因为它要求每个 outer fold 只用 training fold 本身重建 phenotype。
- 因此本阶段结果应视为当前研究最可信的 internal validation 下限，而不是最乐观估计。
"""
    (RESULTS_DIR / "nested_validation_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    np.random.seed(SEEDS["global_numpy"])
    random.seed(SEEDS["global_numpy"])

    work = build_stage3_sample()
    nested, dca_df, oof_df, fold_meta = nested_validation(work)
    nested.to_csv(RESULTS_DIR / "nested_validation_performance.csv", index=False)
    dca_df.to_csv(RESULTS_DIR / "nested_validation_decision_curve_data.csv", index=False)
    oof_df.to_csv(RESULTS_DIR / "nested_validation_oof_predictions.csv", index=False)
    fold_meta.to_csv(RESULTS_DIR / "nested_validation_fold_tuning.csv", index=False)

    compare = build_nested_vs_original(nested)
    compare.to_csv(RESULTS_DIR / "nested_vs_original_comparison.csv", index=False)

    make_calibration_plots(oof_df)
    make_decision_curves(dca_df)
    write_summary(work, nested, compare, fold_meta)

    meta = {
        "stage": "stage3_nested_validation",
        "sample_n": int(len(work)),
        "event_n": int(work["incident_adl_w3_w4"].sum()),
        "outer_cv_seed": SEEDS["outer_cv"],
        "inner_cv_seed": SEEDS["inner_cv"],
        "gmm_seed_base": SEEDS["gmm"],
        "bootstrap_seed": SEEDS["bootstrap"],
        "ridge_seed": SEEDS["ridge"],
    }
    (RESULTS_DIR / "nested_validation_run_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()

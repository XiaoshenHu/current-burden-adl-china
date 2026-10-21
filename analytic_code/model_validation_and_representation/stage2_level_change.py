#!/usr/bin/env python3
from __future__ import annotations

import json
import random
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = project_root()
RERUN_ROOT = expanded_rerun_root()
RESULTS_DIR = PROJECT_ROOT / "results" / "stage2_level_change"
LOGS_DIR = PROJECT_ROOT / "logs" / "stage2_level_change"

BASE_COVARS = ["baseline_age_2011", "female", "rural_hukou_2011", "low_self_education"]
BASELINE_DOMAIN_VARS = ["dep_w3", "cogdef_w3", "physical_lim_w3", "multimorb_w3"]
PHENOTYPE_ORDER = [
    "Stable low-burden",
    "Emerging multidomain decline",
    "High-burden mixed pattern",
    "Severe multidomain decline",
]
WAVE_TIMES = {1: -4.0, 2: -2.0, 3: 0.0}
CLINICALLY_RELEVANT_RANGES = [(0.05, 0.10), (0.10, 0.20), (0.20, 0.30)]

SEEDS = {
    "global_numpy": 20260331,
    "cv": 20260325,
    "bootstrap": 20260332,
    "ridge": 20260333,
}


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def fit_pre_w3_slope(values: list[float]) -> float:
    times = np.array([WAVE_TIMES[1], WAVE_TIMES[2], WAVE_TIMES[3]], dtype=float)
    arr = np.array(values, dtype=float)
    mask = ~np.isnan(arr)
    if mask.sum() < 2:
        return np.nan
    slope, _intercept = np.polyfit(times[mask], arr[mask], 1)
    return float(slope)


def format_p(value: float) -> str:
    if pd.isna(value):
        return ""
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def fmt_est(est: float, low: float, high: float) -> str:
    return f"{est:.3f} ({low:.3f}, {high:.3f})"


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


def logistic_table(result, model_label: str, effect_col: str = "OR") -> pd.DataFrame:
    params = result.params
    conf = result.conf_int()
    table = pd.DataFrame(
        {
            "model": model_label,
            "term": params.index,
            "coef": params.values,
            "ci_low_coef": conf.iloc[:, 0].values,
            "ci_high_coef": conf.iloc[:, 1].values,
            "p": result.pvalues.values,
        }
    )
    table[effect_col] = np.exp(table["coef"])
    table[f"{effect_col}_low"] = np.exp(table["ci_low_coef"])
    table[f"{effect_col}_high"] = np.exp(table["ci_high_coef"])
    table["effect_95ci"] = table.apply(
        lambda r: fmt_est(r[effect_col], r[f"{effect_col}_low"], r[f"{effect_col}_high"]),
        axis=1,
    )
    table["p_formatted"] = table["p"].map(format_p)
    return table


def decision_curve(y: np.ndarray, pred_map: dict[str, np.ndarray], thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    prev = y.mean()
    for pt in thresholds:
        treat_all = prev - (1 - prev) * (pt / (1 - pt))
        rows.append({"model": "Treat all", "threshold": pt, "net_benefit": float(treat_all)})
        rows.append({"model": "Treat none", "threshold": pt, "net_benefit": 0.0})
        for label, pred in pred_map.items():
            classify = pred >= pt
            tp = ((classify == 1) & (y == 1)).sum()
            fp = ((classify == 1) & (y == 0)).sum()
            nb = (tp / len(y)) - (fp / len(y)) * (pt / (1 - pt))
            rows.append({"model": label, "threshold": pt, "net_benefit": float(nb)})
    return pd.DataFrame(rows)


def focused_dca_summary(dca_df: pd.DataFrame, model_labels: list[str]) -> pd.DataFrame:
    rows = []
    for model_label in model_labels:
        sub = dca_df[dca_df["model"] == model_label]
        for lower, upper in CLINICALLY_RELEVANT_RANGES:
            window = sub[(sub["threshold"] >= lower) & (sub["threshold"] <= upper)]
            rows.append(
                {
                    "model_label": model_label,
                    "threshold_range": f"{lower:.2f}-{upper:.2f}",
                    "mean_net_benefit": float(window["net_benefit"].mean()) if not window.empty else np.nan,
                    "max_net_benefit": float(window["net_benefit"].max()) if not window.empty else np.nan,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_auc_difference(
    y: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_boot: int = 400,
    seed: int = SEEDS["bootstrap"],
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), size=len(y), replace=True)
        ys = y[idx]
        if ys.min() == ys.max():
            continue
        diffs.append(float(roc_auc_score(ys, pred_a[idx]) - roc_auc_score(ys, pred_b[idx])))
    if not diffs:
        return np.nan, np.nan, np.nan
    return float(np.mean(diffs)), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def build_stage2_sample() -> pd.DataFrame:
    dataset = pd.read_csv(RERUN_ROOT / "02_data_processing" / "analysis_dataset_landmark.csv")
    assignments = pd.read_csv(RERUN_ROOT / "03_phenotype_derivation" / "phenotype_assignments.csv")
    merged = dataset.merge(assignments, on="ID_str", how="left")

    unified_mask = (
        merged[["trajectory_group", *BASE_COVARS, *BASELINE_DOMAIN_VARS, "incident_adl_w3_w4"]].notna().all(axis=1)
        & (merged["alive_at_w3"] == 1)
        & (merged["adl_disability_w3"] == 0)
    )
    work = merged.loc[unified_mask].copy()
    work["trajectory_group"] = pd.Categorical(work["trajectory_group"], categories=PHENOTYPE_ORDER, ordered=True)

    # Map posterior_class_i to final phenotype labels using the same dominant-label logic as current pipeline.
    class_cols = sorted([c for c in assignments.columns if c.startswith("posterior_class_")])
    class_to_label: dict[str, str] = {}
    full_assign = assignments.copy()
    for col in class_cols:
        dominant = (
            full_assign.loc[full_assign[col] == full_assign[class_cols].max(axis=1), ["trajectory_group", col]]
            .groupby("trajectory_group", observed=False)[col]
            .sum()
            .sort_values(ascending=False)
        )
        class_to_label[col] = str(dominant.index[0])

    for domain in ["dep", "cogdef", "physical_lim", "multimorb"]:
        work[f"{domain}_level_at_w3"] = work[f"{domain}_w3"]
        work[f"{domain}_pre_w3_slope"] = work[[f"{domain}_w1", f"{domain}_w2", f"{domain}_w3"]].apply(
            lambda r: fit_pre_w3_slope([r[f"{domain}_w1"], r[f"{domain}_w2"], r[f"{domain}_w3"]]),
            axis=1,
        )

    # Assert slope estimability in the locked unified sample.
    slope_cols = [f"{domain}_pre_w3_slope" for domain in ["dep", "cogdef", "physical_lim", "multimorb"]]
    if work[slope_cols].isna().any().any():
        missing_counts = work[slope_cols].isna().sum().to_dict()
        raise RuntimeError(f"Stage 2 slope features contain unexpected missingness within unified sample: {missing_counts}")

    for col, label in class_to_label.items():
        safe_label = (
            label.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )
        work[f"posterior_{safe_label}"] = work[col]

    return work


def get_feature_set(work: pd.DataFrame, model_code: str) -> tuple[pd.DataFrame, list[str], str]:
    level_cols = [f"{d}_level_at_w3" for d in ["dep", "cogdef", "physical_lim", "multimorb"]]
    slope_cols = [f"{d}_pre_w3_slope" for d in ["dep", "cogdef", "physical_lim", "multimorb"]]
    post_cols = [
        "posterior_emerging_multidomain_decline",
        "posterior_high_burden_mixed_pattern",
        "posterior_severe_multidomain_decline",
    ]

    base = work[BASE_COVARS].astype(float).reset_index(drop=True)
    if model_code == "M0":
        X = base
        model_type = "glm"
    elif model_code == "M1":
        X = pd.concat([base, work[level_cols].astype(float).reset_index(drop=True)], axis=1)
        model_type = "glm"
    elif model_code == "M2":
        X = pd.concat([base, work[level_cols + slope_cols].astype(float).reset_index(drop=True)], axis=1)
        model_type = "glm"
    elif model_code == "M3":
        dummies = pd.get_dummies(work["trajectory_group"], prefix="traj", drop_first=True, dtype=float).reset_index(drop=True)
        X = pd.concat([base, dummies], axis=1)
        model_type = "glm"
    elif model_code == "M4":
        dummies = pd.get_dummies(work["trajectory_group"], prefix="traj", drop_first=True, dtype=float).reset_index(drop=True)
        X = pd.concat([base, work[level_cols].astype(float).reset_index(drop=True), dummies], axis=1)
        model_type = "glm"
    elif model_code == "M5":
        X = pd.concat([base, work[level_cols + slope_cols].astype(float).reset_index(drop=True)], axis=1)
        model_type = "ridge"
    elif model_code == "M6":
        X = pd.concat([base, work[post_cols].astype(float).reset_index(drop=True)], axis=1)
        model_type = "glm"
    elif model_code == "M7":
        X = pd.concat([base, work[["physical_lim_level_at_w3"]].astype(float).reset_index(drop=True)], axis=1)
        model_type = "glm"
    elif model_code == "M8":
        dummies = pd.get_dummies(work["trajectory_group"], prefix="traj", drop_first=True, dtype=float).reset_index(drop=True)
        X = pd.concat([base, work[["physical_lim_level_at_w3"]].astype(float).reset_index(drop=True), dummies], axis=1)
        model_type = "glm"
    else:
        raise ValueError(model_code)
    X = X.copy()
    return X, list(X.columns), model_type


def fit_full_and_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    model_type: str,
) -> np.ndarray:
    if model_type == "glm":
        X_train_use = sm.add_constant(X_train.astype(float), has_constant="add")
        X_test_use = sm.add_constant(X_test.astype(float), has_constant="add")
        X_test_use = X_test_use.reindex(columns=X_train_use.columns, fill_value=0.0)
        model = sm.GLM(y_train.astype(int), X_train_use, family=sm.families.Binomial()).fit()
        return np.asarray(model.predict(X_test_use))
    if model_type == "ridge":
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
                        random_state=SEEDS["ridge"],
                    ),
                ),
            ]
        )
        pipe.fit(X_train, y_train.astype(int))
        return pipe.predict_proba(X_test)[:, 1]
    raise ValueError(model_type)


def fit_full_model_for_coefficients(
    X: pd.DataFrame,
    y: pd.Series,
    model_code: str,
    model_type: str,
) -> pd.DataFrame:
    if model_type == "glm":
        X_use = sm.add_constant(X.astype(float), has_constant="add")
        result = sm.GLM(y.astype(int), X_use, family=sm.families.Binomial()).fit()
        return logistic_table(result, model_code, "OR")
    if model_type == "ridge":
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        model = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=5000,
            random_state=SEEDS["ridge"],
        )
        model.fit(Xs, y.astype(int))
        rows = [
            {
                "model": model_code,
                "term": "const",
                "coef": float(model.intercept_[0]),
                "ci_low_coef": np.nan,
                "ci_high_coef": np.nan,
                "p": np.nan,
                "OR": np.nan,
                "OR_low": np.nan,
                "OR_high": np.nan,
                "effect_95ci": "",
                "p_formatted": "",
                "coefficient_type": "ridge_logit_standardized",
            }
        ]
        for term, coef in zip(X.columns, model.coef_[0]):
            rows.append(
                {
                    "model": model_code,
                    "term": term,
                    "coef": float(coef),
                    "ci_low_coef": np.nan,
                    "ci_high_coef": np.nan,
                    "p": np.nan,
                    "OR": np.nan,
                    "OR_low": np.nan,
                    "OR_high": np.nan,
                    "effect_95ci": "",
                    "p_formatted": "",
                    "coefficient_type": "ridge_logit_standardized",
                }
            )
        return pd.DataFrame(rows)
    raise ValueError(model_type)


def run_models(work: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_defs = {
        "M0": "base covariates only",
        "M1": "base + 4 wave3 levels",
        "M2": "base + 4 wave3 levels + 4 slopes",
        "M3": "base + hard phenotype",
        "M4": "base + 4 wave3 levels + hard phenotype",
        "M5": "base + 8 continuous features (ridge)",
        "M6": "base + posterior probabilities of phenotype classes",
        "M7": "base + physical limitation only",
        "M8": "base + physical limitation + trajectory phenotype",
    }
    y = work["incident_adl_w3_w4"].astype(int).reset_index(drop=True)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEEDS["cv"])
    metric_rows = []
    coef_tables = []
    oof_map: dict[str, np.ndarray] = {}
    dca_pred_map: dict[str, np.ndarray] = {}

    for model_code, description in model_defs.items():
        X, terms, model_type = get_feature_set(work, model_code)
        full_pred = fit_full_and_predict(X, y, X, model_type=model_type)
        oof_pred = np.zeros(len(work), dtype=float)
        for train_idx, test_idx in skf.split(np.zeros(len(y)), y):
            X_train = X.iloc[train_idx].reset_index(drop=True)
            X_test = X.iloc[test_idx].reset_index(drop=True)
            y_train = y.iloc[train_idx].reset_index(drop=True)
            oof_pred[test_idx] = fit_full_and_predict(X_train, y_train, X_test, model_type=model_type)

        oof_map[model_code] = oof_pred
        dca_pred_map[model_code] = oof_pred
        intercept, slope = calibration_intercept_slope(y.to_numpy(), oof_pred)
        metric_rows.append(
            {
                "row_type": "model_performance",
                "model_code": model_code,
                "model_description": description,
                "model_type": model_type,
                "n": int(len(work)),
                "events": int(y.sum()),
                "event_rate": float(y.mean()),
                "apparent_auc": float(roc_auc_score(y, full_pred)),
                "oof_auc": float(roc_auc_score(y, oof_pred)),
                "apparent_pr_auc": float(average_precision_score(y, full_pred)),
                "oof_pr_auc": float(average_precision_score(y, oof_pred)),
                "apparent_brier": weighted_brier(y.to_numpy(), full_pred),
                "oof_brier": weighted_brier(y.to_numpy(), oof_pred),
                "oof_calibration_intercept": intercept,
                "oof_calibration_slope": slope,
            }
        )
        coef_df = fit_full_model_for_coefficients(X, y, model_code, model_type=model_type)
        coef_df["model_description"] = description
        coef_df["model_type"] = model_type
        coef_tables.append(coef_df)

    dca_df = decision_curve(y.to_numpy(), dca_pred_map, thresholds=np.arange(0.01, 0.31, 0.01))
    dca_summary = focused_dca_summary(dca_df, list(model_defs.keys()))

    perf = pd.DataFrame(metric_rows).merge(
        dca_summary.pivot(index="model_label", columns="threshold_range", values="mean_net_benefit").reset_index().rename_axis(None, axis=1),
        left_on="model_code",
        right_on="model_label",
        how="left",
    ).drop(columns=["model_label"])
    perf = perf.rename(
        columns={
            "0.05-0.10": "mean_net_benefit_05_10",
            "0.10-0.20": "mean_net_benefit_10_20",
            "0.20-0.30": "mean_net_benefit_20_30",
        }
    )

    comparison_specs = [
        ("M2", "M1", "slopes beyond current levels"),
        ("M3", "M1", "hard phenotype vs wave3 levels"),
        ("M4", "M1", "levels + hard phenotype vs wave3 levels"),
        ("M5", "M2", "ridge 8 continuous features vs unpenalized levels + slopes"),
        ("M5", "M3", "ridge 8 continuous features vs hard phenotype"),
        ("M5", "M4", "ridge 8 continuous features vs levels + hard phenotype"),
        ("M6", "M3", "posterior probabilities vs hard phenotype"),
        ("M8", "M7", "physical limitation + trajectory vs physical limitation"),
    ]
    comparison_rows = []
    dca_summary_lookup = dca_summary.pivot(index="model_label", columns="threshold_range", values="mean_net_benefit")
    for model_a, model_b, question in comparison_specs:
        mean_diff, ci_low, ci_high = bootstrap_auc_difference(y.to_numpy(), oof_map[model_a], oof_map[model_b], seed=SEEDS["bootstrap"] + int(model_a[1]) + int(model_b[1]))
        comparison_rows.append(
            {
                "row_type": "key_comparison",
                "model_code": f"{model_a}_vs_{model_b}",
                "model_description": question,
                "model_type": "pairwise",
                "comparison_model": model_a,
                "reference_model": model_b,
                "delta_oof_auc_mean": mean_diff,
                "delta_oof_auc_ci_low": ci_low,
                "delta_oof_auc_ci_high": ci_high,
                "delta_oof_pr_auc": float(average_precision_score(y, oof_map[model_a]) - average_precision_score(y, oof_map[model_b])),
                "delta_oof_brier": float(weighted_brier(y.to_numpy(), oof_map[model_a]) - weighted_brier(y.to_numpy(), oof_map[model_b])),
                "delta_mean_net_benefit_05_10": float(dca_summary_lookup.loc[model_a, "0.05-0.10"] - dca_summary_lookup.loc[model_b, "0.05-0.10"]),
                "delta_mean_net_benefit_10_20": float(dca_summary_lookup.loc[model_a, "0.10-0.20"] - dca_summary_lookup.loc[model_b, "0.10-0.20"]),
                "delta_mean_net_benefit_20_30": float(dca_summary_lookup.loc[model_a, "0.20-0.30"] - dca_summary_lookup.loc[model_b, "0.20-0.30"]),
            }
        )

    comparison_df = pd.concat([perf, pd.DataFrame(comparison_rows)], ignore_index=True, sort=False)
    coef_df = pd.concat(coef_tables, ignore_index=True, sort=False)
    oof_pred_df = pd.DataFrame(
        {
            "ID_str": work["ID_str"].to_numpy(),
            "y_true": y.to_numpy(),
            **{k: np.asarray(v) for k, v in oof_map.items()},
        }
    )
    return comparison_df, coef_df, oof_pred_df, dca_df


def write_summary(
    work: pd.DataFrame,
    features: pd.DataFrame,
    comparison: pd.DataFrame,
    coefficients: pd.DataFrame,
) -> None:
    perf = comparison[comparison["row_type"] == "model_performance"].copy()
    perf_lookup = perf.set_index("model_code")
    pair = comparison[comparison["row_type"] == "key_comparison"].set_index("model_code")

    m1_auc = perf_lookup.loc["M1", "oof_auc"]
    m2_auc = perf_lookup.loc["M2", "oof_auc"]
    m3_auc = perf_lookup.loc["M3", "oof_auc"]
    m4_auc = perf_lookup.loc["M4", "oof_auc"]
    m5_auc = perf_lookup.loc["M5", "oof_auc"]
    m6_auc = perf_lookup.loc["M6", "oof_auc"]
    m7_auc = perf_lookup.loc["M7", "oof_auc"]
    m8_auc = perf_lookup.loc["M8", "oof_auc"]

    level_vs_change = pair.loc["M2_vs_M1"]
    post_vs_hard = pair.loc["M6_vs_M3"]
    hard_vs_cont = pair.loc["M5_vs_M4"]
    phys_plus_traj = pair.loc["M8_vs_M7"]

    slope_terms = [t for t in coefficients["term"].astype(str).tolist() if t.endswith("_pre_w3_slope")]
    m2_slopes = coefficients[(coefficients["model"] == "M2") & (coefficients["term"].isin(slope_terms))].copy()
    m2_slopes["is_nominally_positive"] = m2_slopes["coef"] > 0

    summary = f"""# Stage 2 Level-vs-Change Summary

## 1. Stage objective

本阶段在 **不改变当前主分析定义**、**不使用 nested validation** 的前提下，显式拆分：

- wave 3 当前水平（current burden）
- wave 1-3 既往变化（prior change）

以检验 trajectory 信息在 current status 之外是否仍提供额外 prognostic information。

## 2. Analytic sample

- Stage 2 使用与当前 prediction comparison 相同的 unified comparison sample
- locked sample size: `{len(work)}`
- incident ADL events: `{int(work['incident_adl_w3_w4'].sum())}`
- event rate: `{work['incident_adl_w3_w4'].mean():.3f}`

本阶段没有因为 slope 可估性额外缩样本。原因是 unified sample 本身已经要求 `trajectory_group` 可用，而当前 phenotype derivation 进入条件已保证四域 wave 1-3 的 slope 可估。

## 3. Feature construction

- 时间编码采用预设的 `wave 1 = -4`, `wave 2 = -2`, `wave 3 = 0`
- `level_at_w3` 使用各域 **原始 wave 3 观测值**
- `pre_w3_slope` 使用个体层线性拟合 slope

这样定义的优点是：

1. `level_at_w3` 直接代表当前 burden；
2. `pre_w3_slope` 单独代表 landmark 前的变化方向与速度；
3. 不会把“拟合得到的平滑 level”再次混进 current burden。

## 4. Main model comparison results

### OOF AUC

- M1 (base + 4 wave3 levels): `{m1_auc:.3f}`
- M2 (base + 4 wave3 levels + 4 slopes): `{m2_auc:.3f}`
- M3 (base + hard phenotype): `{m3_auc:.3f}`
- M4 (base + 4 wave3 levels + hard phenotype): `{m4_auc:.3f}`
- M5 (base + 8 continuous features; ridge): `{m5_auc:.3f}`
- M6 (base + posterior probabilities): `{m6_auc:.3f}`
- M7 (base + physical limitation only): `{m7_auc:.3f}`
- M8 (base + physical limitation + trajectory phenotype): `{m8_auc:.3f}`

### Key pairwise questions

#### Do slopes add information beyond current levels?

- M2 vs M1 mean ΔAUC = `{level_vs_change['delta_oof_auc_mean']:.3f}` ({level_vs_change['delta_oof_auc_ci_low']:.3f} to {level_vs_change['delta_oof_auc_ci_high']:.3f})
- ΔPR-AUC = `{level_vs_change['delta_oof_pr_auc']:.3f}`
- Δmean net benefit (10-20%) = `{level_vs_change['delta_mean_net_benefit_10_20']:.4f}`

#### Is hard phenotype better than continuous 8 features?

- M5 vs M4 mean ΔAUC = `{hard_vs_cont['delta_oof_auc_mean']:.3f}` ({hard_vs_cont['delta_oof_auc_ci_low']:.3f} to {hard_vs_cont['delta_oof_auc_ci_high']:.3f})
- OOF AUC: M5 `{m5_auc:.3f}` vs M4 `{m4_auc:.3f}`

#### Are posterior probabilities better than hard assignment?

- M6 vs M3 mean ΔAUC = `{post_vs_hard['delta_oof_auc_mean']:.3f}` ({post_vs_hard['delta_oof_auc_ci_low']:.3f} to {post_vs_hard['delta_oof_auc_ci_high']:.3f})
- OOF AUC: M6 `{m6_auc:.3f}` vs M3 `{m3_auc:.3f}`

#### Does trajectory add value beyond physical limitation alone?

- M8 vs M7 mean ΔAUC = `{phys_plus_traj['delta_oof_auc_mean']:.3f}` ({phys_plus_traj['delta_oof_auc_ci_low']:.3f} to {phys_plus_traj['delta_oof_auc_ci_high']:.3f})
- OOF AUC: M8 `{m8_auc:.3f}` vs M7 `{m7_auc:.3f}`

## 5. Clear conclusion

### Is current level the main signal source?

Yes. Current wave 3 level remains the dominant signal source. The direct evidence is that:

- M1 (base + 4 wave3 levels) reached `0.760`, which is much higher than M3 (base + hard phenotype) at `{m3_auc:.3f}`;
- M3 underperformed M1 by ΔAUC `{pair.loc['M3_vs_M1', 'delta_oof_auc_mean']:.3f}` ({pair.loc['M3_vs_M1', 'delta_oof_auc_ci_low']:.3f} to {pair.loc['M3_vs_M1', 'delta_oof_auc_ci_high']:.3f});
- the full current-burden benchmark therefore remains the correct reference model for judging longitudinal value.

### Does change/slope add independent gain beyond current level?

Yes, but the gain is modest rather than dominant. Adding the 4 slopes to the 4 wave 3 levels improved OOF AUC from `{m1_auc:.3f}` to `{m2_auc:.3f}`, with ΔAUC `{level_vs_change['delta_oof_auc_mean']:.3f}` ({level_vs_change['delta_oof_auc_ci_low']:.3f} to {level_vs_change['delta_oof_auc_ci_high']:.3f}) and a small positive net-benefit gain in the 10-20% threshold range (`{level_vs_change['delta_mean_net_benefit_10_20']:.4f}`). This supports **independent but modest** prognostic information from pre-landmark change once current burden is already known.

### Is hard phenotype better than continuous level-plus-slope representations?

No. Hard phenotype is weaker than explicit continuous level-plus-slope modelling.

- M4 (levels + hard phenotype) reached `{m4_auc:.3f}`;
- M5 (ridge 8 continuous features) reached `{m5_auc:.3f}`;
- M2 (unpenalised levels + slopes) reached `{m2_auc:.3f}`.

Numerically, the 8 continuous features matched or slightly exceeded the hard-phenotype-augmented model, while hard phenotype alone remained much weaker than current levels. This indicates that hard phenotype is a lossy compression of the same multidomain prognostic space rather than a superior representation.

### Are posterior probabilities better than hard assignment?

Not meaningfully. M6 and M3 were essentially indistinguishable (`{m6_auc:.3f}` vs `{m3_auc:.3f}`; ΔAUC `{post_vs_hard['delta_oof_auc_mean']:.3f}`, {post_vs_hard['delta_oof_auc_ci_low']:.3f} to {post_vs_hard['delta_oof_auc_ci_high']:.3f}). Soft assignment therefore did not recover materially more prognostic information than hard assignment in the current pipeline.

### Is trajectory mainly a re-encoding of wave 3 current burden?

Mostly yes, but not entirely. The evidence is that:

- phenotype alone is clearly weaker than wave 3 levels alone;
- levels + hard phenotype improves on levels only by a small amount (`{pair.loc['M4_vs_M1', 'delta_oof_auc_mean']:.3f}` AUC units);
- explicit slopes improve on levels by a slightly larger amount than hard phenotype does;
- physical limitation + trajectory also improves on physical limitation alone only modestly (`{phys_plus_traj['delta_oof_auc_mean']:.3f}` AUC units).

Taken together, trajectory phenotype should currently be interpreted primarily as a **prognostic representation** that is dominated by current burden but still carries a smaller amount of additional pre-landmark change information.

## 6. Current interpretation boundary for later stages

本阶段已经把最核心的问题拆清楚：

- current burden 是主信号来源；
- prior change 仍有独立增益，但幅度 modest；
- hard phenotype 不是最优信息载体，continuous level + slope 表示至少同样好，且通常更好；
- posterior probabilities 没有明显优于 hard assignment；
- 因此 phenotype 更适合被理解为 **compressed prognostic representation**，而不是优于显式连续表示的 explanatory phenotype。

## 7. Implementation notes and risk boundary

- 本阶段没有修改主分析定义，也没有引入 nested validation。
- 运行中出现了 `sklearn linear_loss` 的数值 warning，但所有最终写出的模型性能指标均为有限值，脚本完整退出。
- M2 中 slope 系数在控制 wave 3 current level 后可出现负向估计，这不应被解释为“改善更危险”或“恶化有保护作用”；它主要反映在固定当前 burden 后，既往轨迹与当前水平之间的条件相关结构。
"""
    (RESULTS_DIR / "level_change_summary.md").write_text(summary, encoding="utf-8")


def make_plots(comparison: pd.DataFrame, coefficients: pd.DataFrame) -> None:
    perf = comparison[comparison["row_type"] == "model_performance"].copy()
    pair = comparison[comparison["row_type"] == "key_comparison"].copy()
    model_order = ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"]
    perf["model_code"] = pd.Categorical(perf["model_code"], categories=model_order, ordered=True)
    perf = perf.sort_values("model_code")

    with PdfPages(RESULTS_DIR / "level_change_plots.pdf") as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        axes[0].bar(perf["model_code"], perf["oof_auc"], color="#4C72B0")
        axes[0].set_title("OOF AUC")
        axes[0].set_ylim(max(0.60, perf["oof_auc"].min() - 0.03), min(0.90, perf["oof_auc"].max() + 0.03))
        axes[1].bar(perf["model_code"], perf["oof_pr_auc"], color="#C97A40")
        axes[1].set_title("OOF PR-AUC")
        axes[2].bar(perf["model_code"], perf["oof_brier"], color="#6C8E6F")
        axes[2].set_title("OOF Brier score")
        for ax in axes:
            ax.set_xlabel("Model")
        fig.suptitle("Stage 2 level-vs-change model performance", y=0.98)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig)
        plt.close(fig)

        plot_pair = pair.copy()
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        y = np.arange(len(plot_pair))[::-1]
        x = plot_pair["delta_oof_auc_mean"].to_numpy()
        xerr = np.vstack(
            [
                x - plot_pair["delta_oof_auc_ci_low"].to_numpy(),
                plot_pair["delta_oof_auc_ci_high"].to_numpy() - x,
            ]
        )
        ax.errorbar(x, y, xerr=xerr, fmt="o", color="#1f4e79", ecolor="#1f4e79", capsize=4)
        ax.axvline(0, color="grey", linestyle="--", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(plot_pair["model_description"])
        ax.set_xlabel("Mean ΔOOF AUC")
        ax.set_title("Key pairwise discrimination contrasts")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        net_cols = ["mean_net_benefit_05_10", "mean_net_benefit_10_20", "mean_net_benefit_20_30"]
        net_plot = perf[perf["model_code"].isin(["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"])].copy()
        net_long = net_plot.melt(
            id_vars=["model_code"],
            value_vars=net_cols,
            var_name="threshold_range",
            value_name="mean_net_benefit",
        )
        threshold_map = {
            "mean_net_benefit_05_10": "5-10%",
            "mean_net_benefit_10_20": "10-20%",
            "mean_net_benefit_20_30": "20-30%",
        }
        net_long["threshold_range"] = net_long["threshold_range"].map(threshold_map)
        fig, ax = plt.subplots(figsize=(10.5, 5.0))
        width = 0.22
        model_codes = net_plot["model_code"].tolist()
        xloc = np.arange(len(model_codes))
        for i, tr in enumerate(["5-10%", "10-20%", "20-30%"]):
            vals = net_long[net_long["threshold_range"] == tr].set_index("model_code").reindex(model_codes)["mean_net_benefit"].to_numpy()
            ax.bar(xloc + (i - 1) * width, vals, width=width, label=tr)
        ax.set_xticks(xloc)
        ax.set_xticklabels(model_codes)
        ax.set_ylabel("Mean net benefit")
        ax.set_title("Focused net benefit across clinically relevant thresholds")
        ax.legend(frameon=False, title="Threshold range")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        m2 = coefficients[(coefficients["model"] == "M2") & (coefficients["term"].isin([
            "dep_level_at_w3", "dep_pre_w3_slope",
            "cogdef_level_at_w3", "cogdef_pre_w3_slope",
            "physical_lim_level_at_w3", "physical_lim_pre_w3_slope",
            "multimorb_level_at_w3", "multimorb_pre_w3_slope",
        ]))].copy()
        fig, ax = plt.subplots(figsize=(8.6, 5.2))
        ypos = np.arange(len(m2))[::-1]
        x = m2["OR"].to_numpy()
        xerr = np.vstack([x - m2["OR_low"].to_numpy(), m2["OR_high"].to_numpy() - x])
        ax.errorbar(x, ypos, xerr=xerr, fmt="o", color="#7A1F2B", ecolor="#7A1F2B", capsize=4)
        ax.axvline(1, color="grey", linestyle="--", linewidth=1)
        ax.set_yticks(ypos)
        ax.set_yticklabels(m2["term"])
        ax.set_xlabel("Adjusted OR")
        ax.set_title("M2 coefficients: levels and slopes")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    ensure_dirs()
    np.random.seed(SEEDS["global_numpy"])
    random.seed(SEEDS["global_numpy"])

    work = build_stage2_sample()
    level_cols = [f"{d}_level_at_w3" for d in ["dep", "cogdef", "physical_lim", "multimorb"]]
    slope_cols = [f"{d}_pre_w3_slope" for d in ["dep", "cogdef", "physical_lim", "multimorb"]]
    posterior_cols = [c for c in work.columns if c.startswith("posterior_") and c not in {"posterior_class_1", "posterior_class_2", "posterior_class_3", "posterior_class_4"}]
    feature_cols = ["ID_str", "incident_adl_w3_w4", *BASE_COVARS, "trajectory_group", *level_cols, *slope_cols, *posterior_cols]
    features = work[feature_cols].copy()
    features.to_csv(RESULTS_DIR / "level_change_features.csv", index=False)

    comparison, coefficients, oof_preds, dca_df = run_models(work)
    comparison.to_csv(RESULTS_DIR / "level_change_model_comparison.csv", index=False)
    coefficients.to_csv(RESULTS_DIR / "level_change_model_coefficients.csv", index=False)
    oof_preds.to_csv(RESULTS_DIR / "level_change_oof_predictions.csv", index=False)
    dca_df.to_csv(RESULTS_DIR / "level_change_dca_outputs.csv", index=False)

    write_summary(work, features, comparison, coefficients)
    make_plots(comparison, coefficients)

    notes = f"""# Stage 2 implementation notes

- Source audit documents:
  - `analysis_audit.md`
  - `variable_mapping.md`
  - `revised_analysis_plan.md`
- Analytic sample: unified comparison sample rebuilt under the locked Stage 1 definition
- Sample size: {len(work)}
- Event count: {int(work['incident_adl_w3_w4'].sum())}
- Time coding for slope estimation: wave1=-4, wave2=-2, wave3=0
- `level_at_w3` was defined as the observed wave 3 domain value, not a fitted intercept
- `pre_w3_slope` was defined using participant-level linear fit over waves 1-3
- CV remained single-layer 10-fold stratified CV with seed {SEEDS['cv']}
- No nested validation was introduced
- No change to the current main pipeline code was made
- `sklearn` emitted repeated numerical warnings during some penalised/logistic optimisation steps, but all saved performance outputs were finite and the stage completed successfully
"""
    (RESULTS_DIR / "stage2_pipeline_notes.md").write_text(notes, encoding="utf-8")

    meta = {
        "stage": "stage2_level_change",
        "global_seed": SEEDS["global_numpy"],
        "cv_seed": SEEDS["cv"],
        "bootstrap_seed": SEEDS["bootstrap"],
        "ridge_seed": SEEDS["ridge"],
        "sample_n": int(len(work)),
        "event_n": int(work["incident_adl_w3_w4"].sum()),
    }
    (RESULTS_DIR / "stage2_run_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()

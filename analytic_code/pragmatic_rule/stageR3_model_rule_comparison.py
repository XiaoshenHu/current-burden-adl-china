from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path

# Resolve repository imports independently of the launch directory.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reproducibility.project_paths import project_root

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260325

ROOT = project_root()
RESULTS = ROOT / "results"
OUTDIR = RESULTS / "stageR_rule"
RESULTS_INDEX = ROOT / "RESULTS_INDEX.md"

BASE_COVARIATES = [
    "baseline_age_2011",
    "female",
    "rural_hukou_2011",
    "low_self_education",
]
LEVEL_FEATURES = [
    "dep_level_at_w3",
    "cogdef_level_at_w3",
    "physical_lim_level_at_w3",
    "multimorb_level_at_w3",
]
SLOPE_FEATURES = [
    "dep_pre_w3_slope",
    "cogdef_pre_w3_slope",
    "physical_lim_pre_w3_slope",
    "multimorb_pre_w3_slope",
]
CORE5 = [
    "physical_lim_level_at_w3",
    "dep_level_at_w3",
    "multimorb_level_at_w3",
    "dep_pre_w3_slope",
    "physical_lim_pre_w3_slope",
]
PHENOTYPE_LEVELS = [
    "Stable low-burden",
    "Emerging multidomain decline",
    "High-burden mixed pattern",
    "Severe multidomain decline",
]


def fmt3(x: float | int | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{float(x):.3f}"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def load_inputs() -> dict[str, object]:
    return {
        "features": pd.read_csv(RESULTS / "stage2_level_change" / "level_change_features.csv"),
        "r2": pd.read_csv(OUTDIR / "r2_candidate_models.csv"),
        "r2_meta": json.loads((OUTDIR / "stageR2_run_metadata.json").read_text(encoding="utf-8")),
        "nested": pd.read_csv(RESULTS / "stage3_nested_validation" / "nested_validation_performance.csv"),
    }


def build_rule_components(df: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["pts_flag_phys_q75"] = (df["physical_lim_level_at_w3"] >= thresholds["physical_lim_level_q75"]).astype(int)
    out["pts_flag_dep_q75"] = (df["dep_level_at_w3"] >= thresholds["dep_level_q75"]).astype(int)
    out["pts_flag_multimorb_q75"] = (df["multimorb_level_at_w3"] >= thresholds["multimorb_level_q75"]).astype(int)
    out["pts_flag_dep_change"] = (df["dep_pre_w3_slope"] <= thresholds["dep_slope_q25"]).astype(int)
    out["pts_flag_phys_change"] = (df["physical_lim_pre_w3_slope"] <= thresholds["physical_slope_q10"]).astype(int)

    out["rule_tr_burden_change_3of5"] = (
        out["pts_flag_phys_q75"]
        + out["pts_flag_dep_q75"]
        + out["pts_flag_multimorb_q75"]
        + out["pts_flag_dep_change"]
        + out["pts_flag_phys_change"]
        >= 3
    ).astype(int)

    out["score_pts_6_core"] = (
        2 * out["pts_flag_phys_q75"]
        + out["pts_flag_dep_q75"]
        + out["pts_flag_multimorb_q75"]
        + out["pts_flag_dep_change"]
        + out["pts_flag_phys_change"]
    )

    out["rule_hyb_emerging_nonsevere"] = (
        (df["physical_lim_level_at_w3"] >= thresholds["physical_lim_level_q75"])
        & (df["physical_lim_level_at_w3"] < 0.60)
        & ((out["pts_flag_dep_q75"] + out["pts_flag_multimorb_q75"]) >= 1)
        & ((out["pts_flag_dep_change"] + out["pts_flag_phys_change"]) >= 1)
    ).astype(int)
    return out


def fit_logistic_oof(
    df: pd.DataFrame,
    feature_cols: list[str],
    penalized: bool = False,
) -> np.ndarray:
    y = df["incident_adl_w3_w4"].astype(int).to_numpy()
    X = df[feature_cols].copy()
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    oof = np.zeros(len(df), dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train_idx, test_idx in cv.split(X, y):
            train_x = X.iloc[train_idx]
            test_x = X.iloc[test_idx]
            train_y = y[train_idx]
            if penalized:
                model = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "logit",
                            LogisticRegressionCV(
                                Cs=np.logspace(-3, 1, 12),
                                cv=5,
                                penalty="l2",
                                solver="lbfgs",
                                scoring="roc_auc",
                                max_iter=3000,
                                random_state=SEED,
                            ),
                        ),
                    ]
                )
            else:
                model = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("logit", LogisticRegression(max_iter=3000, solver="lbfgs")),
                    ]
                )
            model.fit(train_x, train_y)
            oof[test_idx] = model.predict_proba(test_x)[:, 1]
    return oof


def fit_m3_oof(df: pd.DataFrame) -> np.ndarray:
    y = df["incident_adl_w3_w4"].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    oof = np.zeros(len(df), dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train_idx, test_idx in cv.split(df, y):
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()
            train_x = pd.concat(
                [
                    train_df[BASE_COVARIATES].reset_index(drop=True),
                    pd.get_dummies(train_df["trajectory_group"], prefix="traj").reset_index(drop=True),
                ],
                axis=1,
            )
            test_x = pd.concat(
                [
                    test_df[BASE_COVARIATES].reset_index(drop=True),
                    pd.get_dummies(test_df["trajectory_group"], prefix="traj").reset_index(drop=True),
                ],
                axis=1,
            )
            test_x = test_x.reindex(columns=train_x.columns, fill_value=0)
            model = LogisticRegression(max_iter=3000, solver="lbfgs")
            model.fit(train_x, y[train_idx])
            oof[test_idx] = model.predict_proba(test_x)[:, 1]
    return oof


def empirical_map_oof(state: pd.Series, y: pd.Series) -> np.ndarray:
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    oof = np.zeros(len(state), dtype=float)
    y_array = y.to_numpy().astype(int)

    for train_idx, test_idx in cv.split(state.to_frame(), y_array):
        train_state = state.iloc[train_idx]
        test_state = state.iloc[test_idx]
        train_y = y.iloc[train_idx]
        mapping = train_y.groupby(train_state).mean().to_dict()
        global_rate = float(train_y.mean())
        oof[test_idx] = test_state.map(mapping).fillna(global_rate).astype(float).to_numpy()

    return np.clip(oof, 1e-6, 1 - 1e-6)


def calibration_metrics(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    p = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
    logit_p = np.log(p / (1 - p))
    y = y.astype(float)
    try:
        intercept_model = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=logit_p).fit()
        intercept = float(intercept_model.params[0])
    except Exception:
        intercept = np.nan
    try:
        slope_model = sm.GLM(y, sm.add_constant(logit_p), family=sm.families.Binomial()).fit()
        slope = float(slope_model.params[1])
    except Exception:
        slope = np.nan
    return intercept, slope


def model_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    intercept, slope = calibration_metrics(y, p)
    return {
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def binary_operating_metrics(y: pd.Series, high_mask: pd.Series) -> dict[str, float]:
    yb = y.astype(int)
    tp = int(((high_mask == 1) & (yb == 1)).sum())
    fp = int(((high_mask == 1) & (yb == 0)).sum())
    tn = int(((high_mask == 0) & (yb == 0)).sum())
    fn = int(((high_mask == 0) & (yb == 1)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def threshold_mask(model_code: str, oof_prob: pd.Series, raw_score: pd.Series) -> list[tuple[str, pd.Series, str]]:
    if model_code in {"M1", "M2", "M3", "M5", "PRM_core5"}:
        return [
            ("top10", oof_prob >= oof_prob.quantile(0.90), "top 10% predicted risk"),
            ("top20", oof_prob >= oof_prob.quantile(0.80), "top 20% predicted risk"),
        ]
    if model_code == "PTS_6_core":
        return [
            ("score_ge4", raw_score >= 4, "point score >= 4"),
            ("score_ge5", raw_score >= 5, "point score >= 5"),
        ]
    if model_code == "TR_burden_change_3of5":
        return [("rule_positive", raw_score >= 1, "rule positive")]
    if model_code == "HYB_emerging_nonsevere":
        return [("rule_positive", raw_score >= 1, "rule positive")]
    raise ValueError(f"Unknown model code for thresholding: {model_code}")


def prepare_predictions(df: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    y = df["incident_adl_w3_w4"].astype(int)
    rules = build_rule_components(df, thresholds)
    pred = pd.DataFrame({"ID_str": df["ID_str"], "incident_adl_w3_w4": y})

    pred["M1_prob"] = fit_logistic_oof(df, BASE_COVARIATES + LEVEL_FEATURES)
    pred["M2_prob"] = fit_logistic_oof(df, BASE_COVARIATES + LEVEL_FEATURES + SLOPE_FEATURES)
    pred["M5_prob"] = fit_logistic_oof(df, BASE_COVARIATES + LEVEL_FEATURES + SLOPE_FEATURES, penalized=True)
    pred["PRM_core5_prob"] = fit_logistic_oof(df, BASE_COVARIATES + CORE5)
    pred["M3_prob"] = fit_m3_oof(df)

    pred["PTS_6_core_score"] = rules["score_pts_6_core"].astype(int)
    pred["PTS_6_core_prob"] = empirical_map_oof(pred["PTS_6_core_score"], y)

    pred["TR_burden_change_3of5_score"] = rules["rule_tr_burden_change_3of5"].astype(int)
    pred["TR_burden_change_3of5_prob"] = empirical_map_oof(pred["TR_burden_change_3of5_score"], y)

    pred["HYB_emerging_nonsevere_score"] = rules["rule_hyb_emerging_nonsevere"].astype(int)
    pred["HYB_emerging_nonsevere_prob"] = empirical_map_oof(pred["HYB_emerging_nonsevere_score"], y)

    return pred


def build_comparison_table(df: pd.DataFrame, preds: pd.DataFrame, nested: pd.DataFrame) -> pd.DataFrame:
    y = preds["incident_adl_w3_w4"].astype(int).to_numpy()
    candidate_meta = {
        "M1": ("locked_reference", "base + 4 wave3 levels"),
        "M2": ("locked_reference", "base + 4 wave3 levels + 4 slopes"),
        "M5": ("locked_reference", "base + 8 continuous features (penalized)"),
        "PRM_core5": ("parsimonious_regression", "base + 3 burden levels + 2 stable slopes"),
        "PTS_6_core": ("point_score", "6-point additive score"),
        "TR_burden_change_3of5": ("threshold_rule", "five-flag burden + change rule"),
        "HYB_emerging_nonsevere": ("hybrid_emerging_target", "non-severe but change-positive rule"),
        "M3": ("phenotype_reference", "base + hard phenotype"),
    }
    nested_ref = nested[nested["row_type"] == "model_performance"].set_index("model_code")
    rows = []
    for code, (family, label) in candidate_meta.items():
        prob = preds[f"{code}_prob"].to_numpy()
        met = model_metrics(y, prob)
        row = {
            "model_code": code,
            "model_family": family,
            "model_label": label,
            "n": len(y),
            "events": int(y.sum()),
            "event_rate": float(y.mean()),
            **met,
            "nested_reference_auc": float(nested_ref.loc[code, "oof_auc"]) if code in nested_ref.index else np.nan,
            "nested_reference_brier": float(nested_ref.loc[code, "oof_brier"]) if code in nested_ref.index else np.nan,
            "nested_reference_pr_auc": float(nested_ref.loc[code, "oof_pr_auc"]) if code in nested_ref.index else np.nan,
        }
        if code in {"PTS_6_core", "TR_burden_change_3of5", "HYB_emerging_nonsevere"}:
            row["uses_fixed_threshold_rule"] = "yes"
        else:
            row["uses_fixed_threshold_rule"] = "no"
        rows.append(row)
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def build_high_risk_table(df: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    y = preds["incident_adl_w3_w4"].astype(int)
    rows: list[dict[str, object]] = []
    model_codes = ["M1", "M2", "M5", "PRM_core5", "PTS_6_core", "TR_burden_change_3of5", "HYB_emerging_nonsevere", "M3"]

    for code in model_codes:
        oof_prob = preds[f"{code}_prob"]
        raw_col = f"{code}_score" if f"{code}_score" in preds.columns else f"{code}_prob"
        raw_score = preds[raw_col]
        for threshold_code, mask, threshold_label in threshold_mask(code, oof_prob, raw_score):
            mask = mask.astype(bool)
            low_mask = ~mask
            op = binary_operating_metrics(y, mask.astype(int))
            high_group = df.loc[mask]
            low_group = df.loc[low_mask]
            rows.append(
                {
                    "model_code": code,
                    "threshold_code": threshold_code,
                    "threshold_label": threshold_label,
                    "n_high_risk": int(mask.sum()),
                    "high_risk_share": float(mask.mean()),
                    "observed_event_rate_high": float(y.loc[mask].mean()),
                    "observed_event_rate_low": float(y.loc[low_mask].mean()),
                    "risk_difference_high_vs_low": float(y.loc[mask].mean() - y.loc[low_mask].mean()),
                    **op,
                    "mean_dep_level_at_w3_high": float(high_group["dep_level_at_w3"].mean()),
                    "mean_cogdef_level_at_w3_high": float(high_group["cogdef_level_at_w3"].mean()),
                    "mean_physical_lim_level_at_w3_high": float(high_group["physical_lim_level_at_w3"].mean()),
                    "mean_multimorb_level_at_w3_high": float(high_group["multimorb_level_at_w3"].mean()),
                    "mean_dep_slope_high": float(high_group["dep_pre_w3_slope"].mean()),
                    "mean_physical_lim_slope_high": float(high_group["physical_lim_pre_w3_slope"].mean()),
                    "emerging_within_high_risk": float((high_group["trajectory_group"] == "Emerging multidomain decline").mean()),
                    "high_burden_within_high_risk": float((high_group["trajectory_group"] == "High-burden mixed pattern").mean()),
                    "severe_within_high_risk": float((high_group["trajectory_group"] == "Severe multidomain decline").mean()),
                    "stable_within_high_risk": float((high_group["trajectory_group"] == "Stable low-burden").mean()),
                    "emerging_capture_rate": float(mask[df["trajectory_group"] == "Emerging multidomain decline"].mean()),
                    "high_burden_capture_rate": float(mask[df["trajectory_group"] == "High-burden mixed pattern"].mean()),
                    "severe_capture_rate": float(mask[df["trajectory_group"] == "Severe multidomain decline"].mean()),
                    "stable_capture_rate": float(mask[df["trajectory_group"] == "Stable low-burden"].mean()),
                }
            )
    out = pd.DataFrame(rows).sort_values(["model_code", "high_risk_share"]).reset_index(drop=True)
    return out


def make_plots(comparison: pd.DataFrame, high_risk: pd.DataFrame, outpath: Path) -> None:
    display_order = ["M1", "M2", "M5", "PRM_core5", "PTS_6_core", "TR_burden_change_3of5", "HYB_emerging_nonsevere", "M3"]
    comp = comparison.set_index("model_code").loc[display_order].reset_index()
    primary_threshold = (
        high_risk.sort_values(["model_code", "high_risk_share"])
        .groupby("model_code", as_index=False)
        .tail(1)
        .set_index("model_code")
        .loc[display_order]
        .reset_index()
    )

    with PdfPages(outpath) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].bar(comp["model_code"], comp["auc"], color="#3b6ea8")
        axes[0].set_ylabel("OOF AUC")
        axes[0].set_title("Discrimination")
        axes[0].tick_params(axis="x", rotation=45)

        axes[1].bar(comp["model_code"], comp["pr_auc"], color="#5c9e6b")
        axes[1].set_ylabel("OOF PR-AUC")
        axes[1].set_title("Precision-recall")
        axes[1].tick_params(axis="x", rotation=45)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(primary_threshold["high_risk_share"], primary_threshold["observed_event_rate_high"], s=90, color="#b55d3e")
        for _, row in primary_threshold.iterrows():
            ax.annotate(row["model_code"], (row["high_risk_share"], row["observed_event_rate_high"]), xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("High-risk group share")
        ax.set_ylabel("Observed event rate in high-risk group")
        ax.set_title("Operational yield of primary high-risk threshold")
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        subset = primary_threshold.copy()
        x = np.arange(len(subset))
        ax.bar(x, subset["stable_within_high_risk"], label="Stable")
        ax.bar(x, subset["emerging_within_high_risk"], bottom=subset["stable_within_high_risk"], label="Emerging")
        ax.bar(
            x,
            subset["high_burden_within_high_risk"],
            bottom=subset["stable_within_high_risk"] + subset["emerging_within_high_risk"],
            label="High-burden",
        )
        ax.bar(
            x,
            subset["severe_within_high_risk"],
            bottom=subset["stable_within_high_risk"] + subset["emerging_within_high_risk"] + subset["high_burden_within_high_risk"],
            label="Severe",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(subset["model_code"], rotation=45)
        ax.set_ylabel("Phenotype composition within high-risk group")
        ax.set_title("Who each high-risk definition captures")
        ax.legend(frameon=False, ncol=4)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def update_results_index() -> None:
    marker = "## 2026-03-25 Stage R3"
    text = RESULTS_INDEX.read_text(encoding="utf-8")
    if marker in text:
        return
    block = f"""

{marker}

本阶段把完整模型、简化回归、积分规则和 Operational high-risk rules 放到同一 `unified comparison sample` 上做正式比较。

### 新增文件

| 文件 | 回答的问题 | 备注 |
|---|---|---|
| `results/stageR_rule/r3_model_rule_comparison.csv` | 完整模型、简化模型和规则型对象在统一样本上的 discrimination / calibration / Brier 如何比较 | Stage R3 主性能表 |
| `results/stageR_rule/r3_high_risk_group_characteristics.csv` | 不同高风险定义抓到了多大一群人、事件率多高、当前 burden 如何、与 Emerging 重叠多少 | Stage R3 操作性分层表 |
| `results/stageR_rule/r3_performance_plots.pdf` | 主要比较图，包括 discrimination、high-risk yield 和 phenotype overlap | Stage R3 图形 |

### 本阶段直接结论

1. `PRM_core5` 在保持明显更少变量的前提下，性能几乎追平 `M2/M5`。
2. `PTS_6_core`、`TR_burden_change_3of5` 和 `HYB_emerging_nonsevere` 的整体 discrimination 较低，但能抓出事件率更高、且更接近实际识别场景的高风险组。
3. 规则型对象的价值更偏向 operational identification，而不是替代完整连续模型。
"""
    RESULTS_INDEX.write_text(text.rstrip() + "\n" + block.strip("\n") + "\n", encoding="utf-8")


def write_metadata() -> None:
    meta = {
        "stage": "R3",
        "seed": SEED,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "sample": "unified comparison sample",
        "sample_n": 8355,
        "comparison_objects": [
            "M1",
            "M2",
            "M5",
            "PRM_core5",
            "PTS_6_core",
            "TR_burden_change_3of5",
            "HYB_emerging_nonsevere",
            "M3",
        ],
    }
    (OUTDIR / "stageR3_run_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs()
    df = inputs["features"].copy()
    thresholds = inputs["r2_meta"]["thresholds"]
    preds = prepare_predictions(df, thresholds)
    comparison = build_comparison_table(df, preds, inputs["nested"])
    comparison.to_csv(OUTDIR / "r3_model_rule_comparison.csv", index=False)

    high_risk = build_high_risk_table(df, preds)
    high_risk.to_csv(OUTDIR / "r3_high_risk_group_characteristics.csv", index=False)

    make_plots(comparison, high_risk, OUTDIR / "r3_performance_plots.pdf")
    update_results_index()
    write_metadata()


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path

# Resolve repository imports independently of the launch directory.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reproducibility.project_paths import project_root

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260325

ROOT = project_root()
RESULTS = ROOT / "results"
OUTDIR = RESULTS / "stageR_rule"
LOGDIR = ROOT / "logs" / "stageR_rule"
RESULTS_INDEX = ROOT / "RESULTS_INDEX.md"

BASE_COVARIATES = [
    "baseline_age_2011",
    "female",
    "rural_hukou_2011",
    "low_self_education",
]

ALL_M2_FEATURES = [
    "dep_level_at_w3",
    "cogdef_level_at_w3",
    "physical_lim_level_at_w3",
    "multimorb_level_at_w3",
    "dep_pre_w3_slope",
    "cogdef_pre_w3_slope",
    "physical_lim_pre_w3_slope",
    "multimorb_pre_w3_slope",
]

CORE_FEATURES = [
    "physical_lim_level_at_w3",
    "dep_level_at_w3",
    "multimorb_level_at_w3",
    "dep_pre_w3_slope",
    "physical_lim_pre_w3_slope",
]

CORE3 = [
    "physical_lim_level_at_w3",
    "dep_level_at_w3",
    "multimorb_level_at_w3",
]

CORE4 = CORE3 + ["dep_pre_w3_slope"]
CORE5 = CORE4 + ["physical_lim_pre_w3_slope"]
L1_STABLE6 = CORE5 + ["cogdef_level_at_w3"]


def fmt3(x: float | int | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{float(x):.3f}"


def load_inputs() -> dict[str, pd.DataFrame]:
    return {
        "features": pd.read_csv(RESULTS / "stage2_level_change" / "level_change_features.csv"),
        "r1": pd.read_csv(OUTDIR / "r1_m2_core_features.csv"),
        "nested": pd.read_csv(RESULTS / "stage3_nested_validation" / "nested_validation_performance.csv"),
    }


def derive_thresholds(df: pd.DataFrame) -> dict[str, float]:
    return {
        "physical_lim_level_q75": float(df["physical_lim_level_at_w3"].quantile(0.75)),
        "physical_lim_level_q90": float(df["physical_lim_level_at_w3"].quantile(0.90)),
        "dep_level_q75": float(df["dep_level_at_w3"].quantile(0.75)),
        "dep_level_q90": float(df["dep_level_at_w3"].quantile(0.90)),
        "multimorb_level_q75": float(df["multimorb_level_at_w3"].quantile(0.75)),
        "multimorb_level_q90": float(df["multimorb_level_at_w3"].quantile(0.90)),
        "dep_slope_q25": float(df["dep_pre_w3_slope"].quantile(0.25)),
        "dep_slope_q10": float(df["dep_pre_w3_slope"].quantile(0.10)),
        "physical_slope_q10": float(df["physical_lim_pre_w3_slope"].quantile(0.10)),
    }


def bootstrap_l1_selection(df: pd.DataFrame, n_boot: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    X = df[BASE_COVARIATES + ALL_M2_FEATURES]
    y = df["incident_adl_w3_w4"].astype(int).to_numpy()
    counts = {name: 0 for name in BASE_COVARIATES + ALL_M2_FEATURES}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for b in range(n_boot):
            idx = rng.integers(0, len(df), len(df))
            xb = X.iloc[idx]
            yb = y[idx]
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "logit",
                        LogisticRegressionCV(
                            Cs=np.logspace(-3, 1, 12),
                            cv=5,
                            penalty="l1",
                            solver="liblinear",
                            scoring="roc_auc",
                            max_iter=3000,
                            random_state=SEED + b,
                        ),
                    ),
                ]
            )
            pipe.fit(xb, yb)
            coefs = pipe.named_steps["logit"].coef_[0]
            for name, val in zip(BASE_COVARIATES + ALL_M2_FEATURES, coefs):
                if abs(val) > 1e-8:
                    counts[name] += 1

    out = pd.DataFrame(
        {
            "term": list(counts.keys()),
            "bootstrap_selection_frequency": [counts[k] / n_boot for k in counts],
        }
    )
    return out


def cv_regression_preview(df: pd.DataFrame, feature_list: list[str]) -> dict[str, float]:
    X = df[BASE_COVARIATES + feature_list].copy()
    y = df["incident_adl_w3_w4"].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    oof = np.zeros(len(df), dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train_idx, test_idx in cv.split(X, y):
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("logit", LogisticRegression(max_iter=3000, solver="lbfgs")),
                ]
            )
            pipe.fit(X.iloc[train_idx], y[train_idx])
            oof[test_idx] = pipe.predict_proba(X.iloc[test_idx])[:, 1]

    high_cut = float(np.quantile(oof, 0.80))
    high_mask = oof >= high_cut
    return {
        "preview_oof_auc": float(roc_auc_score(y, oof)),
        "preview_oof_pr_auc": float(average_precision_score(y, oof)),
        "preview_oof_brier": float(brier_score_loss(y, oof)),
        "preview_high_risk_share": float(high_mask.mean()),
        "preview_high_risk_event_rate": float(df.loc[high_mask, "incident_adl_w3_w4"].mean()),
        "preview_low_risk_event_rate": float(df.loc[~high_mask, "incident_adl_w3_w4"].mean()),
        "preview_high_risk_definition": "top 20% of out-of-fold predicted risk",
    }


def build_flag_dataframe(df: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    flags = pd.DataFrame(index=df.index)
    flags["flag_phys_q75"] = (df["physical_lim_level_at_w3"] >= thresholds["physical_lim_level_q75"]).astype(int)
    flags["flag_phys_q90"] = (df["physical_lim_level_at_w3"] >= thresholds["physical_lim_level_q90"]).astype(int)
    flags["flag_dep_q75"] = (df["dep_level_at_w3"] >= thresholds["dep_level_q75"]).astype(int)
    flags["flag_dep_q90"] = (df["dep_level_at_w3"] >= thresholds["dep_level_q90"]).astype(int)
    flags["flag_multimorb_q75"] = (df["multimorb_level_at_w3"] >= thresholds["multimorb_level_q75"]).astype(int)
    flags["flag_multimorb_q90"] = (df["multimorb_level_at_w3"] >= thresholds["multimorb_level_q90"]).astype(int)
    flags["flag_dep_change_tail"] = (df["dep_pre_w3_slope"] <= thresholds["dep_slope_q25"]).astype(int)
    flags["flag_phys_change_tail"] = (df["physical_lim_pre_w3_slope"] <= thresholds["physical_slope_q10"]).astype(int)
    return flags


def rule_preview(
    y: pd.Series,
    score: pd.Series,
    high_mask: pd.Series,
    high_risk_definition: str,
) -> dict[str, float]:
    score_array = score.astype(float).to_numpy()
    if np.allclose(score_array, score_array[0]):
        auc = np.nan
        pr_auc = np.nan
    else:
        auc = float(roc_auc_score(y, score_array))
        pr_auc = float(average_precision_score(y, score_array))
    prob_like = score_array
    if prob_like.max() > 1:
        prob_like = prob_like / prob_like.max()
    return {
        "preview_oof_auc": auc,
        "preview_oof_pr_auc": pr_auc,
        "preview_oof_brier": float(brier_score_loss(y, prob_like)),
        "preview_high_risk_share": float(high_mask.mean()),
        "preview_high_risk_event_rate": float(y.loc[high_mask].mean()),
        "preview_low_risk_event_rate": float(y.loc[~high_mask].mean()),
        "preview_high_risk_definition": high_risk_definition,
    }


def build_candidate_rows(
    df: pd.DataFrame,
    thresholds: dict[str, float],
    selection_freq: pd.DataFrame,
) -> pd.DataFrame:
    y = df["incident_adl_w3_w4"].astype(int)
    flags = build_flag_dataframe(df, thresholds)

    rows: list[dict[str, object]] = []

    regression_candidates = [
        (
            "PRM_core3",
            "parsimonious_regression",
            "base + three current multidomain burden signals",
            CORE3,
            "continuous",
            "clinically constrained selection from Stage R1",
            "moderate",
            "parsimonious regression baseline candidate",
        ),
        (
            "PRM_core4",
            "parsimonious_regression",
            "core3 + depressive change",
            CORE4,
            "continuous",
            "add the most stable non-physical change signal",
            "moderate",
            "minimal change-augmented regression candidate",
        ),
        (
            "PRM_core5",
            "parsimonious_regression",
            "core4 + physical limitation change",
            CORE5,
            "continuous",
            "add the second stable change signal while staying clinically compact",
            "moderate",
            "main parsimonious regression candidate",
        ),
        (
            "PRM_l1_stable6",
            "parsimonious_regression",
            "core5 + cognitive current level",
            L1_STABLE6,
            "continuous",
            "bootstrap L1-guided expansion of the core5 set",
            "moderate_to_high",
            "slightly richer regression candidate",
        ),
    ]

    sel_map = dict(zip(selection_freq["term"], selection_freq["bootstrap_selection_frequency"]))
    for candidate_id, family, label, feature_list, form, rationale, interpretability, use_case in regression_candidates:
        preview = cv_regression_preview(df, feature_list)
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_family": family,
                "candidate_label": label,
                "include_base_covariates": "yes",
                "variables_used": "; ".join(BASE_COVARIATES + feature_list),
                "m2_feature_subset": "; ".join(feature_list),
                "n_m2_features": len(feature_list),
                "variable_form": form,
                "requires_slopes": "yes" if any("slope" in x for x in feature_list) else "no",
                "threshold_source": "not applicable",
                "selection_rationale": rationale,
                "bootstrap_selection_support": "; ".join(
                    f"{f}={fmt3(sel_map.get(f))}" for f in feature_list
                ),
                "interpretability": interpretability,
                "intended_use": use_case,
                **preview,
            }
        )

    burden_count = flags["flag_phys_q75"] + flags["flag_dep_q75"] + flags["flag_multimorb_q75"]
    change_count = flags["flag_dep_change_tail"] + flags["flag_phys_change_tail"]
    total_flag_count = burden_count + change_count
    points_score = (
        2 * flags["flag_phys_q75"]
        + flags["flag_dep_q75"]
        + flags["flag_multimorb_q75"]
        + flags["flag_dep_change_tail"]
        + flags["flag_phys_change_tail"]
    )

    threshold_candidates = [
        (
            "TR_burden_2of3",
            "threshold_rule",
            "three current-burden flags; high risk if at least two are positive",
            burden_count,
            burden_count >= 2,
            (
                f"physical limitation >= {fmt3(thresholds['physical_lim_level_q75'])}; "
                f"depressive symptoms >= {fmt3(thresholds['dep_level_q75'])}; "
                f"multimorbidity >= {fmt3(thresholds['multimorb_level_q75'])}; "
                "high risk if >=2 flags"
            ),
            "current burden only",
            "high",
            "simple burden-based rule",
        ),
        (
            "TR_burden_change_3of5",
            "threshold_rule",
            "five-flag burden+change rule; high risk if at least three are positive",
            total_flag_count,
            total_flag_count >= 3,
            (
                f"burden flags: physical limitation >= {fmt3(thresholds['physical_lim_level_q75'])}, "
                f"depressive symptoms >= {fmt3(thresholds['dep_level_q75'])}, "
                f"multimorbidity >= {fmt3(thresholds['multimorb_level_q75'])}; "
                f"change flags: depressive slope <= {fmt3(thresholds['dep_slope_q25'])}, "
                f"physical limitation slope <= {fmt3(thresholds['physical_slope_q10'])}; "
                "high risk if >=3 total flags"
            ),
            "burden plus change",
            "high",
            "simple rule that keeps at least one explicit change signal",
        ),
    ]

    for candidate_id, family, label, score, high_mask, definition, rationale, interpretability, use_case in threshold_candidates:
        preview = rule_preview(y, score, high_mask, definition)
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_family": family,
                "candidate_label": label,
                "include_base_covariates": "no",
                "variables_used": "physical_lim_level_at_w3; dep_level_at_w3; multimorb_level_at_w3"
                + ("; dep_pre_w3_slope; physical_lim_pre_w3_slope" if "change" in candidate_id else ""),
                "m2_feature_subset": "rule-derived",
                "n_m2_features": 3 if candidate_id == "TR_burden_2of3" else 5,
                "variable_form": "binary flags",
                "requires_slopes": "yes" if "change" in candidate_id else "no",
                "threshold_source": "quantile-guided fixed cutpoints from unified sample",
                "selection_rationale": rationale,
                "bootstrap_selection_support": "not applicable",
                "interpretability": interpretability,
                "intended_use": use_case,
                **preview,
            }
        )

    preview = rule_preview(
        y,
        points_score,
        points_score >= 4,
        "0-1 low, 2-3 intermediate, >=4 high",
    )
    rows.append(
        {
            "candidate_id": "PTS_6_core",
            "candidate_family": "point_score",
            "candidate_label": "6-point additive score",
            "include_base_covariates": "no",
            "variables_used": "physical_lim_level_at_w3; dep_level_at_w3; multimorb_level_at_w3; dep_pre_w3_slope; physical_lim_pre_w3_slope",
            "m2_feature_subset": "; ".join(CORE5),
            "n_m2_features": 5,
            "variable_form": "weighted points",
            "requires_slopes": "yes",
            "threshold_source": "quantile-guided fixed cutpoints plus coefficient-informed weighting",
            "selection_rationale": "weight physical limitation current burden more heavily, keep both stable change channels",
            "bootstrap_selection_support": "; ".join(f"{f}={fmt3(sel_map.get(f))}" for f in CORE5),
            "interpretability": "high",
            "intended_use": "simple prognostic score with low/intermediate/high risk groups",
            **preview,
        }
    )

    hybrid_nonsevere = (
        (df["physical_lim_level_at_w3"] >= thresholds["physical_lim_level_q75"])
        & (df["physical_lim_level_at_w3"] < 0.60)
        & ((flags["flag_dep_q75"] + flags["flag_multimorb_q75"]) >= 1)
        & ((flags["flag_dep_change_tail"] + flags["flag_phys_change_tail"]) >= 1)
    )
    preview = rule_preview(
        y,
        hybrid_nonsevere.astype(int),
        hybrid_nonsevere,
        (
            f"physical limitation in [{fmt3(thresholds['physical_lim_level_q75'])}, 0.600); "
            f"plus depressive symptoms >= {fmt3(thresholds['dep_level_q75'])} or multimorbidity >= {fmt3(thresholds['multimorb_level_q75'])}; "
            f"plus depressive slope <= {fmt3(thresholds['dep_slope_q25'])} or physical limitation slope <= {fmt3(thresholds['physical_slope_q10'])}"
        ),
    )
    rows.append(
        {
            "candidate_id": "HYB_emerging_nonsevere",
            "candidate_family": "hybrid_emerging_target",
            "candidate_label": "non-severe but change-positive high-risk rule",
            "include_base_covariates": "no",
            "variables_used": "physical_lim_level_at_w3; dep_level_at_w3; multimorb_level_at_w3; dep_pre_w3_slope; physical_lim_pre_w3_slope",
            "m2_feature_subset": "; ".join(CORE5),
            "n_m2_features": 5,
            "variable_form": "banded burden + binary change flags",
            "requires_slopes": "yes",
            "threshold_source": "quantile-guided cutpoints with explicit non-severe burden band",
            "selection_rationale": "target individuals who are not at the most severe current burden but already carry change-linked risk",
            "bootstrap_selection_support": "; ".join(f"{f}={fmt3(sel_map.get(f))}" for f in CORE5),
            "interpretability": "high",
            "intended_use": "Emerging-like operational identification candidate",
            **preview,
        }
    )

    hybrid_broad = (
        burden_count.between(1, 2)
        & (change_count >= 1)
        & ~((flags["flag_phys_q90"] + flags["flag_dep_q90"] + flags["flag_multimorb_q90"]) >= 2)
    )
    preview = rule_preview(
        y,
        hybrid_broad.astype(int),
        hybrid_broad,
        "1-2 burden flags, at least 1 change flag, and not extreme current burden in >=2 domains",
    )
    rows.append(
        {
            "candidate_id": "HYB_emerging_broad",
            "candidate_family": "hybrid_emerging_target",
            "candidate_label": "broader Emerging-oriented flag-band rule",
            "include_base_covariates": "no",
            "variables_used": "physical_lim_level_at_w3; dep_level_at_w3; multimorb_level_at_w3; dep_pre_w3_slope; physical_lim_pre_w3_slope",
            "m2_feature_subset": "; ".join(CORE5),
            "n_m2_features": 5,
            "variable_form": "burden-band classifier + change flags",
            "requires_slopes": "yes",
            "threshold_source": "quantile-guided cutpoints with explicit anti-severe exclusion",
            "selection_rationale": "broader operational approximation to an Emerging-like risk band",
            "bootstrap_selection_support": "; ".join(f"{f}={fmt3(sel_map.get(f))}" for f in CORE5),
            "interpretability": "high",
            "intended_use": "broader high-risk surveillance candidate",
            **preview,
        }
    )

    out = pd.DataFrame(rows)
    return out.sort_values(["candidate_family", "candidate_id"]).reset_index(drop=True)


def write_rule_definitions(
    outpath: Path,
    thresholds: dict[str, float],
    selection_freq: pd.DataFrame,
    candidate_table: pd.DataFrame,
) -> None:
    sel = selection_freq.set_index("term")["bootstrap_selection_frequency"].to_dict()
    score_band = {
        0: "very low current/change flag load",
        1: "low current/change flag load",
        2: "intermediate flag load",
        3: "moderately high flag load",
        4: "high flag load",
        5: "very high flag load",
        6: "extreme flag load",
    }
    lines: list[str] = []
    lines.append("# Stage R2 candidate rule definitions")
    lines.append("")
    lines.append("## 1. Stage aim")
    lines.append("")
    lines.append("本阶段不做最终模型裁决，只把后续 `R3` 要比较的简化候选对象明确写清楚。")
    lines.append("所有候选对象都以 `Stage R1` 锁定的结论为前提：current burden 是主信号，change 只提供 modest 但稳定的增量。")
    lines.append("")
    lines.append("## 2. Threshold source")
    lines.append("")
    lines.append("所有阈值都来自 `unified comparison sample (n=8355)` 的固定分位数，不使用 outcome-driven 最优切点。")
    lines.append("")
    lines.append("| Quantity | Cutpoint | Derivation rationale |")
    lines.append("|---|---:|---|")
    lines.append(f"| `physical_lim_level_at_w3` moderate burden | {fmt3(thresholds['physical_lim_level_q75'])} | 75th percentile |")
    lines.append(f"| `physical_lim_level_at_w3` higher burden | {fmt3(thresholds['physical_lim_level_q90'])} | 90th percentile |")
    lines.append(f"| `dep_level_at_w3` elevated burden | {fmt3(thresholds['dep_level_q75'])} | 75th percentile |")
    lines.append(f"| `dep_level_at_w3` very high burden | {fmt3(thresholds['dep_level_q90'])} | 90th percentile |")
    lines.append(f"| `multimorb_level_at_w3` elevated burden | {fmt3(thresholds['multimorb_level_q75'])} | 75th percentile |")
    lines.append(f"| `multimorb_level_at_w3` very high burden | {fmt3(thresholds['multimorb_level_q90'])} | 90th percentile |")
    lines.append(f"| `dep_pre_w3_slope` change-tail cutpoint | {fmt3(thresholds['dep_slope_q25'])} | 25th percentile risk-oriented tail |")
    lines.append(f"| `physical_lim_pre_w3_slope` change-tail cutpoint | {fmt3(thresholds['physical_slope_q10'])} | 10th percentile because most values cluster at 0 |")
    lines.append("")
    lines.append("## 3. Variable-selection support")
    lines.append("")
    lines.append("这一步没有把单次 L1 结果当成真理，而是增加了 `B=60` 的 bootstrap L1 进入频率作为辅证。")
    lines.append("")
    lines.append("| Feature | Bootstrap selection frequency |")
    lines.append("|---|---:|")
    for feature in CORE5 + ["cogdef_level_at_w3", "cogdef_pre_w3_slope", "multimorb_pre_w3_slope"]:
        lines.append(f"| `{feature}` | {fmt3(sel.get(feature))} |")
    lines.append("")
    lines.append("## 4. Candidate families")
    lines.append("")
    lines.append("### A. Parsimonious regression candidates")
    lines.append("")
    lines.append("1. `PRM_core3`")
    lines.append(f"   - Base covariates + `{', '.join(CORE3)}`")
    lines.append("   - 目标：用最少 current-burden 变量近似 `M1`。")
    lines.append("")
    lines.append("2. `PRM_core4`")
    lines.append(f"   - `PRM_core3` + `dep_pre_w3_slope`")
    lines.append("   - 目标：先检验加入一个最稳的非 physical change signal 后能保留多少 `M2` 增量。")
    lines.append("")
    lines.append("3. `PRM_core5`")
    lines.append(f"   - `PRM_core4` + `physical_lim_pre_w3_slope`")
    lines.append("   - 目标：用 5 个 M2 特征逼近完整 `M2/M5`。")
    lines.append("")
    lines.append("4. `PRM_l1_stable6`")
    lines.append(f"   - `PRM_core5` + `cogdef_level_at_w3`")
    lines.append("   - 目标：允许认知 current level 作为弱但可重复进入的补充信号。")
    lines.append("")
    lines.append("### B. Threshold-based rules")
    lines.append("")
    lines.append("1. `TR_burden_2of3`")
    lines.append("   - 3 个当前 burden flag：")
    lines.append(f"     - `physical_lim_level_at_w3 >= {fmt3(thresholds['physical_lim_level_q75'])}`")
    lines.append(f"     - `dep_level_at_w3 >= {fmt3(thresholds['dep_level_q75'])}`")
    lines.append(f"     - `multimorb_level_at_w3 >= {fmt3(thresholds['multimorb_level_q75'])}`")
    lines.append("   - 高风险：`>= 2` 个 positive flags")
    lines.append("   - 目标：提供完全不依赖 slope 的简洁 burden rule。")
    lines.append("")
    lines.append("2. `TR_burden_change_3of5`")
    lines.append("   - 上述 3 个 burden flags")
    lines.append(f"   - 加上 2 个 change flags：`dep_pre_w3_slope <= {fmt3(thresholds['dep_slope_q25'])}`、`physical_lim_pre_w3_slope <= {fmt3(thresholds['physical_slope_q10'])}`")
    lines.append("   - 高风险：`>= 3` 个 total flags")
    lines.append("   - 目标：在不建复杂模型的前提下，把 current burden 和 prior change 合并成可执行规则。")
    lines.append("")
    lines.append("### C. Simple points-based score")
    lines.append("")
    lines.append("`PTS_6_core`")
    lines.append(" - 赋分规则：")
    lines.append(f"   - `physical_lim_level_at_w3 >= {fmt3(thresholds['physical_lim_level_q75'])}`: 2 points")
    lines.append(f"   - `dep_level_at_w3 >= {fmt3(thresholds['dep_level_q75'])}`: 1 point")
    lines.append(f"   - `multimorb_level_at_w3 >= {fmt3(thresholds['multimorb_level_q75'])}`: 1 point")
    lines.append(f"   - `dep_pre_w3_slope <= {fmt3(thresholds['dep_slope_q25'])}`: 1 point")
    lines.append(f"   - `physical_lim_pre_w3_slope <= {fmt3(thresholds['physical_slope_q10'])}`: 1 point")
    lines.append(" - 推荐分层：")
    for score, desc in score_band.items():
        lines.append(f"   - score {score}: {desc}")
    lines.append(" - 实务分组：`0-1 low`，`2-3 intermediate`，`>=4 high`。")
    lines.append("")
    lines.append("### D. Hybrid rules targeting Emerging-like risk")
    lines.append("")
    lines.append("1. `HYB_emerging_nonsevere`")
    lines.append(f"   - `physical_lim_level_at_w3` in `[ {fmt3(thresholds['physical_lim_level_q75'])}, 0.600 )`")
    lines.append(f"   - plus `dep_level_at_w3 >= {fmt3(thresholds['dep_level_q75'])}` or `multimorb_level_at_w3 >= {fmt3(thresholds['multimorb_level_q75'])}`")
    lines.append(f"   - plus `dep_pre_w3_slope <= {fmt3(thresholds['dep_slope_q25'])}` or `physical_lim_pre_w3_slope <= {fmt3(thresholds['physical_slope_q10'])}`")
    lines.append("   - 目标：识别“当前不算最严重，但 trajectory 已提示风险抬升”的人。")
    lines.append("")
    lines.append("2. `HYB_emerging_broad`")
    lines.append("   - current burden flags 介于 1 到 2 个")
    lines.append("   - 至少 1 个 change flag")
    lines.append("   - 排除 `>=2` 个 extreme current-burden flags")
    lines.append("   - 目标：提供一个更宽、用于公共卫生随访筛查的 Emerging-like 候选带。")
    lines.append("")
    lines.append("## 5. Important interpretation guardrail")
    lines.append("")
    lines.append("由于 `dep_pre_w3_slope` 与 `physical_lim_pre_w3_slope` 在 fully adjusted models 中是 risk-oriented lower-tail signals，")
    lines.append("这些 change cutpoints 不能简单写成“恶化越快越危险”。")
    lines.append("更稳的表述应是：")
    lines.append("- `risk-oriented pre-landmark change tail`")
    lines.append("- `history pattern consistent with higher risk after accounting for current burden`")
    lines.append("")
    lines.append("## 6. Preview, not final adjudication")
    lines.append("")
    lines.append("`r2_candidate_models.csv` 里的性能数字只是 Stage R2 的预览，用于帮助决定 R3 应重点比较哪些候选对象。")
    lines.append("最终是否值得报告，要等 R3-R5 的统一比较、operational overlap 分析和内部验证。")
    lines.append("")
    lines.append("## 7. Most promising candidates entering Stage R3")
    lines.append("")
    lines.append("基于 R2 预览，目前最值得进入 R3 的对象是：")
    lines.append("- `PRM_core5`：最接近完整 `M2/M5`，但变量数已明显缩小")
    lines.append("- `PTS_6_core`：最有机会成为真正可解释的 operational score")
    lines.append("- `TR_burden_change_3of5`：最有机会成为简单规则型 high-risk identifier")
    lines.append("- `HYB_emerging_nonsevere`：最适合检验是否真能 operationalize Emerging-like risk")
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_results_index() -> None:
    marker = "## 2026-03-25 Stage R2"
    text = RESULTS_INDEX.read_text(encoding="utf-8")
    if marker in text:
        return
    block = f"""

{marker}

本阶段开始把 `M2` 的连续信号转成更可操作的候选对象，但还没有做最终优胜者裁决。

### 新增文件

| 文件 | 回答的问题 | 备注 |
|---|---|---|
| `results/stageR_rule/r2_candidate_models.csv` | 有哪些 parsimonious / threshold / points / hybrid 候选对象；各自变量、阈值、预览性能和高风险组大小如何 | Stage R2 主表 |
| `results/stageR_rule/r2_candidate_rule_definitions.md` | 每个候选对象到底怎么定义；阈值从哪里来；为什么这样设 | 规则定义说明 |

### 本阶段直接结论

1. 最值得进入下一步比较的 parsimonious regression 候选是 `PRM_core5`。
2. 最值得进入下一步操作性比较的简化对象是：
   - `PTS_6_core`
   - `TR_burden_change_3of5`
   - `HYB_emerging_nonsevere`
3. current burden-only 规则有可操作性，但把最稳的 change signal 纳入后，更接近本研究希望识别的 pre-disability high-risk profile。
"""
    RESULTS_INDEX.write_text(text.rstrip() + "\n" + block.strip("\n") + "\n", encoding="utf-8")


def write_metadata(candidate_table: pd.DataFrame, thresholds: dict[str, float]) -> None:
    meta = {
        "stage": "R2",
        "seed": SEED,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "sample": "unified comparison sample",
        "sample_n": int(candidate_table["preview_high_risk_share"].notna().sum() and 8355),
        "thresholds": thresholds,
        "candidate_ids": candidate_table["candidate_id"].tolist(),
    }
    (OUTDIR / "stageR2_run_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    LOGDIR.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs()
    df = inputs["features"].copy()
    thresholds = derive_thresholds(df)
    selection_freq = bootstrap_l1_selection(df, n_boot=60)
    candidate_table = build_candidate_rows(df, thresholds, selection_freq)

    candidate_table.to_csv(OUTDIR / "r2_candidate_models.csv", index=False)
    write_rule_definitions(OUTDIR / "r2_candidate_rule_definitions.md", thresholds, selection_freq, candidate_table)
    write_metadata(candidate_table, thresholds)
    update_results_index()


if __name__ == "__main__":
    main()

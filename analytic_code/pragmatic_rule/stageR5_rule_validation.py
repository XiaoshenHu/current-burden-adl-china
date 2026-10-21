from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

# Resolve repository imports independently of the launch directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reproducibility.project_paths import project_root

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


SEED = 20260325

ROOT = project_root()
OUTDIR = ROOT / "results" / "stageR_rule"
RESULTS_INDEX = ROOT / "RESULTS_INDEX.md"

from analytic_code.model_validation_and_representation import stage4_missing_selection as s4  # noqa: E402


def fmt3(x: float | int | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{float(x):.3f}"


def weighted_mean(x: pd.Series | np.ndarray, w: pd.Series | np.ndarray | None = None) -> float:
    arr = np.asarray(x, dtype=float)
    if w is None:
        return float(np.nanmean(arr))
    ww = np.asarray(w, dtype=float)
    return float(np.average(arr, weights=ww))


def weighted_brier(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    err = (np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2
    if sample_weight is None:
        return float(err.mean())
    return float(np.average(err, weights=np.asarray(sample_weight, dtype=float)))


def calibration_metrics(y: np.ndarray, p: np.ndarray, sample_weight: np.ndarray | None = None) -> tuple[float, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    logit_p = np.log(p / (1 - p))
    try:
        kwargs = {}
        if sample_weight is not None:
            kwargs["freq_weights"] = np.asarray(sample_weight, dtype=float)
        intercept_model = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=logit_p, **kwargs).fit()
        intercept = float(intercept_model.params[0])
    except Exception:
        intercept = np.nan
    try:
        kwargs = {}
        if sample_weight is not None:
            kwargs["freq_weights"] = np.asarray(sample_weight, dtype=float)
        slope_model = sm.GLM(y, sm.add_constant(logit_p), family=sm.families.Binomial(), **kwargs).fit()
        slope = float(slope_model.params[1])
    except Exception:
        slope = np.nan
    return intercept, slope


def weighted_event_rate(y: pd.Series, weights: pd.Series | None = None) -> float:
    if weights is None:
        return float(y.mean())
    return weighted_mean(y, weights)


def prepare_complete_case_df() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "results" / "stage2_level_change" / "level_change_features.csv")
    return df.copy()


def prepare_ipw_df() -> pd.DataFrame:
    base_df, assignments = s4.load_base_data()
    weighted_df, _ = s4.build_selection_weights(base_df, assignments)
    _, unified = s4.current_samples_with_slopes_and_weights(weighted_df)
    out = unified.rename(
        columns={
            "dep_w3": "dep_level_at_w3",
            "cogdef_w3": "cogdef_level_at_w3",
            "physical_lim_w3": "physical_lim_level_at_w3",
            "multimorb_w3": "multimorb_level_at_w3",
        }
    ).copy()
    keep = [
        "ID_str",
        "incident_adl_w3_w4",
        "baseline_age_2011",
        "female",
        "rural_hukou_2011",
        "low_self_education",
        "trajectory_group",
        "dep_level_at_w3",
        "cogdef_level_at_w3",
        "physical_lim_level_at_w3",
        "multimorb_level_at_w3",
        "dep_pre_w3_slope",
        "cogdef_pre_w3_slope",
        "physical_lim_pre_w3_slope",
        "multimorb_pre_w3_slope",
        "selection_weight_truncated",
    ]
    return out[keep].copy()


def derive_thresholds(train_df: pd.DataFrame) -> dict[str, float]:
    return {
        "physical_lim_level_q75": float(train_df["physical_lim_level_at_w3"].quantile(0.75)),
        "dep_level_q75": float(train_df["dep_level_at_w3"].quantile(0.75)),
        "multimorb_level_q75": float(train_df["multimorb_level_at_w3"].quantile(0.75)),
        "dep_slope_q25": float(train_df["dep_pre_w3_slope"].quantile(0.25)),
        "physical_slope_q10": float(train_df["physical_lim_pre_w3_slope"].quantile(0.10)),
    }


def build_rule_states(df: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["flag_phys_q75"] = (df["physical_lim_level_at_w3"] >= thresholds["physical_lim_level_q75"]).astype(int)
    out["flag_dep_q75"] = (df["dep_level_at_w3"] >= thresholds["dep_level_q75"]).astype(int)
    out["flag_multimorb_q75"] = (df["multimorb_level_at_w3"] >= thresholds["multimorb_level_q75"]).astype(int)
    out["flag_dep_change"] = (df["dep_pre_w3_slope"] <= thresholds["dep_slope_q25"]).astype(int)
    out["flag_phys_change"] = (df["physical_lim_pre_w3_slope"] <= thresholds["physical_slope_q10"]).astype(int)
    out["PTS_6_core_score"] = (
        2 * out["flag_phys_q75"]
        + out["flag_dep_q75"]
        + out["flag_multimorb_q75"]
        + out["flag_dep_change"]
        + out["flag_phys_change"]
    )
    out["PTS_6_core_high"] = (out["PTS_6_core_score"] >= 4).astype(int)
    out["PTS_6_core_lowref"] = (out["PTS_6_core_score"] <= 1).astype(int)
    out["HYB_emerging_nonsevere_high"] = (
        (df["physical_lim_level_at_w3"] >= thresholds["physical_lim_level_q75"])
        & (df["physical_lim_level_at_w3"] < 0.60)
        & ((out["flag_dep_q75"] + out["flag_multimorb_q75"]) >= 1)
        & ((out["flag_dep_change"] + out["flag_phys_change"]) >= 1)
    ).astype(int)
    return out


def map_state_to_risk(train_state: pd.Series, train_y: pd.Series, train_w: pd.Series | None, test_state: pd.Series) -> np.ndarray:
    train_state = pd.Series(train_state).reset_index(drop=True)
    train_y = pd.Series(train_y).reset_index(drop=True)
    if train_w is not None:
        train_w = pd.Series(train_w).reset_index(drop=True)
    test_state = pd.Series(test_state).reset_index(drop=True)
    mapping = {}
    global_rate = weighted_event_rate(train_y, train_w)
    for state_value in sorted(train_state.unique()):
        mask = (train_state.to_numpy() == state_value)
        y_sub = train_y.iloc[np.flatnonzero(mask)]
        w_sub = None if train_w is None else train_w.iloc[np.flatnonzero(mask)]
        mapping[state_value] = weighted_event_rate(y_sub, w_sub)
    pred = test_state.map(mapping).fillna(global_rate).astype(float).to_numpy()
    return np.clip(pred, 1e-6, 1 - 1e-6)


def validate_rules(
    df: pd.DataFrame,
    scenario: str,
    scenario_label: str,
    weight_col: str | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    y = df["incident_adl_w3_w4"].astype(int).reset_index(drop=True)
    weights = None if weight_col is None else df[weight_col].astype(float).reset_index(drop=True)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

    oof_prob = {
        "PTS_6_core": np.zeros(len(df), dtype=float),
        "HYB_emerging_nonsevere": np.zeros(len(df), dtype=float),
    }
    fold_rows = []

    for fold_id, (train_idx, test_idx) in enumerate(cv.split(df, y), start=1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        train_y = y.iloc[train_idx].reset_index(drop=True)
        test_y = y.iloc[test_idx].reset_index(drop=True)
        train_w = None if weights is None else weights.iloc[train_idx].reset_index(drop=True)
        test_w = None if weights is None else weights.iloc[test_idx].reset_index(drop=True)

        thresholds = derive_thresholds(train_df)
        train_states = build_rule_states(train_df, thresholds)
        test_states = build_rule_states(test_df, thresholds)

        oof_prob["PTS_6_core"][test_idx] = map_state_to_risk(
            train_states["PTS_6_core_score"], train_y, train_w, test_states["PTS_6_core_score"]
        )
        oof_prob["HYB_emerging_nonsevere"][test_idx] = map_state_to_risk(
            train_states["HYB_emerging_nonsevere_high"], train_y, train_w, test_states["HYB_emerging_nonsevere_high"]
        )

        for rule_id, high_col in [
            ("PTS_6_core", "PTS_6_core_high"),
            ("HYB_emerging_nonsevere", "HYB_emerging_nonsevere_high"),
        ]:
            high_mask = test_states[high_col].astype(bool)
            low_mask = test_states["PTS_6_core_lowref"].astype(bool)
            if high_mask.sum() == 0:
                high_rate = np.nan
                high_share = 0.0
                emerging_share = np.nan
            else:
                high_rate = weighted_event_rate(test_y.loc[high_mask], None if test_w is None else test_w.loc[high_mask])
                high_share = float(high_mask.mean())
                emerging_share = weighted_event_rate(
                    test_df.loc[high_mask, "trajectory_group"].eq("Emerging multidomain decline").astype(float),
                    None if test_w is None else test_w.loc[high_mask],
                )
            if low_mask.sum() == 0:
                risk_diff = np.nan
            else:
                low_rate = weighted_event_rate(test_y.loc[low_mask], None if test_w is None else test_w.loc[low_mask])
                risk_diff = float(high_rate - low_rate) if not np.isnan(high_rate) else np.nan
            fold_rows.append(
                {
                    "scenario": scenario,
                    "rule_id": rule_id,
                    "fold": fold_id,
                    "validation_n": int(len(test_df)),
                    "validation_events": int(test_y.sum()),
                    "high_risk_share": high_share,
                    "high_risk_event_rate": high_rate,
                    "emerging_within_high_risk": emerging_share,
                    "risk_difference_vs_lowref": risk_diff,
                }
            )

    fold_df = pd.DataFrame(fold_rows)

    thresholds_full = derive_thresholds(df)
    apparent_states = build_rule_states(df, thresholds_full)
    apparent_prob = {
        "PTS_6_core": map_state_to_risk(
            apparent_states["PTS_6_core_score"], y, weights, apparent_states["PTS_6_core_score"]
        ),
        "HYB_emerging_nonsevere": map_state_to_risk(
            apparent_states["HYB_emerging_nonsevere_high"], y, weights, apparent_states["HYB_emerging_nonsevere_high"]
        ),
    }

    rows = []
    for rule_id, high_col in [
        ("PTS_6_core", "PTS_6_core_high"),
        ("HYB_emerging_nonsevere", "HYB_emerging_nonsevere_high"),
    ]:
        sw = None if weights is None else weights.to_numpy()
        app_auc = float(roc_auc_score(y, apparent_prob[rule_id], sample_weight=sw))
        try:
            app_pr = float(average_precision_score(y, apparent_prob[rule_id], sample_weight=sw))
        except TypeError:
            app_pr = float(average_precision_score(y, apparent_prob[rule_id]))
        app_brier = weighted_brier(y.to_numpy(), apparent_prob[rule_id], sw)
        app_ci, app_cs = calibration_metrics(y.to_numpy(), apparent_prob[rule_id], sw)

        oof_auc = float(roc_auc_score(y, oof_prob[rule_id], sample_weight=sw))
        try:
            oof_pr = float(average_precision_score(y, oof_prob[rule_id], sample_weight=sw))
        except TypeError:
            oof_pr = float(average_precision_score(y, oof_prob[rule_id]))
        oof_brier = weighted_brier(y.to_numpy(), oof_prob[rule_id], sw)
        oof_ci, oof_cs = calibration_metrics(y.to_numpy(), oof_prob[rule_id], sw)

        high_mask = apparent_states[high_col].astype(bool).to_numpy()
        low_mask = apparent_states["PTS_6_core_lowref"].astype(bool).to_numpy()
        high_idx = np.flatnonzero(high_mask)
        low_idx = np.flatnonzero(low_mask)
        y_pos = y.reset_index(drop=True)
        w_pos = None if weights is None else weights.reset_index(drop=True)
        df_pos = df.reset_index(drop=True)
        app_high_rate = weighted_event_rate(
            y_pos.iloc[high_idx],
            None if w_pos is None else w_pos.iloc[high_idx],
        )
        app_low_rate = weighted_event_rate(
            y_pos.iloc[low_idx],
            None if w_pos is None else w_pos.iloc[low_idx],
        )
        app_emerging = weighted_event_rate(
            df_pos.iloc[high_idx]["trajectory_group"].eq("Emerging multidomain decline").astype(float),
            None if w_pos is None else w_pos.iloc[high_idx],
        )

        sub = fold_df[(fold_df["scenario"] == scenario) & (fold_df["rule_id"] == rule_id)]
        rows.append(
            {
                "scenario": scenario,
                "scenario_label": scenario_label,
                "rule_id": rule_id,
                "n": int(len(df)),
                "events": int(y.sum()),
                "event_rate": weighted_event_rate(y, weights),
                "weighting": "IPW-truncated" if weight_col else "none",
                "apparent_auc": app_auc,
                "oof_auc": oof_auc,
                "optimism_auc": app_auc - oof_auc,
                "apparent_pr_auc": app_pr,
                "oof_pr_auc": oof_pr,
                "optimism_pr_auc": app_pr - oof_pr,
                "apparent_brier": app_brier,
                "oof_brier": oof_brier,
                "optimism_brier": app_brier - oof_brier,
                "apparent_calibration_intercept": app_ci,
                "oof_calibration_intercept": oof_ci,
                "apparent_calibration_slope": app_cs,
                "oof_calibration_slope": oof_cs,
                "apparent_high_risk_share": float(high_mask.mean()),
                "apparent_high_risk_event_rate": app_high_rate,
                "apparent_emerging_within_high": app_emerging,
                "apparent_risk_difference_vs_lowref": float(app_high_rate - app_low_rate),
                "oof_high_risk_share_mean": float(sub["high_risk_share"].mean()),
                "oof_high_risk_share_sd": float(sub["high_risk_share"].std(ddof=1)),
                "oof_high_risk_event_rate_mean": float(sub["high_risk_event_rate"].mean()),
                "oof_high_risk_event_rate_sd": float(sub["high_risk_event_rate"].std(ddof=1)),
                "oof_emerging_within_high_mean": float(sub["emerging_within_high_risk"].mean()),
                "oof_emerging_within_high_sd": float(sub["emerging_within_high_risk"].std(ddof=1)),
                "oof_risk_difference_vs_lowref_mean": float(sub["risk_difference_vs_lowref"].mean()),
                "oof_risk_difference_vs_lowref_sd": float(sub["risk_difference_vs_lowref"].std(ddof=1)),
                "high_risk_definition": "score >= 4" if rule_id == "PTS_6_core" else "hybrid rule positive",
                "validation_note": "training-fold thresholds and training-fold empirical risk mapping",
            }
        )

    return pd.DataFrame(rows), {"fold_metrics": fold_df}


def write_summary(outpath: Path, table: pd.DataFrame) -> None:
    idx = table.set_index(["scenario", "rule_id"])
    cc_pts = idx.loc[("complete_case", "PTS_6_core")]
    cc_hyb = idx.loc[("complete_case", "HYB_emerging_nonsevere")]
    ipw_pts = idx.loc[("ipw_weighted", "PTS_6_core")]
    ipw_hyb = idx.loc[("ipw_weighted", "HYB_emerging_nonsevere")]
    older60_pts = idx.loc[("older60_complete_case", "PTS_6_core")]
    older60_hyb = idx.loc[("older60_complete_case", "HYB_emerging_nonsevere")]
    older65_pts = idx.loc[("older65_complete_case", "PTS_6_core")]
    older65_hyb = idx.loc[("older65_complete_case", "HYB_emerging_nonsevere")]

    lines = []
    lines.append("# Stage R5 rule stability summary")
    lines.append("")
    lines.append("## 1. Stage aim")
    lines.append("")
    lines.append("本阶段只验证两个最终规则：")
    lines.append("- `PTS_6_core`")
    lines.append("- `HYB_emerging_nonsevere`")
    lines.append("")
    lines.append("重点不是再做新规则开发，而是回答：")
    lines.append("- apparent 表现有多少 optimism")
    lines.append("- calibration 是否明显漂移")
    lines.append("- high-risk 分层是否稳定")
    lines.append("- 在 selection-adjusted 与 older-adult 场景下是否还站得住")
    lines.append("")
    lines.append("## 2. Main complete-case internal validation")
    lines.append("")
    lines.append(f"`PTS_6_core` apparent vs OOF AUC: `{fmt3(cc_pts['apparent_auc'])}` -> `{fmt3(cc_pts['oof_auc'])}` (optimism `{fmt3(cc_pts['optimism_auc'])}`)")
    lines.append(f"`HYB_emerging_nonsevere` apparent vs OOF AUC: `{fmt3(cc_hyb['apparent_auc'])}` -> `{fmt3(cc_hyb['oof_auc'])}` (optimism `{fmt3(cc_hyb['optimism_auc'])}`)")
    lines.append("")
    lines.append(f"`PTS_6_core` OOF calibration slope: `{fmt3(cc_pts['oof_calibration_slope'])}`; high-risk share mean across folds: `{fmt3(cc_pts['oof_high_risk_share_mean'])}`; high-risk event rate mean across folds: `{fmt3(cc_pts['oof_high_risk_event_rate_mean'])}`.")
    lines.append(f"`HYB_emerging_nonsevere` OOF calibration slope: `{fmt3(cc_hyb['oof_calibration_slope'])}`; high-risk share mean across folds: `{fmt3(cc_hyb['oof_high_risk_share_mean'])}`; high-risk event rate mean across folds: `{fmt3(cc_hyb['oof_high_risk_event_rate_mean'])}`.")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- `PTS_6_core` 的 internal validation 更稳，作为简化 prognostic score 比较站得住。")
    lines.append("- `HYB_emerging_nonsevere` 的 discrimination 本来就不是主目标，它更像一个选择性更高的 pre-disability operational flag。")
    lines.append("")
    lines.append("## 3. Missing-data adjusted framework")
    lines.append("")
    lines.append("这里没有做 full MI nested validation。原因是 `Stage 4` 没有把 person-level imputed rule datasets 单独落盘；完整重建会超出本阶段“验证已选规则”的范围。")
    lines.append("因此，本阶段采用的是 **IPW-weighted approximation**：")
    lines.append("- 复用 `Stage 4` 的 sequential truncated selection weights")
    lines.append("- 在 observed complete-case sample 上重新评估两个规则")
    lines.append("")
    lines.append(f"`PTS_6_core` complete-case vs IPW OOF AUC: `{fmt3(cc_pts['oof_auc'])}` -> `{fmt3(ipw_pts['oof_auc'])}`")
    lines.append(f"`HYB_emerging_nonsevere` complete-case vs IPW OOF AUC: `{fmt3(cc_hyb['oof_auc'])}` -> `{fmt3(ipw_hyb['oof_auc'])}`")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- 如果 IPW 下排序和高风险组事件率没有根本逆转，则说明这些规则不是纯 complete-case 偶然产物。")
    lines.append("- 但这一步仍然只是 missing-data adjusted approximation，不应包装成 full MI validation。")
    lines.append("")
    lines.append("## 4. Older-adult stability")
    lines.append("")
    lines.append("当前没有显式 `age_w3` 变量，因此 older-adult 子样本沿用 Stage 10 的近似：`baseline_age_2011 + 4` 作为 `wave 3 age`。")
    lines.append("")
    lines.append(f"`PTS_6_core` OOF AUC in age>=60 / age>=65: `{fmt3(older60_pts['oof_auc'])}` / `{fmt3(older65_pts['oof_auc'])}`")
    lines.append(f"`HYB_emerging_nonsevere` OOF AUC in age>=60 / age>=65: `{fmt3(older60_hyb['oof_auc'])}` / `{fmt3(older65_hyb['oof_auc'])}`")
    lines.append(f"`PTS_6_core` older-adult high-risk event rate mean (age>=60 / age>=65): `{fmt3(older60_pts['oof_high_risk_event_rate_mean'])}` / `{fmt3(older65_pts['oof_high_risk_event_rate_mean'])}`")
    lines.append(f"`HYB_emerging_nonsevere` older-adult high-risk event rate mean (age>=60 / age>=65): `{fmt3(older60_hyb['oof_high_risk_event_rate_mean'])}` / `{fmt3(older65_hyb['oof_high_risk_event_rate_mean'])}`")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- 如果 older-adult 子样本中 high-risk 事件率仍明显抬高，则这些规则更接近真实 geriatric follow-up 场景。")
    lines.append("")
    lines.append("## 5. Bottom line")
    lines.append("")
    lines.append("1. `PTS_6_core` 更像 internally stable simplified prognostic score。")
    lines.append("2. `HYB_emerging_nonsevere` 更像 selective operational flag；其价值主要在于抓出 non-severe but high-risk subgroup，而不是整体 discrimination。")
    lines.append("3. 在当前可复用数据结构下，missing-data 部分最稳的结论只能写到 IPW-weighted stability，不应夸大为 full MI rule validation。")
    lines.append("4. 如果进入下一步整合，最值得带入 `R6` 的两个对象仍然是：")
    lines.append("- `PTS_6_core`")
    lines.append("- `HYB_emerging_nonsevere`")
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_results_index() -> None:
    marker = "## 2026-03-25 Stage R5"
    text = RESULTS_INDEX.read_text(encoding="utf-8")
    if marker in text:
        return
    block = f"""

{marker}

本阶段对两个最终规则做了 honest internal validation，并补做了 IPW-weighted 与 older-adult 稳健性检查。

### 新增文件

| 文件 | 回答的问题 | 备注 |
|---|---|---|
| `results/stageR_rule/r5_internal_validation.csv` | 两个最终规则的 apparent / OOF performance、optimism、calibration、high-risk stability 如何 | Stage R5 主验证表 |
| `results/stageR_rule/r5_rule_stability_summary.md` | 最终规则是否足够稳；missing-data 和 older-adult 框架下能写到什么程度 | Stage R5 结论总结 |

### 本阶段直接结论

1. `PTS_6_core` 是更稳的 simplified prognostic score。
2. `HYB_emerging_nonsevere` 的价值主要在于 operational targeting，而不是整体 discrimination。
3. missing-data 部分最稳的写法是 `IPW-weighted stability`, 不能上升为完整 MI rule validation。
"""
    RESULTS_INDEX.write_text(text.rstrip() + "\n" + block.strip("\n") + "\n", encoding="utf-8")


def write_metadata() -> None:
    meta = {
        "stage": "R5",
        "seed": SEED,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "rules_validated": ["PTS_6_core", "HYB_emerging_nonsevere"],
        "scenarios": [
            "complete_case",
            "ipw_weighted",
            "older60_complete_case",
            "older65_complete_case",
        ],
        "missing_data_note": "IPW-weighted approximation used; no full MI nested validation for rule objects",
    }
    (OUTDIR / "stageR5_run_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    cc = prepare_complete_case_df()
    cc_table, _ = validate_rules(cc, "complete_case", "Unified complete-case sample", weight_col=None)

    ipw = prepare_ipw_df()
    ipw_table, _ = validate_rules(ipw, "ipw_weighted", "Unified sample with Stage 4 truncated IPW", weight_col="selection_weight_truncated")

    older60 = cc[(cc["baseline_age_2011"] + 4) >= 60].copy()
    older60_table, _ = validate_rules(older60, "older60_complete_case", "Age >= 60 at wave 3 approximation", weight_col=None)

    older65 = cc[(cc["baseline_age_2011"] + 4) >= 65].copy()
    older65_table, _ = validate_rules(older65, "older65_complete_case", "Age >= 65 at wave 3 approximation", weight_col=None)

    out = pd.concat([cc_table, ipw_table, older60_table, older65_table], ignore_index=True)
    out.to_csv(OUTDIR / "r5_internal_validation.csv", index=False)

    write_summary(OUTDIR / "r5_rule_stability_summary.md", out)
    update_results_index()
    write_metadata()


if __name__ == "__main__":
    main()

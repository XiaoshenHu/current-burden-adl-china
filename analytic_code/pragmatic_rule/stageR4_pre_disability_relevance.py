from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260325

ROOT = Path("/Users/macbookpro-hu/Library/CloudStorage/OneDrive-个人/Xcode/Medical Research/PROJECTS/charls_life_course_aging")
OUTDIR = ROOT / "results" / "stageR_rule"
RESULTS_INDEX = ROOT / "RESULTS_INDEX.md"

sys.path.insert(0, str(ROOT / "scripts" / "stageR_rule"))
import stageR3_model_rule_comparison as r3  # noqa: E402


def fmt3(x: float | int | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{float(x):.3f}"


def build_masks(df: pd.DataFrame, preds: pd.DataFrame) -> dict[str, pd.Series]:
    score = preds["PTS_6_core_score"]
    masks = {
        "pts_high": score >= 4,
        "pts_lowref": score <= 1,
        "hyb_high": preds["HYB_emerging_nonsevere_score"] >= 1,
        "m2_top20": preds["M2_prob"] >= preds["M2_prob"].quantile(0.80),
        "m2_top10": preds["M2_prob"] >= preds["M2_prob"].quantile(0.90),
        "emerging": df["trajectory_group"].eq("Emerging multidomain decline"),
        "high_burden": df["trajectory_group"].eq("High-burden mixed pattern"),
        "severe": df["trajectory_group"].eq("Severe multidomain decline"),
        "stable": df["trajectory_group"].eq("Stable low-burden"),
    }
    return {k: v.astype(bool) for k, v in masks.items()}


def overlap_row(rule_name: str, rule_mask: pd.Series, comp_name: str, comp_mask: pd.Series) -> dict[str, object]:
    inter = int((rule_mask & comp_mask).sum())
    union = int((rule_mask | comp_mask).sum())
    return {
        "rule_id": rule_name,
        "comparator": comp_name,
        "rule_n": int(rule_mask.sum()),
        "comparator_n": int(comp_mask.sum()),
        "intersection_n": inter,
        "overlap_within_rule": float(inter / rule_mask.sum()) if rule_mask.sum() else np.nan,
        "capture_of_comparator": float(inter / comp_mask.sum()) if comp_mask.sum() else np.nan,
        "jaccard_index": float(inter / union) if union else np.nan,
    }


def group_profile_row(
    df: pd.DataFrame,
    preds: pd.DataFrame,
    mask: pd.Series,
    group_id: str,
    group_label: str,
    lowref_mask: pd.Series,
) -> dict[str, object]:
    sub = df.loc[mask].copy()
    y = sub["incident_adl_w3_w4"]
    lowref_risk = float(df.loc[lowref_mask, "incident_adl_w3_w4"].mean())

    thresholds = json.loads((OUTDIR / "stageR2_run_metadata.json").read_text(encoding="utf-8"))["thresholds"]
    extreme_count = (
        (sub["physical_lim_level_at_w3"] >= thresholds["physical_lim_level_q90"]).astype(int)
        + (sub["dep_level_at_w3"] >= thresholds["dep_level_q90"]).astype(int)
        + (sub["multimorb_level_at_w3"] >= thresholds["multimorb_level_q90"]).astype(int)
    )
    moderate_count = (
        (sub["physical_lim_level_at_w3"] >= thresholds["physical_lim_level_q75"]).astype(int)
        + (sub["dep_level_at_w3"] >= thresholds["dep_level_q75"]).astype(int)
        + (sub["multimorb_level_at_w3"] >= thresholds["multimorb_level_q75"]).astype(int)
    )

    return {
        "group_id": group_id,
        "group_label": group_label,
        "n": int(mask.sum()),
        "share_of_sample": float(mask.mean()),
        "observed_event_rate": float(y.mean()),
        "risk_difference_vs_lowref": float(y.mean() - lowref_risk),
        "mean_m2_predicted_risk": float(preds.loc[mask, "M2_prob"].mean()),
        "mean_dep_level_at_w3": float(sub["dep_level_at_w3"].mean()),
        "mean_cogdef_level_at_w3": float(sub["cogdef_level_at_w3"].mean()),
        "mean_physical_lim_level_at_w3": float(sub["physical_lim_level_at_w3"].mean()),
        "mean_multimorb_level_at_w3": float(sub["multimorb_level_at_w3"].mean()),
        "mean_dep_pre_w3_slope": float(sub["dep_pre_w3_slope"].mean()),
        "mean_physical_lim_pre_w3_slope": float(sub["physical_lim_pre_w3_slope"].mean()),
        "mean_moderate_burden_flag_count": float(moderate_count.mean()),
        "share_with_ge2_extreme_burden_flags": float((extreme_count >= 2).mean()),
        "share_stable": float((sub["trajectory_group"] == "Stable low-burden").mean()),
        "share_emerging": float((sub["trajectory_group"] == "Emerging multidomain decline").mean()),
        "share_high_burden": float((sub["trajectory_group"] == "High-burden mixed pattern").mean()),
        "share_severe": float((sub["trajectory_group"] == "Severe multidomain decline").mean()),
    }


def write_summary(
    outpath: Path,
    overlap: pd.DataFrame,
    profile: pd.DataFrame,
) -> None:
    def row(group_id: str) -> pd.Series:
        return profile.set_index("group_id").loc[group_id]

    pts = row("PTS_6_core_high")
    hyb = row("HYB_emerging_nonsevere_high")
    low = row("Low_reference_score_le1")
    emerging = row("Emerging_phenotype")
    m2q = row("M2_top20")
    m2d = row("M2_top10")

    ov = overlap.set_index(["rule_id", "comparator"])
    lines: list[str] = []
    lines.append("# Stage R4 pre-disability relevance")
    lines.append("")
    lines.append("## 1. Stage aim")
    lines.append("")
    lines.append("本阶段不再问“谁的 AUC 更高”，而是直接问两个候选规则抓到的是谁：")
    lines.append("- `PTS_6_core >= 4`：更广义、可执行的高风险 score")
    lines.append("- `HYB_emerging_nonsevere`：更聚焦于 non-severe but high-risk 的 hybrid rule")
    lines.append("")
    lines.append("所有分析都在 `unified comparison sample (n=8355)` 上完成，因此这些人 **在 wave 3 时均未 ADL disabled**。")
    lines.append("")
    lines.append("## 2. Low-risk reference")
    lines.append("")
    lines.append("为了让不同规则的绝对风险差可以直接比较，本阶段统一使用 `PTS_6_core <= 1` 作为 low-risk reference。")
    lines.append(f"- low-risk reference sample size: `{int(low['n'])}`")
    lines.append(f"- low-risk observed ADL event rate: `{fmt3(low['observed_event_rate'])}`")
    lines.append("")
    lines.append("## 3. Direct answers")
    lines.append("")
    lines.append("### 3.1 Does the score-based rule identify a clearly higher-risk pre-disability group?")
    lines.append("")
    lines.append(f"`PTS_6_core >= 4` selected `{int(pts['n'])}` participants (`{fmt3(pts['share_of_sample'])}` of the unified sample).")
    lines.append(f"Its observed incident ADL risk was `{fmt3(pts['observed_event_rate'])}`, compared with `{fmt3(low['observed_event_rate'])}` in the low-risk reference, for a risk difference of `{fmt3(pts['risk_difference_vs_lowref'])}`.")
    lines.append(f"This high-risk score group remained mixed rather than uniformly severe: `{fmt3(pts['share_emerging'])}` were Emerging, `{fmt3(pts['share_high_burden'])}` were High-burden, and `{fmt3(pts['share_severe'])}` were Severe.")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- 这个 score 确实能从未残疾人群里拉出一个后续风险明显升高的 subgroup。")
    lines.append("- 但它并不是专门抓 `Emerging`，而是在 `Emerging + High-burden + some Severe` 之间形成一个更广义的高风险 operational pool。")
    lines.append("")
    lines.append("### 3.2 Does the hybrid rule better target non-severe but high-risk individuals?")
    lines.append("")
    lines.append(f"`HYB_emerging_nonsevere` selected only `{int(hyb['n'])}` participants (`{fmt3(hyb['share_of_sample'])}` of the sample), but their observed incident ADL risk was `{fmt3(hyb['observed_event_rate'])}`, with a risk difference of `{fmt3(hyb['risk_difference_vs_lowref'])}` versus the same low-risk reference.")
    lines.append(f"This group had lower current physical limitation than the `M2` top 20% group (`{fmt3(hyb['mean_physical_lim_level_at_w3'])}` vs `{fmt3(m2q['mean_physical_lim_level_at_w3'])}`), lower depressive burden than the `M2` top 20% group (`{fmt3(hyb['mean_dep_level_at_w3'])}` vs `{fmt3(m2q['mean_dep_level_at_w3'])}`), and a markedly more negative depressive slope (`{fmt3(hyb['mean_dep_pre_w3_slope'])}` vs `{fmt3(m2q['mean_dep_pre_w3_slope'])}`).")
    lines.append(f"Only `{fmt3(hyb['share_with_ge2_extreme_burden_flags'])}` had two or more extreme current-burden flags, and only `{fmt3(hyb['share_severe'])}` belonged to the Severe phenotype.")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- 这个 hybrid rule 更接近“当前还不是最严重，但历史变化模式已经把风险抬高”的 operational target。")
    lines.append("- 它抓到的是一个更窄、更 selective、更像 pre-disability 的 subgroup。")
    lines.append("")
    lines.append("### 3.3 How do these rules relate to Emerging and to top M2 risk?")
    lines.append("")
    lines.append(f"`PTS_6_core >= 4` overlapped with Emerging at `{fmt3(ov.loc[('PTS_6_core_high', 'Emerging phenotype'), 'overlap_within_rule'])}` within-rule and captured `{fmt3(ov.loc[('PTS_6_core_high', 'Emerging phenotype'), 'capture_of_comparator'])}` of all Emerging participants.")
    lines.append(f"`HYB_emerging_nonsevere` overlapped with Emerging at `{fmt3(ov.loc[('HYB_emerging_nonsevere_high', 'Emerging phenotype'), 'overlap_within_rule'])}` within-rule, but captured only `{fmt3(ov.loc[('HYB_emerging_nonsevere_high', 'Emerging phenotype'), 'capture_of_comparator'])}` of all Emerging participants.")
    lines.append(f"`PTS_6_core >= 4` overlapped with the `M2` top quintile at `{fmt3(ov.loc[('PTS_6_core_high', 'M2 top quintile'), 'overlap_within_rule'])}`, whereas `HYB_emerging_nonsevere` overlapped with the same group at `{fmt3(ov.loc[('HYB_emerging_nonsevere_high', 'M2 top quintile'), 'overlap_within_rule'])}`.")
    lines.append(f"`HYB_emerging_nonsevere` overlapped with the `M2` top decile at `{fmt3(ov.loc[('HYB_emerging_nonsevere_high', 'M2 top decile'), 'overlap_within_rule'])}`.")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- `PTS_6_core` 更像是对 `M2` 高风险分位的简化 operationalization。")
    lines.append("- `HYB_emerging_nonsevere` 不是在复刻全部 Emerging，也不是在复刻全部 M2 top-risk；它更像是从两者交集附近切出一个小而高风险的可执行目标群。")
    lines.append("")
    lines.append("## 4. Is either rule better than hard phenotype for operational identification?")
    lines.append("")
    lines.append(f"The Emerging phenotype alone covered `{int(emerging['n'])}` participants (`{fmt3(emerging['share_of_sample'])}` of the sample) with an observed event rate of `{fmt3(emerging['observed_event_rate'])}`.")
    lines.append(f"Compared with Emerging membership, `PTS_6_core >= 4` was narrower (`{fmt3(pts['share_of_sample'])}` vs `{fmt3(emerging['share_of_sample'])}`) and concentrated a higher event rate (`{fmt3(pts['observed_event_rate'])}` vs `{fmt3(emerging['observed_event_rate'])}`).")
    lines.append(f"`HYB_emerging_nonsevere` was much narrower still (`{fmt3(hyb['share_of_sample'])}`) while keeping an event rate of `{fmt3(hyb['observed_event_rate'])}`.")
    lines.append("")
    lines.append("最稳的判断是：")
    lines.append("- 它们不是在 prediction 上“打败” hard phenotype。")
    lines.append("- 但在 operational identification 上，它们比 hard phenotype 更直接，因为不需要 latent class assignment。")
    lines.append("- `PTS_6_core` 更适合做 broad operational score。")
    lines.append("- `HYB_emerging_nonsevere` 更适合做 narrow pre-disability flag。")
    lines.append("")
    lines.append("## 5. Bottom line")
    lines.append("")
    lines.append("1. 我们确实可以从 `M2` 中提炼出更可操作的高风险识别对象。")
    lines.append("2. `PTS_6_core` 更像 simplified prognostic score。")
    lines.append("3. `HYB_emerging_nonsevere` 更像 practical pre-disability screening flag。")
    lines.append("4. 两者都与 Emerging 只有部分重叠，因此更适合被理解为 **Emerging-like operationalization**，而不是 Emerging 的替代物。")
    lines.append("5. 如果要写进论文，最稳的定位应是：`exploratory operational high-risk identification aid built from the stable M2 signals`。")
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_results_index() -> None:
    marker = "## 2026-03-25 Stage R4"
    text = RESULTS_INDEX.read_text(encoding="utf-8")
    if marker in text:
        return
    block = f"""

{marker}

本阶段不再比较谁的 AUC 更高，而是直接检查候选规则抓到的是谁，以及这些规则是否更接近“pre-disability high-risk identification”。

### 新增文件

| 文件 | 回答的问题 | 备注 |
|---|---|---|
| `results/stageR_rule/r4_rule_vs_emerging_overlap.csv` | 两个候选规则与 Emerging、High-burden、Severe、M2 top risk quantiles 的 overlap/capture 关系是什么 | Stage R4 overlap 主表 |
| `results/stageR_rule/r4_operational_high_risk_profile.csv` | 候选规则抓到的人群当前 burden 如何、future ADL risk 多高、与 low-risk 组相差多少 | Stage R4 profile 主表 |
| `results/stageR_rule/r4_pre_disability_relevance.md` | 候选规则是否真的更接近 pre-disability operational identification | Stage R4 结论总结 |

### 本阶段直接结论

1. `PTS_6_core >= 4` 更像 broad operational score：抓到的人更多，事件率高，但不专属于 Emerging。
2. `HYB_emerging_nonsevere` 更像 narrow pre-disability flag：抓到的人更少，但更符合“当前不是最严重、未来风险已抬高”的目标。
3. 两个规则都只与 Emerging 部分重叠，因此更适合作为 Emerging-like operationalization，而不是对 hard phenotype 的一一替代。
"""
    RESULTS_INDEX.write_text(text.rstrip() + "\n" + block.strip("\n") + "\n", encoding="utf-8")


def write_metadata() -> None:
    meta = {
        "stage": "R4",
        "seed": SEED,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "sample": "unified comparison sample",
        "sample_n": 8355,
        "selected_rules": ["PTS_6_core >= 4", "HYB_emerging_nonsevere"],
        "low_risk_reference": "PTS_6_core <= 1",
    }
    (OUTDIR / "stageR4_run_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    inputs = r3.load_inputs()
    df = inputs["features"].copy()
    thresholds = inputs["r2_meta"]["thresholds"]
    preds = r3.prepare_predictions(df, thresholds)
    masks = build_masks(df, preds)

    overlap_rows = []
    comparators = {
        "Emerging phenotype": masks["emerging"],
        "High-burden phenotype": masks["high_burden"],
        "Severe phenotype": masks["severe"],
        "M2 top quintile": masks["m2_top20"],
        "M2 top decile": masks["m2_top10"],
    }
    rules = {
        "PTS_6_core_high": masks["pts_high"],
        "HYB_emerging_nonsevere_high": masks["hyb_high"],
    }
    for rule_name, rule_mask in rules.items():
        for comp_name, comp_mask in comparators.items():
            overlap_rows.append(overlap_row(rule_name, rule_mask, comp_name, comp_mask))
    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(OUTDIR / "r4_rule_vs_emerging_overlap.csv", index=False)

    lowref = masks["pts_lowref"]
    profile_rows = [
        group_profile_row(df, preds, lowref, "Low_reference_score_le1", "Universal low-risk reference (PTS <= 1)", lowref),
        group_profile_row(df, preds, masks["pts_high"], "PTS_6_core_high", "PTS_6_core >= 4", lowref),
        group_profile_row(df, preds, masks["hyb_high"], "HYB_emerging_nonsevere_high", "HYB_emerging_nonsevere positive", lowref),
        group_profile_row(df, preds, masks["emerging"], "Emerging_phenotype", "Emerging phenotype membership", lowref),
        group_profile_row(df, preds, masks["high_burden"], "High_burden_phenotype", "High-burden phenotype membership", lowref),
        group_profile_row(df, preds, masks["severe"], "Severe_phenotype", "Severe phenotype membership", lowref),
        group_profile_row(df, preds, masks["m2_top20"], "M2_top20", "M2 top quintile risk group", lowref),
        group_profile_row(df, preds, masks["m2_top10"], "M2_top10", "M2 top decile risk group", lowref),
    ]
    profile = pd.DataFrame(profile_rows)
    profile.to_csv(OUTDIR / "r4_operational_high_risk_profile.csv", index=False)

    write_summary(OUTDIR / "r4_pre_disability_relevance.md", overlap, profile)
    update_results_index()
    write_metadata()


if __name__ == "__main__":
    main()

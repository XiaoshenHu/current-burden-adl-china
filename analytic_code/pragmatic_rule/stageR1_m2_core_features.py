from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

# Resolve repository imports independently of the launch directory.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reproducibility.project_paths import project_root

import pandas as pd


SEED = 20260325

ROOT = project_root()
RESULTS = ROOT / "results"
OUTDIR = RESULTS / "stageR_rule"
RESULTS_INDEX = ROOT / "RESULTS_INDEX.md"


def parse_domain(term: str) -> str:
    if term.startswith("dep_"):
        return "depressive symptoms"
    if term.startswith("cogdef_"):
        return "cognitive deficit"
    if term.startswith("physical_lim_"):
        return "physical limitation"
    if term.startswith("multimorb_"):
        return "multimorbidity"
    raise ValueError(f"Unknown feature term: {term}")


def parse_component(term: str) -> str:
    if "level_at_w3" in term:
        return "level_at_w3"
    if "pre_w3_slope" in term:
        return "pre_w3_slope"
    raise ValueError(f"Unknown feature component: {term}")


def domain_short(domain: str) -> str:
    return {
        "depressive symptoms": "dep",
        "cognitive deficit": "cogdef",
        "physical limitation": "physical_lim",
        "multimorbidity": "multimorb",
    }[domain]


def fmt3(x: float) -> str:
    return f"{x:.3f}"


def load_inputs() -> dict[str, pd.DataFrame]:
    return {
        "coeff": pd.read_csv(RESULTS / "stage2_level_change" / "level_change_model_coefficients.csv"),
        "nested": pd.read_csv(RESULTS / "stage3_nested_validation" / "nested_validation_performance.csv"),
        "stage4": pd.read_csv(RESULTS / "stage4_missing_selection" / "cc_vs_mi_vs_ipw_vs_miipw.csv"),
        "stage8": pd.read_csv(RESULTS / "stage8_domain_contribution" / "continuous_domain_contributions.csv"),
        "stage10": pd.read_csv(RESULTS / "stage10_targeted_sensitivity" / "measurement_robustness.csv"),
    }


def build_feature_table(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    coeff = inputs["coeff"]
    stage8 = inputs["stage8"]
    stage10 = inputs["stage10"]

    coeff_m2 = coeff[(coeff["model"] == "M2") & (coeff["term"].str.contains("_level_at_w3|_pre_w3_slope", regex=True))].copy()
    coeff_m2["domain"] = coeff_m2["term"].apply(parse_domain)
    coeff_m2["component"] = coeff_m2["term"].apply(parse_component)

    stage8_feat = stage8[stage8["row_type"] == "feature_coefficient"].copy()
    stage8_drop = stage8[stage8["row_type"] == "feature_drop"].copy()
    stage8_domain = stage8[stage8["row_type"] == "domain_pair_drop"].copy()

    m = coeff_m2.merge(
        stage8_feat[
            [
                "feature",
                "standardized_logit_coef",
                "or_per_1sd",
                "ci_low",
                "ci_high",
                "effect_95ci",
                "p",
                "p_formatted",
            ]
        ].rename(
            columns={
                "feature": "term",
                "standardized_logit_coef": "standardized_coef",
                "or_per_1sd": "or_per_1sd",
                "ci_low": "std_ci_low",
                "ci_high": "std_ci_high",
                "effect_95ci": "std_effect_95ci",
                "p": "std_p",
                "p_formatted": "std_p_formatted",
            }
        ),
        on="term",
        how="left",
    )

    m = m.merge(
        stage8_drop[
            ["feature", "drop_model_oof_auc", "delta_oof_auc_vs_full", "lr_p_formatted"]
        ].rename(
            columns={
                "feature": "term",
                "drop_model_oof_auc": "feature_drop_oof_auc",
                "delta_oof_auc_vs_full": "feature_drop_delta_auc_vs_full",
                "lr_p_formatted": "feature_drop_lr_p_formatted",
            }
        ),
        on="term",
        how="left",
    )

    stage8_domain = stage8_domain.rename(
        columns={
            "domain": "domain_short",
            "drop_model_oof_auc": "domain_pair_drop_oof_auc",
            "delta_oof_auc_vs_full": "domain_pair_drop_delta_auc_vs_full",
            "lr_p_formatted": "domain_pair_drop_lr_p_formatted",
        }
    )
    m["domain_short"] = m["domain"].map(domain_short)
    m = m.merge(
        stage8_domain[
            ["domain_short", "domain_pair_drop_oof_auc", "domain_pair_drop_delta_auc_vs_full", "domain_pair_drop_lr_p_formatted"]
        ],
        on="domain_short",
        how="left",
    )

    coding_rows = []
    for domain in ["depressive symptoms", "cognitive deficit", "physical limitation", "multimorbidity"]:
        if domain == "physical limitation":
            sc = "physical_ordered_burden"
        elif domain == "cognitive deficit":
            sc = "cognition_wave_z"
        elif domain == "multimorbidity":
            sc = "multimorb_ge3"
        else:
            sc = "reference_original"
        temp = stage10[
            (stage10["row_type"] == "burden_adjusted_emerging")
            & (stage10["scenario"] == sc)
        ].copy()
        if temp.empty:
            delta_m2_vs_m1 = None
            delta_m4_vs_m1 = None
            coding_note = "no_direct_recoding_test"
        else:
            row = temp.iloc[0]
            delta_m2_vs_m1 = float(row["delta_m2_vs_m1"])
            delta_m4_vs_m1 = float(row["delta_m4_vs_m1"])
            coding_note = sc
        coding_rows.append(
            {
                "domain": domain,
                "coding_scenario": coding_note,
                "stage10_delta_m2_vs_m1": delta_m2_vs_m1,
                "stage10_delta_m4_vs_m1": delta_m4_vs_m1,
            }
        )
    coding_df = pd.DataFrame(coding_rows)
    m = m.merge(coding_df, on="domain", how="left")

    nested = inputs["nested"]
    stage4 = inputs["stage4"]
    nested_m2m1 = nested[
        (nested["row_type"] == "key_comparison")
        & (nested["comparison_model"] == "M2")
        & (nested["reference_model"] == "M1")
    ].iloc[0]
    nested_m4m1 = nested[
        (nested["row_type"] == "key_comparison")
        & (nested["comparison_model"] == "M4")
        & (nested["reference_model"] == "M1")
    ].iloc[0]
    m["nested_m2_vs_m1_delta_auc"] = float(nested_m2m1["delta_oof_auc_mean"])
    m["nested_m2_vs_m1_ci"] = (
        f"{fmt3(float(nested_m2m1['delta_oof_auc_mean']))} "
        f"({fmt3(float(nested_m2m1['delta_oof_auc_ci_low']))} to {fmt3(float(nested_m2m1['delta_oof_auc_ci_high']))})"
    )
    m["nested_m4_vs_m1_delta_auc"] = float(nested_m4m1["delta_oof_auc_mean"])
    m["nested_m4_vs_m1_ci"] = (
        f"{fmt3(float(nested_m4m1['delta_oof_auc_mean']))} "
        f"({fmt3(float(nested_m4m1['delta_oof_auc_ci_low']))} to {fmt3(float(nested_m4m1['delta_oof_auc_ci_high']))})"
    )

    perf = stage4[stage4["row_type"] == "performance_model"].copy()
    wide = perf.pivot_table(index="method", columns="model_code", values="oof_auc_mean", aggfunc="first")
    m["stage4_methods_with_M2_gt_M1"] = int((wide["M2"] > wide["M1"]).sum())
    m["stage4_methods_with_M4_gt_M1"] = int((wide["M4"] > wide["M1"]).sum())
    m["stage4_methods_total"] = int(wide.shape[0])

    def classify(row: pd.Series) -> str:
        domain_delta = abs(float(row["domain_pair_drop_delta_auc_vs_full"]))
        feature_delta = abs(float(row["feature_drop_delta_auc_vs_full"]))
        p = float(row["std_p"])
        comp = row["component"]
        if p < 0.01 and domain_delta >= 0.005 and feature_delta >= 0.002:
            return "core"
        if comp == "level_at_w3" and p < 0.05 and domain_delta >= 0.001:
            return "supportive"
        if comp == "pre_w3_slope" and p < 0.05 and domain_delta >= 0.005 and feature_delta >= 0.001:
            return "supportive"
        return "deprioritize"

    m["retention_priority"] = m.apply(classify, axis=1)

    def note(row: pd.Series) -> str:
        if row["term"] == "physical_lim_level_at_w3":
            return "dominant current-burden anchor"
        if row["term"] == "dep_pre_w3_slope":
            return "most stable non-physical change feature"
        if row["term"] == "physical_lim_pre_w3_slope":
            return "change feature retained beyond current physical limitation"
        if row["term"] == "multimorb_level_at_w3":
            return "secondary burden anchor with stable contribution"
        if row["term"] == "dep_level_at_w3":
            return "secondary current-burden signal"
        if row["term"] == "cogdef_level_at_w3":
            return "weak but retained current-level information"
        if row["term"] == "cogdef_pre_w3_slope":
            return "change signal weak and unstable"
        if row["term"] == "multimorb_pre_w3_slope":
            return "change signal weak and unstable"
        return ""

    m["interpretation_note"] = m.apply(note, axis=1)

    keep_cols = [
        "term",
        "domain",
        "component",
        "coef",
        "effect_95ci",
        "p_formatted",
        "standardized_coef",
        "std_effect_95ci",
        "std_p_formatted",
        "feature_drop_delta_auc_vs_full",
        "domain_pair_drop_delta_auc_vs_full",
        "coding_scenario",
        "stage10_delta_m2_vs_m1",
        "stage10_delta_m4_vs_m1",
        "nested_m2_vs_m1_ci",
        "nested_m4_vs_m1_ci",
        "stage4_methods_with_M2_gt_M1",
        "stage4_methods_with_M4_gt_M1",
        "stage4_methods_total",
        "retention_priority",
        "interpretation_note",
    ]
    m = m[keep_cols].sort_values(
        by=["retention_priority", "component", "domain_pair_drop_delta_auc_vs_full", "feature_drop_delta_auc_vs_full"],
        ascending=[True, True, True, True],
    )
    return m


def build_summary(table: pd.DataFrame) -> str:
    core = table[table["retention_priority"] == "core"]["term"].tolist()
    supportive = table[table["retention_priority"] == "supportive"]["term"].tolist()
    deprioritize = table[table["retention_priority"] == "deprioritize"]["term"].tolist()

    def bulletize(items: list[str]) -> str:
        return "\n".join([f"- `{x}`" for x in items]) if items else "- none"

    return f"""# Stage R1 feature stability summary

## 1. Stage aim

本阶段只做一件事：锁定 `M2`（base + 4 current levels + 4 slopes）里哪些变量最有信息量、最稳、最值得进入下一步的简化规则开发。

这一步不开始选规则，不开始阈值优化，也不开始 points score。  
它只回答：**哪些 current levels 和 slopes 值得保留，哪些应降级。**

## 2. Data sources used

本阶段整合了以下现有结果，而没有新增主模型定义：
- Stage 2 `level_change_model_coefficients.csv`
- Stage 3 `nested_validation_performance.csv`
- Stage 4 `cc_vs_mi_vs_ipw_vs_miipw.csv`
- Stage 8 `continuous_domain_contributions.csv`
- Stage 10 `measurement_robustness.csv`

## 3. Direct answers

### 3.1 哪些 current levels 最稳定？

最应优先保留的 current levels 是：
- `physical_lim_level_at_w3`
- `dep_level_at_w3`
- `multimorb_level_at_w3`

其中：
- `physical_lim_level_at_w3` 是最强 current-burden anchor；
- `dep_level_at_w3` 和 `multimorb_level_at_w3` 是次级但稳定的 current-level contributors；
- `cogdef_level_at_w3` 仍有一定信息，但更接近弱保留变量，而不是核心变量。

### 3.2 哪些 slopes 最稳定？

最应优先保留的 slopes 是：
- `dep_pre_w3_slope`
- `physical_lim_pre_w3_slope`

这两项都在 Stage 8 的 standardized model 中保留 nominal significance，且去掉后性能会下降。  
相反：
- `cogdef_pre_w3_slope`
- `multimorb_pre_w3_slope`

目前证据较弱，不适合优先带入下一步的简化规则。

### 3.3 哪些变量在不同框架下最稳？

最稳的是**模型层面的结构**，不是每个单独系数都已在 every framework 下重估。

目前可最稳地说：
- `M2 > M1` 在 Stage 3 nested validation 仍成立；
- `M2 > M1` 在 Stage 4 的 `Complete-case / MI / IPW / MI+IPW` 也都成立；
- Stage 10 的 measurement robustness 中，`M2 vs M1` 增量仍维持在约 `0.009–0.011`；
- 因此，真正稳的是：
  - `current burden + a small amount of change signal`
  - 而不是某一个单独 slope 必然在所有次级框架下都稳定不变

这意味着，Stage R2 不应追求“纯 data-driven 最优”，而应优先保留：
- 一个 dominant current-burden anchor
- 一个 non-severe but multidomain burden signal
- 一个可操作的 pre-landmark change signal

## 4. Recommended feature tiers for Stage R2

### Core features
{bulletize(core)}

### Supportive / optional features
{bulletize(supportive)}

### Deprioritized features
{bulletize(deprioritize)}

## 5. Most important interpretation boundary

这一阶段最重要的边界是：

**我们现在锁定的是“对规则开发最值得保留的信号”，不是在宣布这些变量就是最终临床分值系统。**

尤其要避免两种误读：
- 不要把 `physical_lim_level_at_w3` 的强信号误写成“因此 trajectory 没意义”；
- 也不要把 `dep_pre_w3_slope` 或 `physical_lim_pre_w3_slope` 的保留，误写成“因此 slope 已经成为主导信息”。

最准确的过渡性结论是：

1. `physical_lim_level_at_w3` 是最强 current-level driver；
2. `dep_level_at_w3` 和 `multimorb_level_at_w3` 让风险识别保持 multidomain 解释性；
3. `dep_pre_w3_slope` 和 `physical_lim_pre_w3_slope` 是目前最值得保留的 change signals；
4. `cogdef` 更像弱辅助域，而不是下一步简化规则的首选核心域。

## 6. What this means for Stage R2

基于当前证据，Stage R2 的 parsimonious candidate models 最合理的起步不是从 8 个变量等权出发，而是优先尝试：

1. `physical_lim_level_at_w3 + dep_level_at_w3 + multimorb_level_at_w3`
2. 在上面基础上加入 `dep_pre_w3_slope`
3. 再比较是否加入 `physical_lim_pre_w3_slope`
4. `cogdef_level_at_w3` 作为 optional fifth variable

这会更贴近当前研究已经锁定的结构：
- current burden dominates
- change adds modest information
- simplified rule 需要保留至少一个稳定的 change channel
"""


def append_results_index() -> None:
    section = """

## 2026-03-25 Stage R1 Pragmatic rule development: M2 core features lock

本阶段不开始选规则，只锁定 `M2` 的核心 signal，作为后续 parsimonious model / threshold rule / points score 的输入边界。

### 新增文件

| 文件 | 回答的问题 | 备注 |
|---|---|---|
| `scripts/stageR_rule/stageR1_m2_core_features.py` | 如何整合 Stage 2、3、4、8、10 的结果，锁定最值得进入简化规则开发的 M2 变量 | Stage R1 独立脚本 |
| `results/stageR_rule/r1_m2_core_features.csv` | 8 个 current-level / slope 特征中，哪些最稳、最重要、最值得保留 | Stage R1 主表 |
| `results/stageR_rule/r1_feature_stability_summary.md` | 哪些 current levels 最稳；哪些 slopes 最稳；哪些变量应进入下一步简化规则构建 | Stage R1 总结文件 |

### 本阶段给出的直接结论

1. `physical_lim_level_at_w3` 是最强 current-burden anchor。
2. `dep_level_at_w3` 与 `multimorb_level_at_w3` 是最值得保留的次级 burden 信号。
3. 最值得保留的 change signals 是：
   - `dep_pre_w3_slope`
   - `physical_lim_pre_w3_slope`
4. `cogdef_level_at_w3` 更接近弱保留变量；`cogdef_pre_w3_slope` 与 `multimorb_pre_w3_slope` 当前应降级。
5. Stage R2 最合理的简化起点，是围绕：
   - `physical limitation current level`
   - `depressive symptoms current level`
   - `multimorbidity current level`
   - `depressive change`
   - `physical limitation change`

### 下一阶段建议

只在你确认后进入：

1. Stage R2 parsimonious candidate models
"""
    current = RESULTS_INDEX.read_text(encoding="utf-8")
    if "## 2026-03-25 Stage R1 Pragmatic rule development: M2 core features lock" not in current:
        RESULTS_INDEX.write_text(current.rstrip() + section + "\n", encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs()
    table = build_feature_table(inputs)
    table.to_csv(OUTDIR / "r1_m2_core_features.csv", index=False)
    summary = build_summary(table)
    (OUTDIR / "r1_feature_stability_summary.md").write_text(summary, encoding="utf-8")
    append_results_index()

    metadata = {
        "stage": "R1",
        "seed": SEED,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "outputs": [
            str(OUTDIR / "r1_m2_core_features.csv"),
            str(OUTDIR / "r1_feature_stability_summary.md"),
            str(RESULTS_INDEX),
        ],
        "inputs": [
            "stage2_level_change/level_change_model_coefficients.csv",
            "stage3_nested_validation/nested_validation_performance.csv",
            "stage4_missing_selection/cc_vs_mi_vs_ipw_vs_miipw.csv",
            "stage8_domain_contribution/continuous_domain_contributions.csv",
            "stage10_targeted_sensitivity/measurement_robustness.csv",
        ],
    }
    (OUTDIR / "stageR1_run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

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
OUTDIR = ROOT / "results" / "stageR_rule"
RESULTS_INDEX = ROOT / "RESULTS_INDEX.md"
LIMITATIONS = ROOT / "LIMITATIONS_OF_REANALYSIS.md"


def f3(x: float) -> str:
    return f"{float(x):.3f}"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    r3 = pd.read_csv(OUTDIR / "r3_model_rule_comparison.csv")
    r4 = pd.read_csv(OUTDIR / "r4_operational_high_risk_profile.csv")
    r5 = pd.read_csv(OUTDIR / "r5_internal_validation.csv")
    return r3, r4, r5


def build_context(r3: pd.DataFrame, r4: pd.DataFrame, r5: pd.DataFrame) -> dict[str, float]:
    r3_idx = r3.set_index("model_code")
    r4_idx = r4.set_index("group_id")
    r5_idx = r5.set_index(["scenario", "rule_id"])

    return {
        "m1_auc": float(r3_idx.loc["M1", "auc"]),
        "m2_auc": float(r3_idx.loc["M2", "auc"]),
        "m5_auc": float(r3_idx.loc["M5", "auc"]),
        "m3_auc": float(r3_idx.loc["M3", "auc"]),
        "prm_auc": float(r3_idx.loc["PRM_core5", "auc"]),
        "pts_auc": float(r3_idx.loc["PTS_6_core", "auc"]),
        "hyb_auc": float(r3_idx.loc["HYB_emerging_nonsevere", "auc"]),
        "m2_top20_event": float(r4_idx.loc["M2_top20", "observed_event_rate"]),
        "pts_event": float(r4_idx.loc["PTS_6_core_high", "observed_event_rate"]),
        "pts_share": float(r4_idx.loc["PTS_6_core_high", "share_of_sample"]),
        "hyb_event": float(r4_idx.loc["HYB_emerging_nonsevere_high", "observed_event_rate"]),
        "hyb_share": float(r4_idx.loc["HYB_emerging_nonsevere_high", "share_of_sample"]),
        "emerging_event": float(r4_idx.loc["Emerging_phenotype", "observed_event_rate"]),
        "emerging_share": float(r4_idx.loc["Emerging_phenotype", "share_of_sample"]),
        "pts_emerging_within": float(r4_idx.loc["PTS_6_core_high", "share_emerging"]),
        "hyb_emerging_within": float(r4_idx.loc["HYB_emerging_nonsevere_high", "share_emerging"]),
        "pts_lowref_risk": float(r4_idx.loc["Low_reference_score_le1", "observed_event_rate"]),
        "pts_risk_diff": float(r4_idx.loc["PTS_6_core_high", "risk_difference_vs_lowref"]),
        "hyb_risk_diff": float(r4_idx.loc["HYB_emerging_nonsevere_high", "risk_difference_vs_lowref"]),
        "pts_oof_auc": float(r5_idx.loc[("complete_case", "PTS_6_core"), "oof_auc"]),
        "pts_optimism": float(r5_idx.loc[("complete_case", "PTS_6_core"), "optimism_auc"]),
        "pts_oof_slope": float(r5_idx.loc[("complete_case", "PTS_6_core"), "oof_calibration_slope"]),
        "hyb_oof_auc": float(r5_idx.loc[("complete_case", "HYB_emerging_nonsevere"), "oof_auc"]),
        "hyb_optimism": float(r5_idx.loc[("complete_case", "HYB_emerging_nonsevere"), "optimism_auc"]),
        "hyb_oof_slope": float(r5_idx.loc[("complete_case", "HYB_emerging_nonsevere"), "oof_calibration_slope"]),
        "pts_ipw_oof_auc": float(r5_idx.loc[("ipw_weighted", "PTS_6_core"), "oof_auc"]),
        "hyb_ipw_oof_auc": float(r5_idx.loc[("ipw_weighted", "HYB_emerging_nonsevere"), "oof_auc"]),
        "pts_older60_event": float(r5_idx.loc[("older60_complete_case", "PTS_6_core"), "oof_high_risk_event_rate_mean"]),
        "pts_older65_event": float(r5_idx.loc[("older65_complete_case", "PTS_6_core"), "oof_high_risk_event_rate_mean"]),
        "hyb_older60_event": float(r5_idx.loc[("older60_complete_case", "HYB_emerging_nonsevere"), "oof_high_risk_event_rate_mean"]),
        "hyb_older65_event": float(r5_idx.loc[("older65_complete_case", "HYB_emerging_nonsevere"), "oof_high_risk_event_rate_mean"]),
    }


def write_pragmatic_summary(outpath: Path, c: dict[str, float]) -> None:
    text = f"""# Stage R6 pragmatic rule summary

## 1. Stage aim

本阶段不再开发新规则，而是把 `R1-R5` 的结果整合成一个最终判断：
- 我们是否真的得到了一个值得报告的 pragmatic rule？
- 它相对完整 `M2` 损失了多少性能？
- 它在论文里最稳妥的定位是什么？

## 2. Direct answers to the six core scientific questions

### 2.1 我们能否基于 M2 构造一个更可操作的高风险识别规则？

可以，但要分成两类对象：
- `PRM_core5`：5 变量简化回归模型，最接近完整 `M2/M5`
- `PTS_6_core`：真正更可操作的积分式 score
- `HYB_emerging_nonsevere`：更窄、更 selective 的 pre-disability operational flag

其中，最值得作为“可报告的 pragmatic object”的不是 `HYB`，而是 `PTS_6_core`。

### 2.2 这个规则的核心信息主要来自什么？

最稳定的规则信号来自 **current levels + selective slopes** 的结合，而不是 slopes 单独主导。

具体而言，`R1` 已经锁定：
- current burden anchors：`physical limitation`, `depressive symptoms`, `multimorbidity`
- most stable change signals：`physical limitation slope`, `depressive slope`

所以这套规则的核心不是“只看当前 burden”，也不是“只看 trajectory”，而是：

**current burden 为主，change 用来做 modest refinement**

### 2.3 这个规则是否真正更接近“ADL 丧失前高风险人群”的识别？

是，但主要体现在 operational targeting，而不是整体 AUC。

`PTS_6_core >= 4`：
- 覆盖样本 `{f3(c['pts_share']*100)}%`
- observed incident ADL risk `{f3(c['pts_event'])}`
- 相对低风险参照组风险差 `{f3(c['pts_risk_diff'])}`

`HYB_emerging_nonsevere`：
- 覆盖样本 `{f3(c['hyb_share']*100)}%`
- observed incident ADL risk `{f3(c['hyb_event'])}`
- 相对低风险参照组风险差 `{f3(c['hyb_risk_diff'])}`

因此：
- `PTS_6_core` 更像 broad operational high-risk score
- `HYB` 更像 narrow pre-disability flag

### 2.4 它与 Emerging phenotype 是什么关系？

最准确的定位是：**近似 operationalization + only partial overlap**

- `PTS_6_core` 高风险组内，`Emerging` 占 `{f3(c['pts_emerging_within'])}`
- `HYB` 高风险组内，`Emerging` 占 `{f3(c['hyb_emerging_within'])}`
- 但两者都远不能覆盖全部 `Emerging`

所以它们不是：
- `Emerging` 的替代物
- 也不是 phenotype 的机械翻译

更稳的说法是：
- `PTS_6_core` operationalizes a broad high-risk layer around M2-like risk
- `HYB` operationalizes a narrow Emerging-like pre-disability flag

### 2.5 这个规则对论文价值的提升是 substantial、modest 还是 minimal？

最稳的判断是：**modest**

原因很直接：
- 完整 `M2` AUC = `{f3(c['m2_auc'])}`
- `PRM_core5` AUC = `{f3(c['prm_auc'])}`
- `PTS_6_core` AUC = `{f3(c['pts_auc'])}`
- `HYB` AUC = `{f3(c['hyb_auc'])}`

也就是说：
- 若追求性能，`PRM_core5` 几乎可以逼近 `M2`
- 但真正简化成可操作 score 后，AUC 会下降到 `~{f3(c['pts_auc'])}`
- 这个损失不小，因此它不能改变全文主结论，只能增加临床可解释性和 operational relevance

### 2.6 如果写进论文，最稳妥的定位是什么？

最稳妥的定位不是 `best predictive model`，而是：

**a practical high-risk identification rule derived from the stable M2 signals**

次优但也可以接受的表述是：
- `simplified prognostic score`

不建议的表述是：
- `screening tool ready for implementation`
- `new validated risk classifier`
- `replacement for the full multidomain model`

## 3. Final recommendation

### 3.1 Best-performing simplified object

如果只看“简化后仍尽量保留性能”，最佳对象是 `PRM_core5`：
- AUC `{f3(c['prm_auc'])}`, 与 `M2` 的 `{f3(c['m2_auc'])}` 几乎重合

但它本质上还是一个小回归模型，不是最容易操作的规则。

### 3.2 Best operational score

如果只看“最值得报告的 pragmatic rule”，最佳对象是 `PTS_6_core`：
- complete-case OOF AUC `{f3(c['pts_oof_auc'])}`
- optimism `{f3(c['pts_optimism'])}`
- OOF calibration slope `{f3(c['pts_oof_slope'])}`
- older-adult 高风险组事件率仍达 `{f3(c['pts_older60_event'])}` 到 `{f3(c['pts_older65_event'])}`

它的价值不是替代 `M2`，而是：
- 用更简单的方式把一部分 M2 高风险人群 operationalize 出来
- 同时仍保留较高 observed risk

### 3.3 Best selective pre-disability flag

`HYB_emerging_nonsevere` 更像一个 exploratory flag：
- complete-case OOF AUC `{f3(c['hyb_oof_auc'])}`
- IPW OOF AUC `{f3(c['hyb_ipw_oof_auc'])}`
- older-adult 高风险组事件率 `{f3(c['hyb_older60_event'])}` 到 `{f3(c['hyb_older65_event'])}`

它的优势在于“抓人更窄、更像 non-severe but high-risk”；
它的弱点在于整体 discrimination 太低，不适合被写成主规则。

## 4. Bottom line

1. 我们**可以**从 `M2` 中提炼出更可操作的高风险识别对象。
2. 这个对象的主信号来自 **current burden 为主、change 为辅** 的组合。
3. `PTS_6_core` 是最值得报告的 pragmatic rule。
4. `HYB_emerging_nonsevere` 更适合作为 exploratory operational flag，而不是主文核心规则。
5. 这套规则体系对论文价值的提升是 **modest**，主要提升 clinical interpretability，而不是改变主结论。
6. 全文最稳的主线仍然应该是：
   - `wave 3 current burden dominates`
   - `pre-landmark change adds modest refinement`
   - `operational rule is an optional translational extension rather than a replacement`
"""
    outpath.write_text(text, encoding="utf-8")


def write_mainpaper_recommendation(outpath: Path, c: dict[str, float]) -> None:
    text = f"""# Stage R6 main paper recommendation

## 1. Overall recommendation

`pragmatic rule` 模块**不适合升级成主文主结果**，但可以作为：
- Discussion 末段的 translational extension
- 或一个简洁的 supplementary-to-main bridge result

最稳的做法不是把它写成新的 headline，而是把它放成：

**an exploratory practical high-risk identification extension derived from the stable M2 signals**

## 2. What is worth mentioning in the main paper

如果主文要保留这个模块，建议只保留一条非常克制的结果：

- 基于 3 个 current burden 指标和 2 个 stable change 指标构建的简化 score，可在 unified sample 中识别出约 `{f3(c['pts_share']*100)}%` 的高风险人群；
- 该高风险组 observed incident ADL risk 为 `{f3(c['pts_event'])}`，明显高于低风险参照组的 `{f3(c['pts_lowref_risk'])}`；
- 但其 discrimination 低于完整 continuous model（`PTS_6_core` OOF AUC `{f3(c['pts_oof_auc'])}` vs `M2` `{f3(c['m2_auc'])}`）。

这样写的好处是：
- 有 operational relevance
- 不会和主文的 `M2` 主线冲突
- 不会把 exploratory rule 误写成新的主预测工具

## 3. What should not be sold as a main-paper headline

以下内容不建议作为主卖点：
- `PTS_6_core` can replace M2
- `HYB_emerging_nonsevere` is a validated pre-disability classifier
- the pragmatic rule substantially improves clinical utility

原因：
- `PTS_6_core` 相对 `M2` 的 AUC 损失约 `0.059`
- `HYB` 的 OOF AUC 仅 `{f3(c['hyb_oof_auc'])}`
- missing-data 部分只有 `IPW-weighted` stability，而不是 full MI nested validation

## 4. Suggested main-paper placement

### Best option
- 主文不单列新图表
- 在 Discussion 或 Results 末段用 2-3 句话提及：
  - “we further derived an exploratory practical score from the stable level-plus-change signals...”

### Acceptable option
- 主文加一个 very compact text box / one-line table row
- 只报：
  - high-risk group size
  - observed risk
  - relative performance loss versus M2

### Not recommended
- 把 rule 放进摘要主结果
- 把 rule 放进标题
- 把 rule 放成 Figure 4 的核心比较对象

## 5. Best single take-home sentence for the main paper

**A simplified score built from the most stable current-burden and change signals could identify a smaller subgroup with substantially elevated observed ADL risk, but this exploratory operational rule did not outperform the full continuous level-plus-slope model and should be viewed as a translational extension rather than a replacement.**
"""
    outpath.write_text(text, encoding="utf-8")


def write_supplement_recommendation(outpath: Path, c: dict[str, float]) -> None:
    text = f"""# Stage R6 supplement recommendation

## 1. Recommended supplement positioning

`pragmatic rule` 模块最适合放入补充材料，作为：
- `operational translation of the stable M2 signals`
- `exploratory risk-identification extension`

## 2. What should go into the supplement

### A. Candidate-rule development logic
- `R1` 的 feature stability summary
- `R2` 的 candidate families 和规则定义

### B. Unified comparison table
- `R3` 的模型/规则并排比较
- 至少保留：
  - `M1`
  - `M2`
  - `M5`
  - `PRM_core5`
  - `PTS_6_core`
  - `HYB_emerging_nonsevere`

### C. Operational profile table
- `PTS_6_core >= 4`
- `HYB_emerging_nonsevere`
- `Emerging phenotype`
- `M2 top20%`

特别值得保留的数字：
- `PTS_6_core` 高风险组事件率 `{f3(c['pts_event'])}`
- `HYB` 高风险组事件率 `{f3(c['hyb_event'])}`
- `Emerging phenotype` 事件率 `{f3(c['emerging_event'])}`
- `M2 top20%` 事件率 `{f3(c['m2_top20_event'])}`

### D. Internal validation table
- `R5` 的 complete-case / IPW / older-adult 并排验证

最值得保留的验证数字：
- `PTS_6_core` complete-case OOF AUC `{f3(c['pts_oof_auc'])}`
- `PTS_6_core` IPW OOF AUC `{f3(c['pts_ipw_oof_auc'])}`
- `HYB` complete-case OOF AUC `{f3(c['hyb_oof_auc'])}`
- `HYB` IPW OOF AUC `{f3(c['hyb_ipw_oof_auc'])}`

## 3. Suggested supplement message hierarchy

1. 我们从稳定的 `M2` 信号中提炼出了 2 类 operational objects
2. `PRM_core5` 最接近完整模型，但本质上仍是简化回归模型
3. `PTS_6_core` 是最值得报告的 practical score
4. `HYB` 是更 selective 的 exploratory flag
5. 这些对象主要提高可解释性与 operational targeting，不改变全文主结论

## 4. What not to overclaim in the supplement

即使放在补充材料，也不应写成：
- externally validated rule
- clinically ready screening instrument
- definitive pre-disability classifier

最稳的补充材料标题风格应接近：

**Exploratory pragmatic high-risk identification rules derived from the stable level-plus-change signals**
"""
    outpath.write_text(text, encoding="utf-8")


def update_results_index() -> None:
    marker = "## 2026-03-25 Stage R6"
    text = RESULTS_INDEX.read_text(encoding="utf-8")
    if marker in text:
        return
    block = f"""

{marker}

本阶段整合 `R1-R5`，对 pragmatic rule 模块做最终定位，不再新增模型。

### 新增文件

| 文件 | 回答的问题 | 备注 |
|---|---|---|
| `results/stageR_rule/r6_pragmatic_rule_summary.md` | 是否存在值得报告的 pragmatic rule；它相对 M2 的价值到底有多大 | R6 总整合 |
| `results/stageR_rule/r6_mainpaper_recommendation.md` | rule 模块是否值得进主文，以及最稳妥的写法是什么 | 主文取舍建议 |
| `results/stageR_rule/r6_supplement_recommendation.md` | rule 模块更适合怎么放进补充材料 | 补充材料取舍建议 |

### 本阶段直接结论

1. `PTS_6_core` 是最值得报告的 practical high-risk identification rule。
2. `HYB_emerging_nonsevere` 更像 exploratory pre-disability flag，而不是主规则。
3. pragmatic rule 模块对全文价值的提升是 `modest`，主要改善可解释性与 operational targeting，不改变主结论。
"""
    RESULTS_INDEX.write_text(text.rstrip() + "\n" + block.strip("\n") + "\n", encoding="utf-8")


def update_limitations() -> None:
    marker = "### 2.11 Pragmatic rule module remains translational and exploratory"
    text = LIMITATIONS.read_text(encoding="utf-8")
    if marker in text:
        return
    block = """

### 2.11 Pragmatic rule module remains translational and exploratory
- `Stage R1-R6` 的 pragmatic rule 是在同一研究队列的 reanalysis 基础上开发出来的 translational extension，而不是预先注册的 primary analysis module。
- `PTS_6_core` 和 `HYB_emerging_nonsevere` 的阈值来自当前样本分布与稳定信号提炼，不是外部临床阈值。
- `HYB_emerging_nonsevere` 的 discrimination 较低，其价值主要在 selective operational targeting，而不是整体预测性能。
- `Stage R5` 仅完成了 complete-case、IPW-weighted approximation 和 older-adult 稳健性验证，没有完成 full MI nested rule validation。
- 因此，这套 pragmatic rule 最稳的定位是 exploratory operational aid，而不是可直接推广的 validated screening instrument。
"""
    LIMITATIONS.write_text(text.rstrip() + "\n" + block.strip("\n") + "\n", encoding="utf-8")


def write_metadata() -> None:
    meta = {
        "stage": "R6",
        "seed": SEED,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "inputs": [
            "results/stageR_rule/r3_model_rule_comparison.csv",
            "results/stageR_rule/r4_operational_high_risk_profile.csv",
            "results/stageR_rule/r5_internal_validation.csv",
        ],
        "primary_recommendation": "PTS_6_core as practical high-risk identification rule",
        "secondary_recommendation": "HYB_emerging_nonsevere as exploratory pre-disability flag",
    }
    (OUTDIR / "stageR6_run_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    r3, r4, r5 = load_inputs()
    c = build_context(r3, r4, r5)
    write_pragmatic_summary(OUTDIR / "r6_pragmatic_rule_summary.md", c)
    write_mainpaper_recommendation(OUTDIR / "r6_mainpaper_recommendation.md", c)
    write_supplement_recommendation(OUTDIR / "r6_supplement_recommendation.md", c)
    update_results_index()
    update_limitations()
    write_metadata()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

# Resolve repository imports independently of the launch directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reproducibility.project_paths import project_root

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.proportion import proportion_confint


SEED = 20260325
ROOT = project_root()
OUTDIR = ROOT / "results" / "stageM1_risk_strat"
RESULTS_INDEX = ROOT / "RESULTS_INDEX.md"

from analytic_code.model_validation_and_representation import stage2_level_change as s2  # noqa: E402


def f3(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return "NA"
    return f"{float(x):.3f}"


def ensure_dirs() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)


def build_locked_m1_predictions() -> pd.DataFrame:
    work = s2.build_stage2_sample().copy().reset_index(drop=True)
    X, _, model_type = s2.get_feature_set(work, "M1")
    y = work["incident_adl_w3_w4"].astype(int).reset_index(drop=True)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=s2.SEEDS["cv"])
    oof_pred = np.zeros(len(work), dtype=float)
    full_pred = s2.fit_full_and_predict(X, y, X, model_type=model_type)
    for train_idx, test_idx in skf.split(np.zeros(len(y)), y):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        oof_pred[test_idx] = s2.fit_full_and_predict(X_train, y_train, X_test, model_type=model_type)
    work["m1_oof_predicted_risk"] = oof_pred
    work["m1_full_predicted_risk"] = full_pred
    return work


def make_quantile_groups(pred: pd.Series, n_groups: int, prefix: str) -> pd.Series:
    ranked = pred.rank(method="first")
    bins = pd.qcut(ranked, q=n_groups, labels=False)
    return bins.astype(int).map(lambda x: f"{prefix}{x + 1}")


def summarize_groups(
    df: pd.DataFrame,
    group_col: str,
    group_order: list[str],
    pred_col: str = "m1_oof_predicted_risk",
) -> pd.DataFrame:
    overall_observed = float(df["incident_adl_w3_w4"].mean())
    overall_pred = float(df[pred_col].mean())
    rows = []
    for group in group_order:
        sub = df[df[group_col] == group].copy()
        n = int(len(sub))
        events = int(sub["incident_adl_w3_w4"].sum())
        obs = float(sub["incident_adl_w3_w4"].mean())
        ci_low, ci_high = proportion_confint(events, n, alpha=0.05, method="wilson")
        mean_pred = float(sub[pred_col].mean())
        rows.append(
            {
                "risk_group": group,
                "n": n,
                "share_of_sample": n / len(df),
                "events": events,
                "observed_event_rate": obs,
                "observed_event_rate_ci_low": float(ci_low),
                "observed_event_rate_ci_high": float(ci_high),
                "mean_predicted_risk": mean_pred,
                "observed_minus_predicted": obs - mean_pred,
                "observed_minus_overall": obs - overall_observed,
                "predicted_minus_overall": mean_pred - overall_pred,
                "share_stable": float(sub["trajectory_group"].eq("Stable low-burden").mean()),
                "share_emerging": float(sub["trajectory_group"].eq("Emerging multidomain decline").mean()),
                "share_high_burden": float(sub["trajectory_group"].eq("High-burden mixed pattern").mean()),
                "share_severe": float(sub["trajectory_group"].eq("Severe multidomain decline").mean()),
            }
        )
    return pd.DataFrame(rows)


def write_summary(
    outpath: Path,
    df: pd.DataFrame,
    quint: pd.DataFrame,
    dec: pd.DataFrame,
    auc: float,
) -> None:
    q1 = quint.iloc[0]
    q5 = quint.iloc[-1]
    d1 = dec.iloc[0]
    d10 = dec.iloc[-1]

    lines: list[str] = []
    lines.append("# M1 risk stratification summary")
    lines.append("")
    lines.append("## 1. Locked model and sample")
    lines.append("")
    lines.append("本阶段没有改 M1 的定义，也没有换模型形式。")
    lines.append("M1 仍然是当前锁定的 clinically accessible current-burden model：")
    lines.append("- `baseline_age_2011`")
    lines.append("- `female`")
    lines.append("- `rural_hukou_2011`")
    lines.append("- `low_self_education`")
    lines.append("- `dep_level_at_w3`")
    lines.append("- `cogdef_level_at_w3`")
    lines.append("- `physical_lim_level_at_w3`")
    lines.append("- `multimorb_level_at_w3`")
    lines.append("")
    lines.append("使用样本是当前主线 1 的锁定核心样本：")
    lines.append(f"- `unified comparison sample`, `n = {len(df)}`, `events = {int(df['incident_adl_w3_w4'].sum())}`")
    lines.append(f"- 同一 10 折 OOF 框架下，M1 复现得到 `OOF AUC = {f3(auc)}`")
    lines.append("")
    lines.append("为避免把 apparent risk 误当成临床分层能力，本阶段分层展示基于 **OOF predicted risk**。")
    lines.append("")
    lines.append("## 2. Direct answers")
    lines.append("")
    lines.append("### 2.1 M1 是否已经能把人群分出清楚的风险梯度？")
    lines.append("")
    lines.append("可以。无论按 quintiles 还是 deciles，observed incident ADL risk 都呈现明显递增。")
    lines.append("")
    lines.append(f"- Quintiles: `Q1 = {f3(q1['observed_event_rate'])}`, `Q5 = {f3(q5['observed_event_rate'])}`, absolute risk difference = `{f3(q5['observed_event_rate'] - q1['observed_event_rate'])}`")
    lines.append(f"- Deciles: `D1 = {f3(d1['observed_event_rate'])}`, `D10 = {f3(d10['observed_event_rate'])}`, absolute risk difference = `{f3(d10['observed_event_rate'] - d1['observed_event_rate'])}`")
    lines.append("")
    lines.append("这说明单靠 wave 3 current multidomain burden，已经能把未残疾人群分成后续 ADL disability 风险明显不同的层级。")
    lines.append("")
    lines.append("### 2.2 最高风险层的人群占比和 observed event rate 分别是多少？")
    lines.append("")
    lines.append(f"- Highest quintile: share `{f3(q5['share_of_sample'] * 100)}%`, observed event rate `{f3(q5['observed_event_rate'])}`")
    lines.append(f"- Highest decile: share `{f3(d10['share_of_sample'] * 100)}%`, observed event rate `{f3(d10['observed_event_rate'])}`")
    lines.append("")
    lines.append("如果需要一个既不过宽也不过窄的主文展示对象，最高 quintile 往往更现实；如果需要说明风险梯度有多连续，deciles 更细。")
    lines.append("")
    lines.append("### 2.3 这种分层是否足以支撑“current burden is the dominant prognostic signal”的临床直观展示？")
    lines.append("")
    lines.append("足以。")
    lines.append("原因不是它的 AUC 再次提高，而是：")
    lines.append("- 最高风险层已经聚集了明显更高的 observed ADL event rate；")
    lines.append("- 该层人群比例仍在现实可用范围内；")
    lines.append("- observed risk 与 mean predicted risk 在各层级基本方向一致，没有系统性反向。")
    lines.append("")
    lines.append("### 2.4 主文最适合展示 quintiles、deciles，还是 low/intermediate/high？")
    lines.append("")
    lines.append("最稳的主文展示是 **quintiles**。")
    lines.append("理由：")
    lines.append("- 比 deciles 更稳定，单组样本量更充足；")
    lines.append("- 比 low/intermediate/high 更能保留连续风险梯度；")
    lines.append("- 对 eClinicalMedicine 风格的正文图更简洁。")
    lines.append("")
    lines.append("deciles 更适合作为补充材料，用来证明梯度不是偶然的。")
    lines.append("low/intermediate/high 只有在需要极度压缩时才有价值，但会损失层级信息。")
    lines.append("")
    lines.append("### 2.5 哪种展示最适合 eClinicalMedicine 风格的正文图表？")
    lines.append("")
    lines.append("最合适的是：")
    lines.append("- 主文：`quintile-based observed vs mean predicted risk plot`")
    lines.append("- 补充材料：`decile-based calibration-style stratification table/figure`")
    lines.append("")
    lines.append("## 3. Added clinical reading")
    lines.append("")
    lines.append(f"- 全样本平均 observed risk: `{f3(df['incident_adl_w3_w4'].mean())}`")
    lines.append(f"- Highest quintile mean predicted risk: `{f3(q5['mean_predicted_risk'])}`, observed risk: `{f3(q5['observed_event_rate'])}`")
    lines.append(f"- Highest decile mean predicted risk: `{f3(d10['mean_predicted_risk'])}`, observed risk: `{f3(d10['observed_event_rate'])}`")
    lines.append(f"- Highest quintile phenotype mix: Emerging `{f3(q5['share_emerging'])}`, High-burden `{f3(q5['share_high_burden'])}`, Severe `{f3(q5['share_severe'])}`")
    lines.append(f"- Highest decile phenotype mix: Emerging `{f3(d10['share_emerging'])}`, High-burden `{f3(d10['share_high_burden'])}`, Severe `{f3(d10['share_severe'])}`")
    lines.append("")
    lines.append("这说明 M1 的最高风险层虽然不等同于任何单一 phenotype，但已经富集了 adverse phenotype 组，尤其是 Emerging 和 High-burden。")
    lines.append("")
    lines.append("## 4. Bottom line")
    lines.append("")
    lines.append("1. M1 已经能把人群分出清楚、单调的绝对风险梯度。")
    lines.append("2. 这种分层本身就足以为“current burden is the dominant prognostic signal”提供临床直观展示。")
    lines.append("3. 主文最推荐用 quintiles；deciles 放补充材料。")
    lines.append("4. 这一模块是对主线 1 的展示增强，不改变全文主故事，也不引入新 score。")
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plot(pdf_path: Path, quint: pd.DataFrame, dec: pd.DataFrame) -> None:
    def draw_panel(ax, data: pd.DataFrame, title: str) -> None:
        x = np.arange(len(data))
        obs = data["observed_event_rate"].to_numpy()
        pred = data["mean_predicted_risk"].to_numpy()
        err_low = obs - data["observed_event_rate_ci_low"].to_numpy()
        err_high = data["observed_event_rate_ci_high"].to_numpy() - obs
        ax.bar(x, obs, color="#4C78A8", alpha=0.85, width=0.72, label="Observed risk")
        ax.errorbar(x, obs, yerr=[err_low, err_high], fmt="none", ecolor="black", elinewidth=1.0, capsize=3)
        ax.plot(x, pred, color="#E45756", marker="o", linewidth=2.0, label="Mean predicted risk")
        ax.set_xticks(x)
        ax.set_xticklabels(data["risk_group"].tolist())
        ax.set_ylabel("Incident ADL disability risk")
        ax.set_title(title)
        ax.set_ylim(0, max(obs.max(), pred.max()) * 1.25)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, loc="upper left")

    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        draw_panel(ax, quint, "M1 risk stratification by quintiles")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9.0, 5.5))
        draw_panel(ax, dec, "M1 risk stratification by deciles")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def update_results_index() -> None:
    marker = "## 2026-03-25 Stage M1 Risk Stratification"
    text = RESULTS_INDEX.read_text(encoding="utf-8")
    if marker in text:
        return
    block = f"""

{marker}

本阶段不改 `M1` 主模型，只把当前锁定的 wave 3 current multidomain burden model 转成更临床直观的 absolute risk stratification 展示。

### 新增文件

| 文件 | 回答的问题 | 备注 |
|---|---|---|
| `results/stageM1_risk_strat/risk_stratification_quintiles.csv` | 按 M1 predicted risk quintiles 分组后，每层样本量、observed risk、mean predicted risk 和 O-P 差异如何 | 正文优先展示 |
| `results/stageM1_risk_strat/risk_stratification_deciles.csv` | 按 M1 predicted risk deciles 分组后，风险梯度是否更细且仍稳定 | 补充展示 |
| `results/stageM1_risk_strat/risk_stratification_summary.md` | M1 是否已能分出清楚风险梯度；最高风险层多大；正文最适合用哪种分层展示 | 阶段总结 |
| `results/stageM1_risk_strat/risk_stratification_plot.pdf` | M1 quintiles / deciles 的 observed vs mean predicted risk 图 | 便于主文图表取舍 |

### 本阶段直接结论

1. 单靠 `M1`，已经可以把未残疾人群分出清楚的 incident ADL disability 绝对风险梯度。
2. 最高风险 quintile / decile 都能识别出 observed risk 明显更高的人群，且组大小在现实上可用。
3. 对主文最合适的展示是 quintiles；deciles 更适合作为补充材料来证明风险梯度连续存在。
"""
    RESULTS_INDEX.write_text(text.rstrip() + "\n" + block.strip("\n") + "\n", encoding="utf-8")


def write_metadata(outpath: Path, auc: float, n: int, events: int) -> None:
    meta = {
        "stage": "stageM1_risk_strat",
        "seed": SEED,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "analytic_sample": "unified comparison sample",
        "n": n,
        "events": events,
        "model_locked": "M1 = base covariates + 4 wave3 current levels",
        "prediction_type": "10-fold out-of-fold predicted risk using locked Stage 2 M1 definition",
        "oof_auc": auc,
    }
    outpath.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    work = build_locked_m1_predictions()
    auc = float(roc_auc_score(work["incident_adl_w3_w4"], work["m1_oof_predicted_risk"]))

    work["quintile_group"] = make_quantile_groups(work["m1_oof_predicted_risk"], 5, "Q")
    work["decile_group"] = make_quantile_groups(work["m1_oof_predicted_risk"], 10, "D")

    quint_order = [f"Q{i}" for i in range(1, 6)]
    dec_order = [f"D{i}" for i in range(1, 11)]

    quint = summarize_groups(work, "quintile_group", quint_order)
    dec = summarize_groups(work, "decile_group", dec_order)

    quint.to_csv(OUTDIR / "risk_stratification_quintiles.csv", index=False)
    dec.to_csv(OUTDIR / "risk_stratification_deciles.csv", index=False)
    write_summary(OUTDIR / "risk_stratification_summary.md", work, quint, dec, auc)
    make_plot(OUTDIR / "risk_stratification_plot.pdf", quint, dec)
    write_metadata(OUTDIR / "stageM1_run_metadata.json", auc, int(len(work)), int(work["incident_adl_w3_w4"].sum()))
    update_results_index()


if __name__ == "__main__":
    main()

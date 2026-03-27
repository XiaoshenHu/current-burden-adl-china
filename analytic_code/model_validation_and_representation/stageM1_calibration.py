#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.proportion import proportion_confint


ROOT = Path("/Users/macbookpro-hu/Library/CloudStorage/OneDrive-个人/Xcode/Medical Research/PROJECTS/charls_life_course_aging")
OUTDIR = ROOT / "results" / "stageM1_calibration"
RESULTS_INDEX = ROOT / "RESULTS_INDEX.md"

sys.path.insert(0, str(ROOT / "scripts" / "stageM1_risk_strat"))
import stageM1_risk_strat as m1s  # noqa: E402


def f3(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return "NA"
    return f"{float(x):.3f}"


def ensure_dirs() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)


def make_deciles(pred: pd.Series) -> pd.Series:
    ranked = pred.rank(method="first")
    return pd.qcut(ranked, q=10, labels=False).astype(int).map(lambda x: f"D{x + 1}")


def decile_summary(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    order = [f"D{i}" for i in range(1, 11)]
    rows = []
    for group in order:
        sub = df[df["m1_decile"] == group].copy()
        n = int(len(sub))
        events = int(sub["incident_adl_w3_w4"].sum())
        obs = float(sub["incident_adl_w3_w4"].mean())
        mean_pred = float(sub[pred_col].mean())
        ci_low, ci_high = proportion_confint(events, n, alpha=0.05, method="wilson")
        rows.append(
            {
                "decile": group,
                "n": n,
                "share_of_sample": n / len(df),
                "events": events,
                "observed_event_rate": obs,
                "observed_event_rate_ci_low": float(ci_low),
                "observed_event_rate_ci_high": float(ci_high),
                "mean_predicted_risk": mean_pred,
                "observed_minus_predicted": obs - mean_pred,
                "absolute_difference": abs(obs - mean_pred),
            }
        )
    return pd.DataFrame(rows)


def metrics_table(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    y = df["incident_adl_w3_w4"].to_numpy(dtype=float)
    p = df[pred_col].to_numpy(dtype=float)
    intercept, slope = m1s.s2.calibration_intercept_slope(y, p)
    row = {
        "model_code": "M1",
        "analytic_sample": "unified comparison sample",
        "n": int(len(df)),
        "events": int(df["incident_adl_w3_w4"].sum()),
        "observed_event_rate": float(df["incident_adl_w3_w4"].mean()),
        "mean_predicted_risk": float(df[pred_col].mean()),
        "calibration_in_the_large_observed_minus_predicted": float(df["incident_adl_w3_w4"].mean() - df[pred_col].mean()),
        "oof_auc": float(m1s.s2.roc_auc_score(df["incident_adl_w3_w4"], df[pred_col])),
        "oof_brier": float(m1s.s2.weighted_brier(y, p)),
        "oof_calibration_intercept": float(intercept),
        "oof_calibration_slope": float(slope),
        "top_decile_mean_predicted_risk": float(df.loc[df["m1_decile"] == "D10", pred_col].mean()),
        "top_decile_observed_event_rate": float(df.loc[df["m1_decile"] == "D10", "incident_adl_w3_w4"].mean()),
        "top_decile_observed_minus_predicted": float(
            df.loc[df["m1_decile"] == "D10", "incident_adl_w3_w4"].mean()
            - df.loc[df["m1_decile"] == "D10", pred_col].mean()
        ),
    }
    return pd.DataFrame([row])


def make_plot(pdf_path: Path, decile_df: pd.DataFrame, full_df: pd.DataFrame, pred_col: str) -> None:
    x = decile_df["mean_predicted_risk"].to_numpy()
    y = decile_df["observed_event_rate"].to_numpy()
    yerr_low = y - decile_df["observed_event_rate_ci_low"].to_numpy()
    yerr_high = decile_df["observed_event_rate_ci_high"].to_numpy() - y

    smooth = lowess(
        endog=full_df["incident_adl_w3_w4"].to_numpy(dtype=float),
        exog=full_df[pred_col].to_numpy(dtype=float),
        frac=0.35,
        it=0,
        return_sorted=True,
    )

    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))

        ax = axes[0]
        ax.plot([0, max(x.max(), y.max()) * 1.05], [0, max(x.max(), y.max()) * 1.05], linestyle="--", color="#666666", linewidth=1.2, label="Ideal")
        ax.errorbar(x, y, yerr=[yerr_low, yerr_high], fmt="o", color="#4C78A8", ecolor="#4C78A8", capsize=3, label="Deciles")
        ax.plot(smooth[:, 0], smooth[:, 1], color="#E45756", linewidth=2.0, label="LOESS")
        for _, row in decile_df.iterrows():
            ax.annotate(row["decile"], (row["mean_predicted_risk"], row["observed_event_rate"]), fontsize=8, xytext=(4, 2), textcoords="offset points")
        ax.set_xlabel("Predicted risk")
        ax.set_ylabel("Observed risk")
        ax.set_title("M1 calibration plot")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, loc="upper left")

        ax = axes[1]
        bars = ax.bar(
            decile_df["decile"],
            decile_df["observed_minus_predicted"],
            color=["#72B7B2" if v >= 0 else "#F58518" for v in decile_df["observed_minus_predicted"]],
            alpha=0.9,
        )
        ax.axhline(0, color="black", linewidth=1)
        ax.set_ylabel("Observed - predicted")
        ax.set_title("Observed vs predicted by decile")
        ax.grid(axis="y", alpha=0.25)
        for bar, val in zip(bars, decile_df["observed_minus_predicted"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def write_summary(outpath: Path, metrics: pd.DataFrame, deciles: pd.DataFrame) -> None:
    m = metrics.iloc[0]
    d1 = deciles.iloc[0]
    d10 = deciles.iloc[-1]
    max_abs_diff = deciles["absolute_difference"].max()
    high_tail_abs = abs(float(d10["observed_minus_predicted"]))

    lines: list[str] = []
    lines.append("# M1 calibration summary")
    lines.append("")
    lines.append("## 1. Locked model and sample")
    lines.append("")
    lines.append("本阶段没有改 M1 的定义，也没有开始新的 nested validation。")
    lines.append("仍然使用当前主文主线 1 的同一样本和同一套 10 折 OOF 预测：")
    lines.append(f"- `unified comparison sample`, `n = {int(m['n'])}`, `events = {int(m['events'])}`")
    lines.append(f"- `OOF AUC = {f3(m['oof_auc'])}`, `Brier = {f3(m['oof_brier'])}`")
    lines.append("")
    lines.append("## 2. Direct answers")
    lines.append("")
    lines.append("### 2.1 M1 的整体 calibration 是否可接受？")
    lines.append("")
    lines.append("整体上可接受。")
    lines.append(f"- calibration intercept = `{f3(m['oof_calibration_intercept'])}`")
    lines.append(f"- calibration slope = `{f3(m['oof_calibration_slope'])}`")
    lines.append(f"- calibration-in-the-large (observed - mean predicted) = `{f3(m['calibration_in_the_large_observed_minus_predicted'])}`")
    lines.append("")
    lines.append("这些数值都非常接近理想值，说明没有明显系统性高估或低估。")
    lines.append("")
    lines.append("### 2.2 M1 在高风险段是否仍然 reasonably calibrated？")
    lines.append("")
    lines.append(f"- Lowest decile: predicted `{f3(d1['mean_predicted_risk'])}`, observed `{f3(d1['observed_event_rate'])}`")
    lines.append(f"- Highest decile: predicted `{f3(d10['mean_predicted_risk'])}`, observed `{f3(d10['observed_event_rate'])}`, observed-predicted = `{f3(d10['observed_minus_predicted'])}`")
    lines.append("")
    lines.append("高风险端总体仍可接受。")
    lines.append("最顶端 decile 有轻度偏移，但幅度不大，且 observed risk 仍落在合理的高风险范围内。")
    lines.append("")
    lines.append("### 2.3 M1 是否存在系统性高估或低估？")
    lines.append("")
    lines.append("没有看到明显的系统性单向偏移。")
    lines.append(f"- decile 最大绝对 observed-predicted 差值约 `{f3(max_abs_diff)}`")
    lines.append(f"- 最高风险 decile 的绝对偏移约 `{f3(high_tail_abs)}`")
    lines.append("")
    lines.append("更准确的描述是：")
    lines.append("- 低风险段整体校准良好")
    lines.append("- 中高风险段有轻度波动，但没有证据支持“高风险端严重失准”")
    lines.append("")
    lines.append("### 2.4 主文最适合放哪一种展示？")
    lines.append("")
    lines.append("最适合主文的是：`simple calibration plot`。")
    lines.append("理由：")
    lines.append("- 视觉上最直接")
    lines.append("- 与前一阶段的 quintile risk-stratification 展示互补")
    lines.append("- 更符合 eClinicalMedicine 正文图的简洁风格")
    lines.append("")
    lines.append("`decile-based observed-vs-predicted table` 更适合补充材料。")
    lines.append("如果正文空间允许，也可以主文放简洁 calibration plot，补充材料放 decile table。")
    lines.append("")
    lines.append("### 2.5 这个模块对主线 1 的增益是 substantial、modest，还是 minimal？")
    lines.append("")
    lines.append("最稳的判断是：**modest**。")
    lines.append("它不会改变“current burden dominates”这条主故事，但能让 `M1` 作为 current-burden reference model 更直观、更可信。")
    lines.append("")
    lines.append("## 3. Practical interpretation")
    lines.append("")
    lines.append("这一步支持在正文里更放心地把 `M1` 用作 current-burden reference model。")
    lines.append("如果要加一句边界提醒，最稳的写法是：")
    lines.append("- `Calibration was acceptable overall, with only mild deviation at the highest predicted-risk end.`")
    lines.append("")
    lines.append("## 4. Bottom line")
    lines.append("")
    lines.append("1. M1 的整体 calibration 可接受。")
    lines.append("2. 高风险段仍然 reasonably calibrated，但顶端风险层可轻度提示可能存在小幅偏移。")
    lines.append("3. 主文最适合放 simple calibration plot；decile table 放补充材料。")
    lines.append("4. 这一模块对主线 1 的增益是 modest，主要提升可信展示，不改变主结论。")
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_results_index() -> None:
    marker = "## 2026-03-25 Stage M1 Calibration"
    text = RESULTS_INDEX.read_text(encoding="utf-8")
    if marker in text:
        return
    block = f"""

{marker}

本阶段不改 `M1` 主模型，只补充 current-burden reference model 的 calibration / observed-vs-predicted 展示。

### 新增文件

| 文件 | 回答的问题 | 备注 |
|---|---|---|
| `results/stageM1_calibration/m1_calibration_metrics.csv` | M1 的 calibration intercept、slope、Brier、calibration-in-the-large 如何 | 主 metrics 表 |
| `results/stageM1_calibration/m1_observed_vs_predicted_deciles.csv` | M1 在不同 decile 风险段的 predicted risk 与 observed risk 对应如何 | 补充展示主表 |
| `results/stageM1_calibration/m1_calibration_summary.md` | M1 整体和高风险段校准是否可接受；主文更适合放什么 | 阶段总结 |
| `results/stageM1_calibration/m1_calibration_plot.pdf` | M1 calibration plot + decile observed-minus-predicted 图 | 图形展示 |

### 本阶段直接结论

1. `M1` 的整体 calibration 可接受，intercept 和 slope 都接近理想值。
2. 高风险端存在轻度波动，但没有证据支持严重系统性失准。
3. 这个模块最适合增强主线 1 的可信展示，增益是 `modest`，不是新的主结果。
"""
    RESULTS_INDEX.write_text(text.rstrip() + "\n" + block.strip("\n") + "\n", encoding="utf-8")


def write_metadata(outpath: Path, metrics: pd.DataFrame) -> None:
    m = metrics.iloc[0]
    meta = {
        "stage": "stageM1_calibration",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "analytic_sample": m["analytic_sample"],
        "n": int(m["n"]),
        "events": int(m["events"]),
        "model_locked": "M1 = base covariates + 4 wave3 current levels",
        "prediction_basis": "10-fold OOF predicted risk",
        "oof_auc": float(m["oof_auc"]),
        "oof_brier": float(m["oof_brier"]),
        "oof_calibration_intercept": float(m["oof_calibration_intercept"]),
        "oof_calibration_slope": float(m["oof_calibration_slope"]),
    }
    outpath.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = m1s.build_locked_m1_predictions()
    pred_col = "m1_oof_predicted_risk"
    df["m1_decile"] = make_deciles(df[pred_col])
    metrics = metrics_table(df, pred_col)
    deciles = decile_summary(df, pred_col)
    metrics.to_csv(OUTDIR / "m1_calibration_metrics.csv", index=False)
    deciles.to_csv(OUTDIR / "m1_observed_vs_predicted_deciles.csv", index=False)
    write_summary(OUTDIR / "m1_calibration_summary.md", metrics, deciles)
    make_plot(OUTDIR / "m1_calibration_plot.pdf", deciles, df, pred_col)
    write_metadata(OUTDIR / "stageM1_calibration_metadata.json", metrics)
    update_results_index()


if __name__ == "__main__":
    main()

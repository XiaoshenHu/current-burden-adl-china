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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = project_root()
RESULTS_DIR = PROJECT_ROOT / "results" / "stage6_representation"
LOGS_DIR = PROJECT_ROOT / "logs" / "stage6_representation"

STAGE2_FEATURES = PROJECT_ROOT / "results" / "stage2_level_change" / "level_change_features.csv"
STAGE2_PERF = PROJECT_ROOT / "results" / "stage2_level_change" / "level_change_model_comparison.csv"
STAGE3_PERF = PROJECT_ROOT / "results" / "stage3_nested_validation" / "nested_validation_performance.csv"
STAGE5_RISKS = PROJECT_ROOT / "results" / "stage5_emerging_beyond_burden" / "adjusted_absolute_risk_by_phenotype.csv"
STAGE5_CONTRASTS = PROJECT_ROOT / "results" / "stage5_emerging_beyond_burden" / "emerging_pairwise_contrasts.csv"
ORIGINAL_PERF = expanded_rerun_root() / "07_tables_figures" / "table_prediction_performance.csv"
RESULTS_INDEX = PROJECT_ROOT / "RESULTS_INDEX.md"

SEED = 20260325
BASE_COVARS = ["baseline_age_2011", "female", "rural_hukou_2011", "low_self_education"]
LEVEL_VARS = ["dep_level_at_w3", "cogdef_level_at_w3", "physical_lim_level_at_w3", "multimorb_level_at_w3"]
SLOPE_VARS = ["dep_pre_w3_slope", "cogdef_pre_w3_slope", "physical_lim_pre_w3_slope", "multimorb_pre_w3_slope"]
POSTERIOR_VARS = [
    "posterior_emerging_multidomain_decline",
    "posterior_high_burden_mixed_pattern",
    "posterior_severe_multidomain_decline",
]
PHENOTYPE_ORDER = [
    "Stable low-burden",
    "Emerging multidomain decline",
    "High-burden mixed pattern",
    "Severe multidomain decline",
]


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(STAGE2_FEATURES)
    perf2 = pd.read_csv(STAGE2_PERF)
    perf3 = pd.read_csv(STAGE3_PERF)
    risks5 = pd.read_csv(STAGE5_RISKS)
    contrasts5 = pd.read_csv(STAGE5_CONTRASTS)
    original = pd.read_csv(ORIGINAL_PERF)
    return features, perf2, perf3, risks5, contrasts5, original


def fit_glm_predict(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    X_use = sm.add_constant(X.astype(float).reset_index(drop=True), has_constant="add")
    y_use = y.astype(int).reset_index(drop=True)
    model = sm.GLM(y_use, X_use, family=sm.families.Binomial()).fit()
    return np.asarray(model.predict(X_use))


def fit_ridge_predict(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
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
                    random_state=SEED,
                ),
            ),
        ]
    )
    pipe.fit(X.reset_index(drop=True), y.astype(int).reset_index(drop=True))
    return pipe.predict_proba(X.reset_index(drop=True))[:, 1]


def quintile_event_gap(pred: np.ndarray, y: pd.Series) -> tuple[float, float, float, float]:
    work = pd.DataFrame({"pred": pred, "y": y.astype(int).reset_index(drop=True)})
    work["quintile"] = pd.qcut(work["pred"].rank(method="first"), q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    event_rates = work.groupby("quintile", observed=False)["y"].mean().reindex(["Q1", "Q2", "Q3", "Q4", "Q5"])
    q1 = float(event_rates.loc["Q1"])
    q5 = float(event_rates.loc["Q5"])
    rr = float(q5 / q1) if q1 > 0 else np.nan
    return q1, q5, q5 - q1, rr


def make_simple_count(df: pd.DataFrame) -> pd.Series:
    thresholds = df[LEVEL_VARS].quantile(0.75)
    count = sum((df[var] >= thresholds[var]).astype(int) for var in LEVEL_VARS)
    return count.astype(int)


def build_representation_metrics(features: pd.DataFrame) -> pd.DataFrame:
    y = features["incident_adl_w3_w4"].astype(int)
    base = features[BASE_COVARS].astype(float)
    pheno = pd.get_dummies(
        pd.Categorical(features["trajectory_group"], categories=PHENOTYPE_ORDER, ordered=True),
        prefix="traj",
        drop_first=True,
        dtype=float,
    )
    post = features[POSTERIOR_VARS].astype(float)
    cont8 = features[LEVEL_VARS + SLOPE_VARS].astype(float)
    simple_count = make_simple_count(features).rename("adverse_domain_count").to_frame()

    rows = []
    reps = {
        "hard_phenotype": fit_glm_predict(pd.concat([base, pheno], axis=1), y),
        "posterior_probabilities": fit_glm_predict(pd.concat([base, post], axis=1), y),
        "continuous_8_unpenalized": fit_glm_predict(pd.concat([base, cont8], axis=1), y),
        "continuous_8_ridge": fit_ridge_predict(pd.concat([base, cont8], axis=1), y),
        "simple_count_rule": fit_glm_predict(pd.concat([base, simple_count], axis=1), y),
    }
    for rep, pred in reps.items():
        q1, q5, gap, rr = quintile_event_gap(pred, y)
        rows.append(
            {
                "representation_id": rep,
                "q1_event_rate": q1,
                "q5_event_rate": q5,
                "q5_minus_q1_event_rate": gap,
                "q5_q1_event_rate_ratio": rr,
                "mean_predicted_risk": float(np.mean(pred)),
                "sd_predicted_risk": float(np.std(pred)),
            }
        )
    return pd.DataFrame(rows)


def performance_lookup_stage2(perf2: pd.DataFrame, model_code: str) -> dict:
    row = perf2[(perf2["row_type"] == "model_performance") & (perf2["model_code"] == model_code)].iloc[0]
    return {
        "stage2_oof_auc": float(row["oof_auc"]),
        "stage2_oof_pr_auc": float(row["oof_pr_auc"]),
        "stage2_oof_brier": float(row["oof_brier"]),
        "stage2_calibration_intercept": float(row["oof_calibration_intercept"]),
        "stage2_calibration_slope": float(row["oof_calibration_slope"]),
        "stage2_nb_10_20": float(row["mean_net_benefit_10_20"]),
    }


def performance_lookup_stage3(perf3: pd.DataFrame, model_code: str) -> dict:
    sub = perf3[(perf3["row_type"] == "model_performance") & (perf3["model_code"] == model_code)]
    if sub.empty:
        return {
            "stage3_oof_auc": np.nan,
            "stage3_oof_pr_auc": np.nan,
            "stage3_oof_brier": np.nan,
            "stage3_calibration_intercept": np.nan,
            "stage3_calibration_slope": np.nan,
            "stage3_nb_10_20": np.nan,
        }
    row = sub.iloc[0]
    return {
        "stage3_oof_auc": float(row["oof_auc"]),
        "stage3_oof_pr_auc": float(row["oof_pr_auc"]),
        "stage3_oof_brier": float(row["oof_brier"]),
        "stage3_calibration_intercept": float(row["oof_calibration_intercept"]),
        "stage3_calibration_slope": float(row["oof_calibration_slope"]),
        "stage3_nb_10_20": float(row["mean_net_benefit_10_20"]),
    }


def performance_lookup_original(original: pd.DataFrame, model_spec: str) -> dict:
    row = original[original["model_spec"] == model_spec].iloc[0]
    return {
        "stage2_oof_auc": float(row["oof_auc"]),
        "stage2_oof_pr_auc": np.nan,
        "stage2_oof_brier": float(row["oof_brier"]),
        "stage2_calibration_intercept": float(row["oof_calibration_intercept"]),
        "stage2_calibration_slope": float(row["oof_calibration_slope"]),
        "stage2_nb_10_20": np.nan,
        "stage3_oof_auc": np.nan,
        "stage3_oof_pr_auc": np.nan,
        "stage3_oof_brier": np.nan,
        "stage3_calibration_intercept": np.nan,
        "stage3_calibration_slope": np.nan,
        "stage3_nb_10_20": np.nan,
    }


def stability_label(stage2_auc: float, stage3_auc: float) -> str:
    if pd.isna(stage3_auc):
        return "not formally revalidated"
    gap = abs(stage3_auc - stage2_auc)
    if gap <= 0.003:
        return "high"
    if gap <= 0.01:
        return "moderate"
    return "low"


def build_comparison_table(
    risk_metrics: pd.DataFrame,
    perf2: pd.DataFrame,
    perf3: pd.DataFrame,
    risks5: pd.DataFrame,
    contrasts5: pd.DataFrame,
    original: pd.DataFrame,
) -> pd.DataFrame:
    risk_map = risk_metrics.set_index("representation_id").to_dict(orient="index")
    emerging_burden_adjusted = contrasts5[
        (contrasts5["sample_name"] == "unified_locked")
        & (contrasts5["model_name"] == "base_levels_phenotype")
        & (contrasts5["contrast"] == "Emerging vs Stable")
    ].iloc[0]["effect_95ci"]
    hard_adj_risks = risks5[
        (risks5["sample_name"] == "unified_locked")
        & (risks5["model_name"] == "base_levels_phenotype")
    ][["phenotype", "predicted_probability"]]
    hard_risk_note = "; ".join(
        f"{r.phenotype}={r.predicted_probability:.3f}" for r in hard_adj_risks.itertuples(index=False)
    )

    specs = [
        {
            "representation_id": "hard_phenotype",
            "representation_label": "Hard phenotype class assignment",
            "stage2_model_code": "M3",
            "stage3_model_code": "M3",
            "performance_source": "stage2+stage3",
            "interpretability_note": "Highest clinical interpretability because it yields named subgroups and direct phenotype labels.",
            "interpretability_score": 5,
            "emerging_identification_ability": "Directly identifies Emerging as a named subgroup.",
            "emerging_identification_score": 5,
            "absolute_risk_note": f"Direct phenotype-specific adjusted risks available; burden-adjusted Emerging vs Stable = {emerging_burden_adjusted}; {hard_risk_note}",
            "positioning": "explanatory subtype with adjunctive prognostic utility",
        },
        {
            "representation_id": "posterior_probabilities",
            "representation_label": "Posterior probabilities of phenotype classes",
            "stage2_model_code": "M6",
            "stage3_model_code": "M6",
            "performance_source": "stage2+stage3",
            "interpretability_note": "Probabilistic rather than categorical; retains uncertainty but is harder to explain clinically.",
            "interpretability_score": 4,
            "emerging_identification_ability": "Probabilistic identification of Emerging; less direct than hard assignment.",
            "emerging_identification_score": 4,
            "absolute_risk_note": "Can support soft subgroup profiling, but no cleaner absolute risk separation than hard assignment.",
            "positioning": "adjunctive technical representation, not preferred for manuscript-facing clinical use",
        },
        {
            "representation_id": "continuous_8_unpenalized",
            "representation_label": "8 continuous features (4 levels + 4 slopes; unpenalized)",
            "stage2_model_code": "M2",
            "stage3_model_code": "M2",
            "performance_source": "stage2+stage3",
            "interpretability_note": "Mechanistically transparent but clinically less compact; needs coefficient-level interpretation.",
            "interpretability_score": 2,
            "emerging_identification_ability": "Does not directly identify Emerging; subgroup recovery would require post-hoc clustering or thresholds.",
            "emerging_identification_score": 1,
            "absolute_risk_note": "Strongest direct risk stratification among explicit non-clustered representations.",
            "positioning": "best explanatory prognostic representation if compression into phenotype is not required",
        },
        {
            "representation_id": "continuous_8_ridge",
            "representation_label": "8 continuous features (4 levels + 4 slopes; ridge-penalized)",
            "stage2_model_code": "M5",
            "stage3_model_code": "M5",
            "performance_source": "stage2+stage3",
            "interpretability_note": "Best-performing predictive form, but least intuitive for phenotype-level clinical storytelling.",
            "interpretability_score": 2,
            "emerging_identification_ability": "No direct Emerging label; useful for prediction, not subgroup naming.",
            "emerging_identification_score": 1,
            "absolute_risk_note": "Best overall risk ordering and most stable net benefit among continuous representations.",
            "positioning": "best-performing predictive representation",
        },
        {
            "representation_id": "simple_count_rule",
            "representation_label": "Simple adverse-domain count (rule-based comparator)",
            "stage2_model_code": "simple_count",
            "stage3_model_code": None,
            "performance_source": "original pipeline only",
            "interpretability_note": "Very simple and portable, but discards both trajectory shape and domain-specific weighting.",
            "interpretability_score": 4,
            "emerging_identification_ability": "Cannot identify Emerging specifically; only provides coarse burden ranking.",
            "emerging_identification_score": 1,
            "absolute_risk_note": "Provides crude absolute risk ordering, but not subgroup-specific risk profiles.",
            "positioning": "parsimonious benchmark rather than preferred final representation",
        },
    ]

    rows = []
    for spec in specs:
        rep_id = spec["representation_id"]
        if spec["performance_source"] == "original pipeline only":
            perf = performance_lookup_original(original, "simple_count")
        else:
            perf = {}
            perf.update(performance_lookup_stage2(perf2, spec["stage2_model_code"]))
            perf.update(performance_lookup_stage3(perf3, spec["stage3_model_code"]))
        risk = risk_map[rep_id]
        stage2_auc = perf["stage2_oof_auc"]
        stage3_auc = perf["stage3_oof_auc"]
        rows.append(
            {
                "representation_id": rep_id,
                "representation_label": spec["representation_label"],
                "performance_source": spec["performance_source"],
                **perf,
                "auc_change_stage3_minus_stage2": stage3_auc - stage2_auc if pd.notna(stage3_auc) else np.nan,
                "stability_label": stability_label(stage2_auc, stage3_auc),
                "q1_event_rate": risk["q1_event_rate"],
                "q5_event_rate": risk["q5_event_rate"],
                "q5_minus_q1_event_rate": risk["q5_minus_q1_event_rate"],
                "q5_q1_event_rate_ratio": risk["q5_q1_event_rate_ratio"],
                "mean_predicted_risk": risk["mean_predicted_risk"],
                "sd_predicted_risk": risk["sd_predicted_risk"],
                "interpretability_score": spec["interpretability_score"],
                "interpretability_note": spec["interpretability_note"],
                "emerging_identification_score": spec["emerging_identification_score"],
                "emerging_identification_ability": spec["emerging_identification_ability"],
                "absolute_risk_note": spec["absolute_risk_note"],
                "positioning_recommendation": spec["positioning"],
            }
        )

    df = pd.DataFrame(rows)
    best_auc = float(df["stage3_oof_auc"].dropna().max())
    df["auc_loss_vs_best_nested"] = best_auc - df["stage3_oof_auc"]
    df["information_loss_vs_continuous"] = np.select(
        [
            df["stage3_oof_auc"].isna(),
            df["auc_loss_vs_best_nested"] <= 0.002,
            df["auc_loss_vs_best_nested"] <= 0.01,
        ],
        [
            "not assessed under strict nested validation",
            "minimal",
            "modest",
        ],
        default="substantial",
    )
    return df


def write_summary(comp: pd.DataFrame) -> None:
    def row(rep_id: str) -> pd.Series:
        return comp[comp["representation_id"] == rep_id].iloc[0]

    hard = row("hard_phenotype")
    post = row("posterior_probabilities")
    cont_u = row("continuous_8_unpenalized")
    cont_r = row("continuous_8_ridge")
    rule = row("simple_count_rule")

    text = f"""# Stage 6 Representation comparison

## 1. Stage aim

本阶段系统比较不同 trajectory 信息表示方式，回答：

1. hard phenotype 是否损失了太多信息？
2. posterior probabilities 是否优于 hard assignment？
3. continuous representation 是否是更优的 prognostic form？
4. phenotype 在论文里最终应如何定位？

## 2. Compared representations

- Hard phenotype class assignment
- Posterior probabilities of classes
- 8 continuous features (4 levels + 4 slopes; unpenalized)
- 8 continuous features (4 levels + 4 slopes; ridge-penalized)
- Simple adverse-domain count（rule-based comparator）

## 3. Direct comparison

### Discrimination and calibration

- Hard phenotype: Stage 2 OOF AUC `{hard['stage2_oof_auc']:.3f}`; Stage 3 strict nested OOF AUC `{hard['stage3_oof_auc']:.3f}`; stability `{hard['stability_label']}`.
- Posterior probabilities: Stage 2 OOF AUC `{post['stage2_oof_auc']:.3f}`; Stage 3 OOF AUC `{post['stage3_oof_auc']:.3f}`; stability `{post['stability_label']}`.
- 8 continuous features, unpenalized: Stage 2 OOF AUC `{cont_u['stage2_oof_auc']:.3f}`; Stage 3 OOF AUC `{cont_u['stage3_oof_auc']:.3f}`; stability `{cont_u['stability_label']}`.
- 8 continuous features, ridge: Stage 2 OOF AUC `{cont_r['stage2_oof_auc']:.3f}`; Stage 3 OOF AUC `{cont_r['stage3_oof_auc']:.3f}`; stability `{cont_r['stability_label']}`.
- Simple count comparator: original pipeline OOF AUC `{rule['stage2_oof_auc']:.3f}`; no separate Stage 3 strict nested rerun was added because this comparator is secondary.

### Information loss

- Relative to the best nested-performing representation, hard phenotype loses `{hard['auc_loss_vs_best_nested']:.3f}` AUC units.
- Posterior probabilities lose `{post['auc_loss_vs_best_nested']:.3f}` AUC units.
- Unpenalized 8-feature representation loses `{cont_u['auc_loss_vs_best_nested']:.3f}` AUC units.
- Ridge 8-feature representation is the nested best-performing form by construction in this comparison set.

### Emerging subgroup identification

- Hard phenotype: {hard['emerging_identification_ability']}
- Posterior probabilities: {post['emerging_identification_ability']}
- Unpenalized 8-feature representation: {cont_u['emerging_identification_ability']}
- Ridge 8-feature representation: {cont_r['emerging_identification_ability']}
- Simple count comparator: {rule['emerging_identification_ability']}

### Absolute risk stratification help

Using same-sample fitted risk quintiles, the `Q5-Q1` event-rate gaps were:

- Hard phenotype representation: `{hard['q5_minus_q1_event_rate']:.3f}`
- Posterior probabilities: `{post['q5_minus_q1_event_rate']:.3f}`
- Unpenalized 8-feature representation: `{cont_u['q5_minus_q1_event_rate']:.3f}`
- Ridge 8-feature representation: `{cont_r['q5_minus_q1_event_rate']:.3f}`
- Simple count comparator: `{rule['q5_minus_q1_event_rate']:.3f}`

For hard phenotype specifically, Stage 5 also showed that burden-adjusted Emerging remained elevated versus Stable, and direct phenotype-specific adjusted risks remained clinically interpretable.

## 4. Direct answers

### Does hard phenotype lose too much information?

Yes for pure prediction. Hard phenotype is materially weaker than either continuous 8-feature representation in both the original and strict nested comparisons. It does not lose all prognostic value, but it loses enough information that it should not be treated as the preferred predictive form.

### Are posterior probabilities better than hard assignment?

No. Posterior probabilities were essentially indistinguishable from hard assignment in Stage 2 and Stage 3. They preserved uncertainty conceptually, but did not recover meaningful predictive performance.

### Is continuous representation the better prognostic form?

Yes. The explicit level-plus-slope representation is the strongest and most stable prognostic form. The ridge-penalized 8-feature model is the best-performing predictive representation, and the unpenalized 8-feature model performs almost identically, indicating that the gain comes mainly from the information content rather than from the penalty itself.

### How should phenotype be positioned in the paper?

The most defensible positioning is:

1. **Primary positioning**: explanatory subtype
2. **Secondary positioning**: adjunctive prognostic tool
3. **Not supported**: best-performing predictive representation

In other words, phenotype is most useful because it compresses multidomain longitudinal information into clinically interpretable groups, especially the Emerging subgroup, rather than because it is the strongest predictive encoding.
"""
    (RESULTS_DIR / "representation_summary.md").write_text(text, encoding="utf-8")


def build_plots(comp: pd.DataFrame) -> None:
    plot_df = comp.copy()
    order = [
        "hard_phenotype",
        "posterior_probabilities",
        "continuous_8_unpenalized",
        "continuous_8_ridge",
        "simple_count_rule",
    ]
    plot_df["representation_id"] = pd.Categorical(plot_df["representation_id"], categories=order, ordered=True)
    plot_df = plot_df.sort_values("representation_id")
    labels = plot_df["representation_label"].tolist()

    with PdfPages(RESULTS_DIR / "representation_plots.pdf") as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        x = np.arange(len(plot_df))
        width = 0.36
        axes[0].bar(x - width / 2, plot_df["stage2_oof_auc"], width=width, label="Stage 2 OOF AUC", color="#4C78A8")
        stage3 = plot_df["stage3_oof_auc"].copy()
        axes[0].bar(x + width / 2, stage3.fillna(0), width=width, label="Stage 3 nested OOF AUC", color="#F58518")
        for i, val in enumerate(stage3):
            if pd.isna(val):
                axes[0].text(i + width / 2, 0.67, "NA", ha="center", va="bottom", fontsize=8, rotation=90)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=25, ha="right")
        axes[0].set_ylim(0.65, 0.79)
        axes[0].set_title("Discrimination by representation")
        axes[0].legend(frameon=False, fontsize=8)

        axes[1].bar(x, plot_df["q5_minus_q1_event_rate"], color="#54A24B")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=25, ha="right")
        axes[1].set_title("Absolute-risk stratification help")
        axes[1].set_ylabel("Q5 - Q1 observed event-rate gap")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        heat = plot_df[["interpretability_score", "emerging_identification_score"]].to_numpy(dtype=float).T
        im = ax.imshow(heat, cmap="YlGnBu", aspect="auto", vmin=1, vmax=5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Interpretability", "Emerging identification"])
        ax.set_xticks(np.arange(len(plot_df)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):
                ax.text(j, i, f"{heat[i, j]:.0f}", ha="center", va="center", fontsize=9)
        ax.set_title("Qualitative representation comparison")
        fig.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def update_results_index() -> None:
    addition = """

## 2026-03-25 Stage 6 Representation comparison

本阶段系统比较了不同 trajectory 信息表示方式，重点区分：
- 哪种形式更适合做 prediction；
- 哪种形式更适合做 phenotype-based explanation；
- `Emerging` 作为 subgroup 的识别价值是否来自 hard class 本身，还是来自更底层的 continuous features。

### 新增文件

| 文件 | 回答的问题 | 备注 |
|---|---|---|
| `scripts/stage6_representation/stage6_representation.py` | 如何复用 Stage 2、Stage 3 和 Stage 5 的结果，系统比较 hard phenotype、posterior probabilities、continuous 8 features 和 rule-based comparator | Stage 6 独立脚本 |
| `results/stage6_representation/representation_comparison.csv` | 各表示方式在 discrimination、calibration、stability、interpretability、Emerging identification 和 absolute risk stratification 上如何比较 | 主比较表 |
| `results/stage6_representation/representation_summary.md` | hard phenotype 是否损失太多信息；posterior 是否优于 hard；continuous 是否更优；phenotype 最终应如何定位 | Stage 6 总结文件 |
| `results/stage6_representation/representation_plots.pdf` | 表示方式比较的性能图、风险分层图和定性热图 | 便于人工核查 |

### 本阶段给出的直接结论

1. Hard phenotype 在 prediction 上明显弱于 continuous 8-feature representations，说明它确实损失了相当一部分信息。
2. Posterior probabilities 几乎不优于 hard assignment，因此“soft assignment”并没有把丢掉的信息重新找回来。
3. Continuous 8-feature representation（尤其 ridge-penalized 版本）是当前最优、最稳的 prognostic form。
4. phenotype 最适合被定位为：
   - **primary**：explanatory subtype
   - **secondary**：adjunctive prognostic tool
   - **not supported**：best-performing predictive representation

### 下一阶段建议

只在你确认后进入：

1. Stage 7 phenotype reconstruction without adversity-gated entry
"""
    current = RESULTS_INDEX.read_text(encoding="utf-8")
    if "## 2026-03-25 Stage 6 Representation comparison" not in current:
        RESULTS_INDEX.write_text(current + addition, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    np.random.seed(SEED)
    random.seed(SEED)

    features, perf2, perf3, risks5, contrasts5, original = load_inputs()
    risk_metrics = build_representation_metrics(features)
    comp = build_comparison_table(risk_metrics, perf2, perf3, risks5, contrasts5, original)
    comp.to_csv(RESULTS_DIR / "representation_comparison.csv", index=False)
    build_plots(comp)
    write_summary(comp)
    update_results_index()

    meta = {
        "stage": "stage6_representation",
        "seed": SEED,
        "n_unified_locked": int(len(features)),
        "representations_compared": int(len(comp)),
    }
    (RESULTS_DIR / "stage6_run_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()

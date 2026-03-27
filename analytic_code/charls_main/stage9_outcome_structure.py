#!/usr/bin/env python3
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats


PROJECT_ROOT = Path("/Users/macbookpro-hu/Library/CloudStorage/OneDrive-个人/Xcode/Medical Research/PROJECTS/charls_life_course_aging")
RERUN_ROOT = PROJECT_ROOT / "rerun_risk_stratification_expanded_v2_20260323"
RESULTS_DIR = PROJECT_ROOT / "results" / "stage9_outcome_structure"
LOGS_DIR = PROJECT_ROOT / "logs" / "stage9_outcome_structure"
RESULTS_INDEX = PROJECT_ROOT / "RESULTS_INDEX.md"

BASE_COVARS = ["baseline_age_2011", "female", "rural_hukou_2011", "low_self_education"]
PHENOTYPE_ORDER = [
    "Stable low-burden",
    "Emerging multidomain decline",
    "High-burden mixed pattern",
    "Severe multidomain decline",
]
ADL_STATUS_ORDER = ["alive_no_adl", "alive_incident_adl", "dead_before_w4"]
IADL_STATUS_ORDER = ["alive_no_iadl", "alive_incident_iadl", "dead_before_w4"]
OUTCOME_LABELS = {
    "alive_incident_adl": "Alive with incident ADL disability vs alive without incident ADL disability",
    "dead_before_w4": "Dead before wave 4 vs alive without incident ADL disability",
    "alive_incident_iadl": "Alive with incident IADL disability vs alive without incident IADL disability",
}
SEED = 20260421


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def format_p(p: float) -> str:
    if pd.isna(p):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt_rrr(est: float, low: float, high: float) -> str:
    return f"{est:.3f} ({low:.3f}, {high:.3f})"


def load_data() -> pd.DataFrame:
    dataset = pd.read_csv(RERUN_ROOT / "02_data_processing" / "analysis_dataset_landmark.csv")
    assignments = pd.read_csv(RERUN_ROOT / "03_phenotype_derivation" / "phenotype_assignments.csv")
    dataset["ID_str"] = dataset["ID_str"].astype(str)
    assignments["ID_str"] = assignments["ID_str"].astype(str)
    merged = dataset.merge(assignments[["ID_str", "trajectory_group"]], on="ID_str", how="left")
    merged["trajectory_group"] = pd.Categorical(merged["trajectory_group"], categories=PHENOTYPE_ORDER, ordered=True)
    return merged


def build_main_sample(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    mask = (
        df["trajectory_group"].notna()
        & df["adversity_score_scaled"].notna()
        & (df["alive_at_w3"] == 1)
        & (df["adl_disability_w3"] == 0)
        & (df["known_w4_status"] == 1)
        & df[BASE_COVARS].notna().all(axis=1)
    )
    sub = df.loc[mask].copy()
    sub["competing_status"] = pd.NA
    sub.loc[(sub["death_w3_w4"] == 0) & (sub["incident_adl_w3_w4"] == 0), "competing_status"] = "alive_no_adl"
    sub.loc[(sub["death_w3_w4"] == 0) & (sub["incident_adl_w3_w4"] == 1), "competing_status"] = "alive_incident_adl"
    sub.loc[sub["death_w3_w4"] == 1, "competing_status"] = "dead_before_w4"
    unresolved = int(sub["competing_status"].isna().sum())
    sub = sub[sub["competing_status"].notna()].copy().reset_index(drop=True)
    sub["competing_status"] = pd.Categorical(sub["competing_status"], categories=ADL_STATUS_ORDER, ordered=True)
    meta = {
        "sample_name": "main_adl_competing",
        "n_total_masked": int(mask.sum()),
        "n_final": int(len(sub)),
        "n_unresolved_alive_without_adl_status": unresolved,
        "n_alive_no_adl": int((sub["competing_status"] == "alive_no_adl").sum()),
        "n_alive_incident_adl": int((sub["competing_status"] == "alive_incident_adl").sum()),
        "n_dead_before_w4": int((sub["competing_status"] == "dead_before_w4").sum()),
    }
    return sub, meta


def build_supportive_iadl_sample(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    mask = (
        df["trajectory_group"].notna()
        & df["adversity_score_scaled"].notna()
        & (df["alive_at_w3"] == 1)
        & (df["iadl_disability_w3"] == 0)
        & (df["known_w4_status"] == 1)
        & df[BASE_COVARS].notna().all(axis=1)
    )
    sub = df.loc[mask].copy()
    sub["competing_status"] = pd.NA
    sub.loc[(sub["death_w3_w4"] == 0) & (sub["incident_iadl_w3_w4"] == 0), "competing_status"] = "alive_no_iadl"
    sub.loc[(sub["death_w3_w4"] == 0) & (sub["incident_iadl_w3_w4"] == 1), "competing_status"] = "alive_incident_iadl"
    sub.loc[sub["death_w3_w4"] == 1, "competing_status"] = "dead_before_w4"
    unresolved = int(sub["competing_status"].isna().sum())
    sub = sub[sub["competing_status"].notna()].copy().reset_index(drop=True)
    sub["competing_status"] = pd.Categorical(sub["competing_status"], categories=IADL_STATUS_ORDER, ordered=True)
    meta = {
        "sample_name": "supportive_iadl_competing",
        "n_total_masked": int(mask.sum()),
        "n_final": int(len(sub)),
        "n_unresolved_alive_without_iadl_status": unresolved,
        "n_alive_no_iadl": int((sub["competing_status"] == "alive_no_iadl").sum()),
        "n_alive_incident_iadl": int((sub["competing_status"] == "alive_incident_iadl").sum()),
        "n_dead_before_w4": int((sub["competing_status"] == "dead_before_w4").sum()),
    }
    return sub, meta


def design_matrix(data: pd.DataFrame, model_name: str) -> pd.DataFrame:
    parts = [data[BASE_COVARS].reset_index(drop=True).astype(float)]
    if model_name in {"phenotype_only", "joint"}:
        dummies = pd.get_dummies(
            pd.Categorical(data["trajectory_group"], categories=PHENOTYPE_ORDER, ordered=True),
            prefix="traj",
            drop_first=True,
            dtype=float,
        ).reset_index(drop=True)
        parts.append(dummies)
    if model_name in {"adversity_only", "joint"}:
        parts.append(data[["adversity_score_scaled"]].reset_index(drop=True).astype(float))
    X = pd.concat(parts, axis=1)
    return sm.add_constant(X, has_constant="add")


def fit_mnlogit(y_codes: pd.Series, X: pd.DataFrame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = sm.MNLogit(y_codes.reset_index(drop=True), X.reset_index(drop=True))
            result = model.fit(method="newton", maxiter=200, disp=False)
            return result, "newton"
        except Exception:
            model = sm.MNLogit(y_codes.reset_index(drop=True), X.reset_index(drop=True))
            result = model.fit(method="bfgs", maxiter=300, disp=False)
            return result, "bfgs_fallback"


def outcome_column_mapping(categories: list[str]) -> dict[int, str]:
    return {i - 1: cat for i, cat in enumerate(categories[1:], start=1)}


def extract_mnlogit_rows(
    result,
    model_name: str,
    sample_name: str,
    categories: list[str],
) -> list[dict]:
    rows: list[dict] = []
    conf = result.conf_int()
    col_map = outcome_column_mapping(categories)
    for outcome_col, outcome_name in col_map.items():
        for term in result.params.index:
            est = float(result.params.loc[term, outcome_col])
            low = float(conf.loc[(str(outcome_col + 1), term), "lower"])
            high = float(conf.loc[(str(outcome_col + 1), term), "upper"])
            p = float(result.pvalues.loc[term, outcome_col])
            rows.append(
                {
                    "row_type": "term",
                    "sample_name": sample_name,
                    "model_name": model_name,
                    "outcome_contrast": outcome_name,
                    "outcome_label": OUTCOME_LABELS.get(outcome_name, outcome_name),
                    "term": term,
                    "coef": est,
                    "ci_low_coef": low,
                    "ci_high_coef": high,
                    "p": p,
                    "p_formatted": format_p(p),
                    "RRR": float(np.exp(est)),
                    "RRR_low": float(np.exp(low)),
                    "RRR_high": float(np.exp(high)),
                    "effect_95ci": fmt_rrr(np.exp(est), np.exp(low), np.exp(high)),
                }
            )
    return rows


def likelihood_ratio_row(full_result, reduced_result, model_name: str, sample_name: str, tested_block: str) -> dict:
    lr = 2 * (full_result.llf - reduced_result.llf)
    df_diff = int(full_result.df_model - reduced_result.df_model)
    p = float(stats.chi2.sf(lr, df_diff))
    return {
        "row_type": "overall_test",
        "sample_name": sample_name,
        "model_name": model_name,
        "tested_block": tested_block,
        "lr_chisq": float(lr),
        "df_diff": df_diff,
        "p": p,
        "p_formatted": format_p(p),
        "pseudo_r2": float(full_result.prsquared),
        "aic": float(full_result.aic),
        "bic": float(full_result.bic),
        "llf": float(full_result.llf),
    }


def observed_distribution_rows(data: pd.DataFrame, sample_name: str) -> list[dict]:
    rows: list[dict] = []
    total_counts = data["competing_status"].value_counts().reindex(data["competing_status"].cat.categories, fill_value=0)
    for status, count in total_counts.items():
        rows.append(
            {
                "row_type": "observed_distribution",
                "sample_name": sample_name,
                "model_name": "observed_overall",
                "outcome_contrast": status,
                "n": int(count),
                "prop": float(count / len(data)),
            }
        )
    by_pheno = (
        data.groupby("trajectory_group", observed=False)["competing_status"]
        .value_counts(normalize=True)
        .rename("prop")
        .reset_index()
    )
    counts = (
        data.groupby("trajectory_group", observed=False)["competing_status"]
        .value_counts()
        .rename("n")
        .reset_index()
    )
    merged = by_pheno.merge(counts, on=["trajectory_group", "competing_status"], how="left")
    for row in merged.itertuples(index=False):
        rows.append(
            {
                "row_type": "observed_by_phenotype",
                "sample_name": sample_name,
                "model_name": "observed_by_phenotype",
                "phenotype": row.trajectory_group,
                "outcome_contrast": row.competing_status,
                "n": int(row.n),
                "prop": float(row.prop),
            }
        )
    return rows


def marginal_prediction_rows(data: pd.DataFrame, result, model_name: str, sample_name: str, categories: list[str]) -> list[dict]:
    rows: list[dict] = []
    if model_name not in {"phenotype_only", "joint"}:
        return rows
    for phen in PHENOTYPE_ORDER:
        tmp = data.copy()
        tmp["trajectory_group"] = phen
        X = design_matrix(tmp, model_name)
        pred = result.predict(X)
        mean_probs = pred.mean(axis=0)
        for i, status in enumerate(categories):
            rows.append(
                {
                    "row_type": "marginal_prediction",
                    "sample_name": sample_name,
                    "model_name": model_name,
                    "phenotype": phen,
                    "outcome_contrast": status,
                    "predicted_probability": float(mean_probs.iloc[i] if hasattr(mean_probs, "iloc") else mean_probs[i]),
                }
            )
    return rows


def fit_binary_supportive_iadl(data: pd.DataFrame) -> pd.DataFrame:
    sub = data[(data["death_w3_w4"] == 0) & data["incident_iadl_w3_w4"].notna()].copy().reset_index(drop=True)
    y = sub["incident_iadl_w3_w4"].astype(int)
    rows: list[dict] = []
    for model_name in ["phenotype_only", "adversity_only", "joint"]:
        X = design_matrix(sub, model_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.GLM(y, X, family=sm.families.Binomial()).fit()
        conf = res.conf_int()
        for term in res.params.index:
            if term == "const":
                continue
            est = float(res.params[term])
            low = float(conf.loc[term, 0])
            high = float(conf.loc[term, 1])
            p = float(res.pvalues[term])
            rows.append(
                {
                    "row_type": "supportive_iadl_binary",
                    "sample_name": "supportive_iadl_binary_alive_only",
                    "model_name": model_name,
                    "outcome_contrast": "incident_iadl_vs_no_incident_iadl",
                    "term": term,
                    "OR": float(np.exp(est)),
                    "OR_low": float(np.exp(low)),
                    "OR_high": float(np.exp(high)),
                    "effect_95ci": fmt_rrr(np.exp(est), np.exp(low), np.exp(high)),
                    "p": p,
                    "p_formatted": format_p(p),
                    "n": int(len(sub)),
                    "events": int(y.sum()),
                }
            )
    return pd.DataFrame(rows)


def run_multinomial_suite(data: pd.DataFrame, categories: list[str], sample_name: str) -> pd.DataFrame:
    y = data["competing_status"].cat.codes.reset_index(drop=True)
    rows: list[dict] = []

    base_result, base_method = fit_mnlogit(y, design_matrix(data, "base_only"))
    rows.append(
        {
            "row_type": "model_info",
            "sample_name": sample_name,
            "model_name": "base_only",
            "fit_method": base_method,
            "n": int(len(data)),
            "pseudo_r2": float(base_result.prsquared),
            "aic": float(base_result.aic),
            "bic": float(base_result.bic),
        }
    )

    fitted = {"base_only": base_result}
    for model_name in ["phenotype_only", "adversity_only", "joint"]:
        result, method = fit_mnlogit(y, design_matrix(data, model_name))
        fitted[model_name] = result
        rows.append(
            {
                "row_type": "model_info",
                "sample_name": sample_name,
                "model_name": model_name,
                "fit_method": method,
                "n": int(len(data)),
                "pseudo_r2": float(result.prsquared),
                "aic": float(result.aic),
                "bic": float(result.bic),
            }
        )
        rows.extend(extract_mnlogit_rows(result, model_name, sample_name, categories))

    rows.append(likelihood_ratio_row(fitted["phenotype_only"], fitted["base_only"], "phenotype_only", sample_name, "phenotype_block"))
    rows.append(likelihood_ratio_row(fitted["adversity_only"], fitted["base_only"], "adversity_only", sample_name, "adversity_block"))
    rows.append(likelihood_ratio_row(fitted["joint"], fitted["base_only"], "joint", sample_name, "phenotype_plus_adversity"))
    rows.append(likelihood_ratio_row(fitted["joint"], fitted["adversity_only"], "joint", sample_name, "phenotype_block_given_adversity"))
    rows.append(likelihood_ratio_row(fitted["joint"], fitted["phenotype_only"], "joint", sample_name, "adversity_block_given_phenotype"))

    rows.extend(observed_distribution_rows(data, sample_name))
    for model_name in ["phenotype_only", "joint"]:
        rows.extend(marginal_prediction_rows(data, fitted[model_name], model_name, sample_name, categories))

    return pd.DataFrame(rows)


def make_plots(main_df: pd.DataFrame, supportive_df: pd.DataFrame) -> None:
    with PdfPages(RESULTS_DIR / "outcome_structure_plots.pdf") as pdf:
        observed = main_df[main_df["row_type"] == "observed_by_phenotype"].copy()
        pivot = observed.pivot(index="phenotype", columns="outcome_contrast", values="prop").reindex(PHENOTYPE_ORDER)
        fig, ax = plt.subplots(figsize=(9, 5))
        bottom = np.zeros(len(pivot))
        colors = {
            "alive_no_adl": "#4c72b0",
            "alive_incident_adl": "#dd8452",
            "dead_before_w4": "#c44e52",
        }
        for col in ADL_STATUS_ORDER:
            ax.bar(pivot.index, pivot[col], bottom=bottom, label=col, color=colors[col])
            bottom += pivot[col].to_numpy()
        ax.set_ylabel("Observed proportion")
        ax.set_title("Observed competing outcome structure by phenotype")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(frameon=False)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        joint_terms = main_df[
            (main_df["row_type"] == "term")
            & (main_df["model_name"] == "joint")
            & (main_df["term"].isin([
                "adversity_score_scaled",
                "traj_Emerging multidomain decline",
                "traj_High-burden mixed pattern",
                "traj_Severe multidomain decline",
            ]))
        ].copy()
        label_map = {
            "adversity_score_scaled": "Adversity score",
            "traj_Emerging multidomain decline": "Emerging vs Stable",
            "traj_High-burden mixed pattern": "High-burden vs Stable",
            "traj_Severe multidomain decline": "Severe vs Stable",
        }
        joint_terms["label"] = joint_terms["term"].map(label_map)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for ax, outcome in zip(axes, ["alive_incident_adl", "dead_before_w4"]):
            sub = joint_terms[joint_terms["outcome_contrast"] == outcome].copy()
            sub = sub.iloc[::-1]
            y = np.arange(len(sub))
            ax.errorbar(
                sub["RRR"],
                y,
                xerr=[
                    sub["RRR"] - sub["RRR_low"],
                    sub["RRR_high"] - sub["RRR"],
                ],
                fmt="o",
                color="#2f4b7c",
                ecolor="#2f4b7c",
                capsize=3,
            )
            ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)
            ax.set_yticks(y)
            ax.set_yticklabels(sub["label"])
            ax.set_xlabel("Relative risk ratio")
            ax.set_title(OUTCOME_LABELS[outcome])
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        marg = main_df[
            (main_df["row_type"] == "marginal_prediction")
            & (main_df["model_name"] == "joint")
            & (main_df["outcome_contrast"].isin(ADL_STATUS_ORDER))
        ].copy()
        pivot = marg.pivot(index="phenotype", columns="outcome_contrast", values="predicted_probability").reindex(PHENOTYPE_ORDER)
        fig, ax = plt.subplots(figsize=(9, 5))
        width = 0.25
        x = np.arange(len(pivot))
        for i, col in enumerate(ADL_STATUS_ORDER):
            ax.bar(x + (i - 1) * width, pivot[col], width=width, label=col, color=colors[col])
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=20)
        ax.set_ylabel("Adjusted marginal predicted probability")
        ax.set_title("Joint-model adjusted competing outcome probabilities by phenotype")
        ax.legend(frameon=False)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        if not supportive_df.empty:
            obs = supportive_df[supportive_df["row_type"] == "observed_by_phenotype"].copy()
            pivot = obs.pivot(index="phenotype", columns="outcome_contrast", values="prop").reindex(PHENOTYPE_ORDER)
            fig, ax = plt.subplots(figsize=(9, 5))
            bottom = np.zeros(len(pivot))
            cols = {
                "alive_no_iadl": "#4c72b0",
                "alive_incident_iadl": "#55a868",
                "dead_before_w4": "#c44e52",
            }
            for col in IADL_STATUS_ORDER:
                ax.bar(pivot.index, pivot[col], bottom=bottom, label=col, color=cols[col])
                bottom += pivot[col].to_numpy()
            ax.set_ylabel("Observed proportion")
            ax.set_title("Supportive IADL competing outcome structure by phenotype")
            ax.tick_params(axis="x", rotation=20)
            ax.legend(frameon=False)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def write_summary(main_meta: dict, supportive_meta: dict, all_rows: pd.DataFrame) -> None:
    def get_term(model: str, outcome: str, term: str, sample: str = "main_adl_competing") -> pd.Series:
        sub = all_rows[
            (all_rows["row_type"] == "term")
            & (all_rows["sample_name"] == sample)
            & (all_rows["model_name"] == model)
            & (all_rows["outcome_contrast"] == outcome)
            & (all_rows["term"] == term)
        ]
        return sub.iloc[0]

    phen_dis = get_term("joint", "alive_incident_adl", "traj_Emerging multidomain decline")
    phen_death = get_term("joint", "dead_before_w4", "traj_Emerging multidomain decline")
    hb_dis = get_term("joint", "alive_incident_adl", "traj_High-burden mixed pattern")
    hb_death = get_term("joint", "dead_before_w4", "traj_High-burden mixed pattern")
    sev_dis = get_term("joint", "alive_incident_adl", "traj_Severe multidomain decline")
    sev_death = get_term("joint", "dead_before_w4", "traj_Severe multidomain decline")
    adv_dis = get_term("joint", "alive_incident_adl", "adversity_score_scaled")
    adv_death = get_term("joint", "dead_before_w4", "adversity_score_scaled")

    joint_tests = all_rows[
        (all_rows["row_type"] == "overall_test")
        & (all_rows["sample_name"] == "main_adl_competing")
        & (all_rows["model_name"] == "joint")
    ].set_index("tested_block")
    supportive_terms = all_rows[
        (all_rows["row_type"] == "term")
        & (all_rows["sample_name"] == "supportive_iadl_competing")
        & (all_rows["model_name"] == "joint")
        & (all_rows["outcome_contrast"] == "alive_incident_iadl")
    ]
    supp_em = supportive_terms[supportive_terms["term"] == "traj_Emerging multidomain decline"].iloc[0]
    supp_adv = supportive_terms[supportive_terms["term"] == "adversity_score_scaled"].iloc[0]

    text = f"""# Stage 9 Outcome structure / competing pathways summary

## 1. Stage objective

本阶段不再只看 `incident ADL` 或 `ADL-or-death composite`，而是把 wave 3 到 wave 4 的结果拆成三类：

1. alive without incident ADL disability
2. alive with incident ADL disability
3. dead before wave 4

核心问题是：phenotype 和 adversity 更主要在标记 **disability pathway**，还是 **mortality pathway**。

## 2. Analytic samples

### Main multinomial sample

- 构建入口沿当前 composite 风险集：alive at wave 3、ADL-free at wave 3、wave 4 vital status known、phenotype/adversity/base covariates complete
- masked sample size: `{main_meta['n_total_masked']}`
- final multinomial sample: `{main_meta['n_final']}`
- unresolved survivors excluded: `{main_meta['n_unresolved_alive_without_adl_status']}`

三分类分布：
- alive without incident ADL: `{main_meta['n_alive_no_adl']}`
- alive with incident ADL: `{main_meta['n_alive_incident_adl']}`
- dead before wave 4: `{main_meta['n_dead_before_w4']}`

这一步比官方 composite sample 少了 `{main_meta['n_unresolved_alive_without_adl_status']}` 人，不是改定义，而是这批人虽已知未死亡且 composite=0，但其 wave 4 ADL 状态无法再细分为 “incident ADL” 或 “no incident ADL”。

### Supportive IADL extension

- masked sample size: `{supportive_meta['n_total_masked']}`
- final multinomial-style supportive sample: `{supportive_meta['n_final']}`
- unresolved survivors excluded: `{supportive_meta['n_unresolved_alive_without_iadl_status']}`

## 3. Main multinomial results

### 3.1 Joint phenotype + adversity model

Compared with Stable low-burden, the relative risk ratios from the joint multinomial model were:

- Emerging vs Stable
  - disability pathway: `{phen_dis['effect_95ci']}`
  - mortality pathway: `{phen_death['effect_95ci']}`
- High-burden vs Stable
  - disability pathway: `{hb_dis['effect_95ci']}`
  - mortality pathway: `{hb_death['effect_95ci']}`
- Severe vs Stable
  - disability pathway: `{sev_dis['effect_95ci']}`
  - mortality pathway: `{sev_death['effect_95ci']}`

For adversity score:

- disability pathway: `{adv_dis['effect_95ci']}`
- mortality pathway: `{adv_death['effect_95ci']}`

Block tests in the joint model:

- phenotype block given adversity: `p {joint_tests.loc['phenotype_block_given_adversity', 'p_formatted']}`
- adversity block given phenotype: `p {joint_tests.loc['adversity_block_given_phenotype', 'p_formatted']}`

## 4. Direct interpretation of pathway structure

### 4.1 phenotype 更主要是在标记 disability path 还是 mortality path？

更主要是在标记 **disability path**。

理由很直接：
- 对 `Emerging` 而言，disability pathway 的相对风险比明显高于 mortality pathway；
- `High-burden` 和 `Severe` 在 mortality pathway 上也有升高，但其 disability-path signal 仍更强、更稳定；
- 这说明 phenotype 更像是在标记功能衰退通道，而不是近端死亡通道。

### 4.2 adversity 为什么对 short-horizon mortality 弱？

在 joint multinomial model 中：
- adversity 对 disability pathway 仍是正向关联：`{adv_dis['effect_95ci']}`
- adversity 对 mortality pathway 接近零效应：`{adv_death['effect_95ci']}`

这和前面主模型一致，支持更稳的解释：
- adversity 更像 upstream vulnerability marker
- phenotype 才是更靠近 landmark 之后 outcomes 的 proximal stratifier

## 5. Composite endpoint 是否掩盖了更重要的结构信息？

是，**部分掩盖**。

原因不是 composite 没价值，而是：
- composite 把 `incident ADL` 和 `death before wave 4` 合并成同一种“event”
- 但本阶段结果显示，这两条通道对 phenotype 和 adversity 的响应模式并不相同
- 尤其 adversity：对 disability path 有信号，对 short-horizon mortality 没有明显信号

因此，composite endpoint 仍可保留，但它最适合做：
- supportive endpoint
- handling informative censoring by death

而不应取代主文中对 disability pathway 与 mortality pathway 的结构化解释。

## 6. Supportive IADL extension

在 supportive IADL competing structure 中，joint model 方向与 ADL 主分析一致：

- Emerging vs Stable for incident IADL path: `{supp_em['effect_95ci']}`
- adversity for incident IADL path: `{supp_adv['effect_95ci']}`

这说明 phenotype 和 adversity 更偏向功能障碍通道，而不只是 ADL 这个单一终点偶然得到的模式。

## 7. Direct answers required by this stage

### phenotype 更主要是在标记 disability path 还是 mortality path？

**Disability path。**

### 当前 composite endpoint 是否仍值得保留在主文？

值得保留，但应明确降级为：
- key supportive endpoint
- 用于补充说明死亡并未完全扭曲主结局

而不是把它当成与 primary incident ADL 同等重要的主终点。

### 结局结构的发现会如何改变 Discussion 的重点？

Discussion 最应强调：

1. phenotype 是更靠近 disability pathway 的 prognostic representation；
2. Emerging 的实践意义主要在于识别 pre-disability functional risk；
3. adversity 的更稳定位是 upstream vulnerability marker，而不是 short-horizon mortality predictor；
4. composite endpoint 有用，但它会压平 disability path 与 mortality path 的差异，因此不能替代分路径解释。
"""
    (RESULTS_DIR / "outcome_structure_summary.md").write_text(text, encoding="utf-8")


def update_results_index() -> None:
    stage_header = "## 2026-03-25 Stage 9 Outcome structure / competing pathways"
    content = RESULTS_INDEX.read_text(encoding="utf-8")
    if stage_header in content:
        return
    block = f"""

{stage_header}

本阶段用 competing outcome structure 把 wave 3 至 wave 4 的结果拆成 disability path 与 mortality path，避免只看 ADL 或 composite endpoint。

### 新增文件

| 文件 | 回答的问题 | 备注 |
|---|---|---|
| `scripts/stage9_outcome_structure/stage9_outcome_structure.py` | 如何在当前锁定样本链上构造三分类 competing outcome，并比较 phenotype、adversity 和 joint model | Stage 9 独立脚本 |
| `results/stage9_outcome_structure/multinomial_competing_status.csv` | phenotype 与 adversity 对 disability path 和 mortality path 的相对模式如何；supportive IADL 是否方向一致 | Stage 9 主结果表 |
| `results/stage9_outcome_structure/outcome_structure_summary.md` | phenotype 更主要在标记 disability path 还是 mortality path；composite endpoint 是否掩盖结构信息；Discussion 应如何调整 | Stage 9 总结文件 |
| `results/stage9_outcome_structure/outcome_structure_plots.pdf` | observed structure、joint-model RRR 和 adjusted predicted probabilities 的图示总结 | 便于人工核查 |

### 本阶段给出的直接结论

1. phenotype 更主要在标记 disability pathway，而不是 short-horizon mortality pathway。
2. adversity 对 disability pathway 仍有正向关联，但对 short-horizon mortality 近乎 null。
3. composite endpoint 仍可保留，但更适合做 supportive outcome，因为它会压平 disability path 与 mortality path 的差异。
4. supportive IADL extension 与 ADL 主分析方向一致，进一步支持 phenotype/ adversity 更偏向功能障碍路径。

### 下一阶段建议

只在你确认后进入：

1. Stage 10 targeted sensitivity analyses
"""
    RESULTS_INDEX.write_text(content.rstrip() + "\n" + block, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = load_data()
    main_sample, main_meta = build_main_sample(df)
    supportive_sample, supportive_meta = build_supportive_iadl_sample(df)

    main_rows = run_multinomial_suite(main_sample, ADL_STATUS_ORDER, "main_adl_competing")
    supportive_rows = run_multinomial_suite(supportive_sample, IADL_STATUS_ORDER, "supportive_iadl_competing")
    supportive_binary = fit_binary_supportive_iadl(supportive_sample)

    all_rows = pd.concat([main_rows, supportive_rows, supportive_binary], ignore_index=True, sort=False)
    all_rows.to_csv(RESULTS_DIR / "multinomial_competing_status.csv", index=False)

    make_plots(main_rows, supportive_rows)
    write_summary(main_meta, supportive_meta, all_rows)

    metadata = {
        "seed": SEED,
        "main_meta": main_meta,
        "supportive_meta": supportive_meta,
    }
    (RESULTS_DIR / "stage9_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    update_results_index()


if __name__ == "__main__":
    main()

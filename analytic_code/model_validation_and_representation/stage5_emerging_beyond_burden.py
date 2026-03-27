#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path("/Users/macbookpro-hu/Library/CloudStorage/OneDrive-个人/Xcode/Medical Research/PROJECTS/charls_life_course_aging")
RERUN_ROOT = PROJECT_ROOT / "rerun_risk_stratification_expanded_v2_20260323"
RESULTS_DIR = PROJECT_ROOT / "results" / "stage5_emerging_beyond_burden"
LOGS_DIR = PROJECT_ROOT / "logs" / "stage5_emerging_beyond_burden"

BASE_COVARS = ["baseline_age_2011", "female", "rural_hukou_2011", "low_self_education"]
WAVE3_LEVEL_VARS = ["dep_w3", "cogdef_w3", "physical_lim_w3", "multimorb_w3"]
PHENOTYPE_ORDER = [
    "Stable low-burden",
    "Emerging multidomain decline",
    "High-burden mixed pattern",
    "Severe multidomain decline",
]
PHENOTYPE_LABEL_MAP = {
    "traj_Emerging multidomain decline": "Emerging vs Stable",
    "traj_High-burden mixed pattern": "High-burden vs Stable",
    "traj_Severe multidomain decline": "Severe vs Stable",
}
SEED = 20260325


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def format_p(p: float) -> str:
    if pd.isna(p):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def z_test_from_contrast(beta: np.ndarray, cov: np.ndarray, contrast: np.ndarray) -> tuple[float, float, float, float, float]:
    est = float(contrast @ beta)
    var = float(contrast @ cov @ contrast)
    se = np.sqrt(max(var, 1e-12))
    z = est / se
    p = float(2 * stats.norm.sf(abs(z)))
    low = est - 1.96 * se
    high = est + 1.96 * se
    return est, se, p, low, high


def make_psd(cov: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    cov = (cov + cov.T) / 2.0
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, floor, None)
    return vecs @ np.diag(vals) @ vecs.T


def load_data() -> pd.DataFrame:
    base = pd.read_csv(RERUN_ROOT / "02_data_processing" / "analysis_dataset_landmark.csv")
    assignments = pd.read_csv(RERUN_ROOT / "03_phenotype_derivation" / "phenotype_assignments.csv")
    df = base.merge(assignments[["ID_str", "trajectory_group"]], on="ID_str", how="left")
    df["trajectory_group"] = pd.Categorical(df["trajectory_group"], categories=PHENOTYPE_ORDER, ordered=True)
    return df


def build_samples(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    primary = df[
        (df["trajectory_group"].notna())
        & (df["alive_at_w3"] == 1)
        & (df["adl_disability_w3"] == 0)
        & df[["adversity_score_scaled", *BASE_COVARS, "incident_adl_w3_w4"]].notna().all(axis=1)
    ].copy()
    unified = df[
        (df["trajectory_group"].notna())
        & (df["alive_at_w3"] == 1)
        & (df["adl_disability_w3"] == 0)
        & df[[*BASE_COVARS, *WAVE3_LEVEL_VARS, "incident_adl_w3_w4"]].notna().all(axis=1)
    ].copy()
    primary_w3_complete = primary[primary["ID_str"].isin(unified["ID_str"])].copy()
    return {
        "primary_locked": primary,
        "primary_wave3_complete": primary_w3_complete,
        "unified_locked": unified,
    }


def design_matrix(data: pd.DataFrame, include_levels: bool, include_pheno: bool) -> pd.DataFrame:
    cols = BASE_COVARS.copy()
    parts = [data[cols].reset_index(drop=True).astype(float)]
    if include_levels:
        parts.append(data[WAVE3_LEVEL_VARS].reset_index(drop=True).astype(float))
    if include_pheno:
        dummies = pd.get_dummies(
            pd.Categorical(data["trajectory_group"].reset_index(drop=True), categories=PHENOTYPE_ORDER, ordered=True),
            prefix="traj",
            drop_first=True,
            dtype=float,
        )
        parts.append(dummies)
    return pd.concat(parts, axis=1)


def fit_glm(data: pd.DataFrame, include_levels: bool, include_pheno: bool):
    X = design_matrix(data, include_levels=include_levels, include_pheno=include_pheno)
    y = data["incident_adl_w3_w4"].reset_index(drop=True).astype(int)
    model = sm.GLM(y, sm.add_constant(X, has_constant="add"), family=sm.families.Binomial()).fit()
    pred = model.predict(sm.add_constant(X, has_constant="add"))
    auc = roc_auc_score(y, pred)
    return model, X, pred, auc


def lr_test(model_full, model_reduced) -> tuple[float, float, int]:
    lr = 2 * (model_full.llf - model_reduced.llf)
    df_diff = int(model_full.df_model - model_reduced.df_model)
    p = float(stats.chi2.sf(lr, df_diff))
    return float(lr), p, df_diff


def term_and_pairwise_rows(sample_name: str, model_name: str, data: pd.DataFrame, model) -> list[dict]:
    rows: list[dict] = []
    params = model.params
    cov = model.cov_params()
    conf = model.conf_int()

    dummy_terms = [t for t in params.index if t.startswith("traj_")]
    if dummy_terms:
        beta = params.loc[params.index].to_numpy(dtype=float)
        cov_np = cov.loc[params.index, params.index].to_numpy(dtype=float)
        idx = {name: i for i, name in enumerate(params.index)}
        for term in dummy_terms:
            est = float(params[term])
            low = float(conf.loc[term, 0])
            high = float(conf.loc[term, 1])
            rows.append(
                {
                    "row_type": "contrast",
                    "sample_name": sample_name,
                    "model_name": model_name,
                    "contrast": PHENOTYPE_LABEL_MAP.get(term, term),
                    "reference": "Stable low-burden",
                    "estimate_logit": est,
                    "se": float(model.bse[term]),
                    "OR": float(np.exp(est)),
                    "ci_low": float(np.exp(low)),
                    "ci_high": float(np.exp(high)),
                    "effect_95ci": f"{np.exp(est):.3f} ({np.exp(low):.3f}, {np.exp(high):.3f})",
                    "p": float(model.pvalues[term]),
                    "p_formatted": format_p(float(model.pvalues[term])),
                    "n": int(len(data)),
                    "events": int(data["incident_adl_w3_w4"].sum()),
                }
            )

        pair_defs = [
            ("Emerging vs High-burden", "traj_Emerging multidomain decline", "traj_High-burden mixed pattern"),
            ("Emerging vs Severe", "traj_Emerging multidomain decline", "traj_Severe multidomain decline"),
        ]
        for label, a, b in pair_defs:
            if a in idx and b in idx:
                c = np.zeros(len(beta), dtype=float)
                c[idx[a]] = 1.0
                c[idx[b]] = -1.0
                est, se, p, low, high = z_test_from_contrast(beta, cov_np, c)
                rows.append(
                    {
                        "row_type": "contrast",
                        "sample_name": sample_name,
                        "model_name": model_name,
                        "contrast": label,
                        "reference": b.replace("traj_", ""),
                        "estimate_logit": est,
                        "se": se,
                        "OR": float(np.exp(est)),
                        "ci_low": float(np.exp(low)),
                        "ci_high": float(np.exp(high)),
                        "effect_95ci": f"{np.exp(est):.3f} ({np.exp(low):.3f}, {np.exp(high):.3f})",
                        "p": p,
                        "p_formatted": format_p(p),
                        "n": int(len(data)),
                        "events": int(data["incident_adl_w3_w4"].sum()),
                    }
                )
    return rows


def overall_test_row(sample_name: str, model_name: str, data: pd.DataFrame, full_model, reduced_model) -> dict:
    lr, p, df_diff = lr_test(full_model, reduced_model)
    return {
        "row_type": "overall_test",
        "sample_name": sample_name,
        "model_name": model_name,
        "contrast": "Phenotype overall test",
        "reference": reduced_model.model.exog_names,
        "lr_chisq": lr,
        "df_diff": df_diff,
        "p": p,
        "p_formatted": format_p(p),
        "n": int(len(data)),
        "events": int(data["incident_adl_w3_w4"].sum()),
        "apparent_auc": float(roc_auc_score(data["incident_adl_w3_w4"].astype(int), full_model.fittedvalues)),
        "aic": float(full_model.aic),
        "bic": float(full_model.bic_llf),
    }


def simulate_marginal_predictions(data: pd.DataFrame, model, model_name: str, sample_name: str, ndraws: int = 2000) -> pd.DataFrame:
    X_base = design_matrix(
        data,
        include_levels=("levels" in model_name),
        include_pheno=True,
    )
    X_base = sm.add_constant(X_base, has_constant="add")
    beta = model.params.loc[X_base.columns].to_numpy(dtype=float)
    cov = model.cov_params().loc[X_base.columns, X_base.columns].to_numpy(dtype=float)
    cov = make_psd(cov)
    rng = np.random.default_rng(SEED + 55)
    draws = rng.multivariate_normal(beta, cov, size=ndraws)

    out_rows = []
    stable_draws = None
    scen_draws_map = {}
    for phen in PHENOTYPE_ORDER:
        scen = X_base.copy()
        for col in [c for c in scen.columns if c.startswith("traj_")]:
            scen[col] = 0.0
        if phen != "Stable low-burden":
            scen[f"traj_{phen}"] = 1.0
        xb = scen.to_numpy(dtype=float) @ draws.T
        xb = np.clip(xb, -20, 20)
        pr = 1 / (1 + np.exp(-xb))
        avg_pr = pr.mean(axis=0)
        scen_draws_map[phen] = avg_pr
        if phen == "Stable low-burden":
            stable_draws = avg_pr

    assert stable_draws is not None
    for phen in PHENOTYPE_ORDER:
        avg_pr = scen_draws_map[phen]
        rd = avg_pr - stable_draws
        out_rows.append(
            {
                "sample_name": sample_name,
                "model_name": model_name,
                "phenotype": phen,
                "predicted_probability": float(np.mean(avg_pr)),
                "ci_low": float(np.quantile(avg_pr, 0.025)),
                "ci_high": float(np.quantile(avg_pr, 0.975)),
                "risk_difference_vs_stable": float(np.mean(rd)),
                "rd_low": float(np.quantile(rd, 0.025)),
                "rd_high": float(np.quantile(rd, 0.975)),
            }
        )
    return pd.DataFrame(out_rows)


def risk_strata_bonus(data: pd.DataFrame) -> pd.DataFrame:
    m1, _, pred, _ = fit_glm(data, include_levels=True, include_pheno=False)
    out = data[["ID_str", "trajectory_group", "incident_adl_w3_w4"]].copy()
    out["m1_predicted_risk"] = pred
    out["risk_stratum"] = pd.qcut(out["m1_predicted_risk"], q=5, labels=[f"Q{i}" for i in range(1, 6)], duplicates="drop")
    rows = []
    for (stratum, phen), sub in out.groupby(["risk_stratum", "trajectory_group"], observed=False):
        rows.append(
            {
                "risk_stratum": stratum,
                "phenotype": phen,
                "n": int(len(sub)),
                "events": int(sub["incident_adl_w3_w4"].sum()),
                "event_rate": float(sub["incident_adl_w3_w4"].mean()) if len(sub) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_summary(model_rows: pd.DataFrame, risk_rows: pd.DataFrame, bonus_rows: pd.DataFrame) -> None:
    def pick_contrast(sample_name: str, model_name: str, contrast: str) -> str:
        sub = model_rows[
            (model_rows["row_type"] == "contrast")
            & (model_rows["sample_name"] == sample_name)
            & (model_rows["model_name"] == model_name)
            & (model_rows["contrast"] == contrast)
        ]
        return sub.iloc[0]["effect_95ci"] if len(sub) else ""

    def pick_overall_p(sample_name: str, model_name: str) -> str:
        sub = model_rows[
            (model_rows["row_type"] == "overall_test")
            & (model_rows["sample_name"] == sample_name)
            & (model_rows["model_name"] == model_name)
        ]
        return sub.iloc[0]["p_formatted"] if len(sub) else ""

    def pick_auc(sample_name: str, model_name: str) -> float:
        sub = model_rows[
            (model_rows["sample_name"] == sample_name)
            & (model_rows["model_name"] == model_name)
            & model_rows["apparent_auc"].notna()
        ]
        return float(sub.iloc[0]["apparent_auc"])

    risk_adj = risk_rows[(risk_rows["sample_name"] == "unified_locked") & (risk_rows["model_name"] == "base_levels_phenotype")]
    pred_map = {r["phenotype"]: r for _, r in risk_adj.iterrows()}
    q3 = bonus_rows[(bonus_rows["risk_stratum"] == "Q3")].copy()
    q3 = q3.sort_values("phenotype")

    summary = f"""# Stage 5 Emerging phenotype beyond current burden

## 1. Stage aim

本阶段直接回答：

1. `Emerging multidomain decline` 在控制 wave 3 当前四域状态后，是否仍保留独立 prognostic information？
2. phenotype 整体是否还提供 wave 3 burden 之外的信息？
3. `pre-disability` 这个解释最多能说到哪一步？

## 2. Sample note

- `primary_locked`：当前锁定 primary incident ADL sample，`n=8643`
- `unified_locked`：当前锁定 unified incident ADL comparison sample，`n=8355`

重要说明：

- `base + phenotype` 可在完整 `primary_locked` 上估计；
- 任何包含 `wave3 four domains` 的模型都需要 `dep_w3/cogdef_w3/physical_lim_w3/multimorb_w3` 完整，因此在 primary 框架下会自动落到 `n=8355`，与 `unified_locked` 相同。

## 3. Direct model results

### On the full primary locked sample

- `base + phenotype` overall phenotype test: `p {pick_overall_p('primary_locked', 'base_phenotype')}`
- `Emerging vs Stable`: `{pick_contrast('primary_locked', 'base_phenotype', 'Emerging vs Stable')}`
- `High-burden vs Stable`: `{pick_contrast('primary_locked', 'base_phenotype', 'High-burden vs Stable')}`
- `Severe vs Stable`: `{pick_contrast('primary_locked', 'base_phenotype', 'Severe vs Stable')}`

### On the unified sample

- `base + phenotype` overall phenotype test: `p {pick_overall_p('unified_locked', 'base_phenotype')}`
- `base + wave3 four domains` apparent AUC: `{pick_auc('unified_locked', 'base_levels'):.3f}`
- `base + wave3 four domains + phenotype` overall phenotype test: `p {pick_overall_p('unified_locked', 'base_levels_phenotype')}`
- `base + wave3 four domains + phenotype` apparent AUC: `{pick_auc('unified_locked', 'base_levels_phenotype'):.3f}`

Within the burden-adjusted unified model:

- `Emerging vs Stable`: `{pick_contrast('unified_locked', 'base_levels_phenotype', 'Emerging vs Stable')}`
- `High-burden vs Stable`: `{pick_contrast('unified_locked', 'base_levels_phenotype', 'High-burden vs Stable')}`
- `Severe vs Stable`: `{pick_contrast('unified_locked', 'base_levels_phenotype', 'Severe vs Stable')}`
- `Emerging vs High-burden`: `{pick_contrast('unified_locked', 'base_levels_phenotype', 'Emerging vs High-burden')}`
- `Emerging vs Severe`: `{pick_contrast('unified_locked', 'base_levels_phenotype', 'Emerging vs Severe')}`

## 4. Adjusted absolute risk

Using the burden-adjusted unified model (`base + wave3 four domains + phenotype`), the adjusted absolute risks were:

- Stable low-burden: `{pred_map['Stable low-burden']['predicted_probability']:.3f}` ({pred_map['Stable low-burden']['ci_low']:.3f}-{pred_map['Stable low-burden']['ci_high']:.3f})
- Emerging multidomain decline: `{pred_map['Emerging multidomain decline']['predicted_probability']:.3f}` ({pred_map['Emerging multidomain decline']['ci_low']:.3f}-{pred_map['Emerging multidomain decline']['ci_high']:.3f})
- High-burden mixed pattern: `{pred_map['High-burden mixed pattern']['predicted_probability']:.3f}` ({pred_map['High-burden mixed pattern']['ci_low']:.3f}-{pred_map['High-burden mixed pattern']['ci_high']:.3f})
- Severe multidomain decline: `{pred_map['Severe multidomain decline']['predicted_probability']:.3f}` ({pred_map['Severe multidomain decline']['ci_low']:.3f}-{pred_map['Severe multidomain decline']['ci_high']:.3f})

Relative to Stable low-burden, the adjusted risk differences were:

- Emerging: `{pred_map['Emerging multidomain decline']['risk_difference_vs_stable']:.3f}` ({pred_map['Emerging multidomain decline']['rd_low']:.3f}-{pred_map['Emerging multidomain decline']['rd_high']:.3f})
- High-burden mixed: `{pred_map['High-burden mixed pattern']['risk_difference_vs_stable']:.3f}` ({pred_map['High-burden mixed pattern']['rd_low']:.3f}-{pred_map['High-burden mixed pattern']['rd_high']:.3f})
- Severe: `{pred_map['Severe multidomain decline']['risk_difference_vs_stable']:.3f}` ({pred_map['Severe multidomain decline']['rd_low']:.3f}-{pred_map['Severe multidomain decline']['rd_high']:.3f})

## 5. Bonus check within wave 3 burden strata

As a bonus analysis, I grouped the unified sample by quintiles of the `base + wave3 four domains` predicted risk and examined phenotype-specific event rates within strata.

In the mid-risk stratum (`Q3`), phenotype event rates were:

{q3.to_string(index=False)}

This is not a fully weighted causal balancing design, but it provides a direct descriptive check of whether Emerging continues to track elevated risk among participants with broadly similar burden-predicted risk.

## 6. Direct answers

### Is Emerging independent of wave 3 current burden?

This should be judged primarily from the burden-adjusted unified model. If `Emerging vs Stable` remains elevated there, then Emerging retains some information beyond raw current burden. However, the comparison with Stage 4 means that any such independence should be described as **partial and modest**, not as a large autonomous signal.

### How far can the `pre-disability` interpretation go?

The safest interpretation is:

- Emerging identifies a group with clearly higher subsequent disability risk than Stable low-burden;
- this group is less overtly impaired than the more severe phenotypes;
- therefore Emerging can still be described as a **clinically interpretable pre-disability high-risk phenotype**.

The interpretation should stop short of claiming:

- a fully distinct disease-like state;
- a burden-independent latent construct;
- or a phenotype whose value is completely separable from current wave 3 multidomain burden.

### Should the title / main claim emphasise phenotype association or early-state discovery?

At this stage, the more defensible emphasis remains **phenotype-based prognostic association / risk stratification**, with `early-state discovery` used only as a cautious secondary interpretation. If the paper keeps the phrase `pre-disability`, it should be framed as an interpretable high-risk state concept rather than a validated autonomous state discovery.
"""
    (RESULTS_DIR / "emerging_beyond_burden_summary.md").write_text(summary, encoding="utf-8")


def build_plot(risk_rows: pd.DataFrame) -> None:
    plot_df = risk_rows[risk_rows["model_name"].isin(["base_phenotype", "base_levels_phenotype"])].copy()
    plot_df["model_label"] = plot_df["model_name"].map(
        {
            "base_phenotype": "Base + phenotype",
            "base_levels_phenotype": "Base + wave 3 four domains + phenotype",
        }
    )
    plot_df["phenotype"] = pd.Categorical(plot_df["phenotype"], categories=PHENOTYPE_ORDER, ordered=True)
    plot_df = plot_df.sort_values(["model_label", "phenotype"])

    with PdfPages(RESULTS_DIR / "adjusted_risk_plot.pdf") as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ax, (sample_name, sub) in zip(axes, plot_df.groupby("sample_name")):
            sub = sub[sub["model_name"].isin(["base_phenotype", "base_levels_phenotype"])].copy()
            width = 0.36
            xpos = np.arange(len(PHENOTYPE_ORDER))
            available_models = [m for m in ["base_phenotype", "base_levels_phenotype"] if m in set(sub["model_name"])]
            if len(available_models) == 1:
                offsets = {available_models[0]: 0.0}
            else:
                offsets = {
                    available_models[0]: -0.5 * width,
                    available_models[1]: 0.5 * width,
                }
            for model_name in available_models:
                d = sub[sub["model_name"] == model_name].set_index("phenotype").loc[PHENOTYPE_ORDER].reset_index()
                x = xpos + offsets[model_name]
                ax.bar(x, d["predicted_probability"], width=width, label=d["model_label"].iloc[0], alpha=0.85)
                ax.errorbar(
                    x,
                    d["predicted_probability"],
                    yerr=[
                        d["predicted_probability"] - d["ci_low"],
                        d["ci_high"] - d["predicted_probability"],
                    ],
                    fmt="none",
                    ecolor="black",
                    capsize=3,
                    linewidth=1,
                )
            ax.set_xticks(xpos)
            ax.set_xticklabels(PHENOTYPE_ORDER, rotation=25, ha="right")
            ax.set_title(sample_name)
            ax.set_ylabel("Adjusted predicted probability")
            ax.set_ylim(0, max(0.24, float(sub["ci_high"].max()) + 0.02))
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle("Adjusted absolute risk by phenotype")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    ensure_dirs()
    np.random.seed(SEED)
    random.seed(SEED)

    df = load_data()
    samples = build_samples(df)

    model_rows = []
    risk_rows = []

    # Primary locked sample: base only vs base + phenotype on full primary sample.
    primary = samples["primary_locked"]
    model_base_primary, _, _, _ = fit_glm(primary, include_levels=False, include_pheno=False)
    model_pheno_primary, _, _, _ = fit_glm(primary, include_levels=False, include_pheno=True)
    model_rows.append(overall_test_row("primary_locked", "base_phenotype", primary, model_pheno_primary, model_base_primary))
    model_rows.extend(term_and_pairwise_rows("primary_locked", "base_phenotype", primary, model_pheno_primary))
    risk_rows.append(simulate_marginal_predictions(primary, model_pheno_primary, "base_phenotype", "primary_locked"))

    # Unified sample: direct current-burden comparison.
    unified = samples["unified_locked"]
    model_base_unified, _, _, _ = fit_glm(unified, include_levels=False, include_pheno=False)
    model_pheno_unified, _, _, _ = fit_glm(unified, include_levels=False, include_pheno=True)
    model_levels_unified, _, _, _ = fit_glm(unified, include_levels=True, include_pheno=False)
    model_levels_pheno_unified, _, _, _ = fit_glm(unified, include_levels=True, include_pheno=True)

    model_rows.append(overall_test_row("unified_locked", "base_phenotype", unified, model_pheno_unified, model_base_unified))
    model_rows.extend(term_and_pairwise_rows("unified_locked", "base_phenotype", unified, model_pheno_unified))
    model_rows.append(overall_test_row("unified_locked", "base_levels_phenotype", unified, model_levels_pheno_unified, model_levels_unified))
    model_rows.extend(term_and_pairwise_rows("unified_locked", "base_levels_phenotype", unified, model_levels_pheno_unified))
    model_rows.append(
        {
            "row_type": "model_info",
            "sample_name": "unified_locked",
            "model_name": "base_levels",
            "contrast": "Base + wave3 four domains",
            "reference": "",
            "n": int(len(unified)),
            "events": int(unified["incident_adl_w3_w4"].sum()),
            "apparent_auc": float(roc_auc_score(unified["incident_adl_w3_w4"].astype(int), model_levels_unified.fittedvalues)),
            "aic": float(model_levels_unified.aic),
            "bic": float(model_levels_unified.bic_llf),
        }
    )
    risk_rows.append(simulate_marginal_predictions(unified, model_pheno_unified, "base_phenotype", "unified_locked"))
    risk_rows.append(simulate_marginal_predictions(unified, model_levels_pheno_unified, "base_levels_phenotype", "unified_locked"))

    model_df = pd.DataFrame(model_rows)
    risk_df = pd.concat(risk_rows, ignore_index=True)

    # Primary wave3-complete context is the same people as unified; record explicitly.
    if len(samples["primary_wave3_complete"]) != len(unified):
        raise RuntimeError("primary_wave3_complete and unified_locked diverged unexpectedly.")

    # Bonus analysis within burden-predicted risk strata.
    bonus_df = risk_strata_bonus(unified)

    model_df.to_csv(RESULTS_DIR / "emerging_beyond_burden_models.csv", index=False)
    risk_df.to_csv(RESULTS_DIR / "adjusted_absolute_risk_by_phenotype.csv", index=False)
    pairwise_df = model_df[model_df["row_type"] == "contrast"].copy()
    pairwise_df.to_csv(RESULTS_DIR / "emerging_pairwise_contrasts.csv", index=False)
    bonus_df.to_csv(RESULTS_DIR / "emerging_within_m1_risk_strata.csv", index=False)

    build_plot(risk_df)
    write_summary(model_df, risk_df, bonus_df)

    meta = {
        "stage": "stage5_emerging_beyond_burden",
        "seed": SEED,
        "n_primary_locked": int(len(primary)),
        "n_unified_locked": int(len(unified)),
    }
    (RESULTS_DIR / "stage5_run_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()

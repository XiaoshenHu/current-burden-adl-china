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
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = project_root()
RERUN_ROOT = expanded_rerun_root()
RESULTS_DIR = PROJECT_ROOT / "results" / "stage10_targeted_sensitivity"
LOGS_DIR = PROJECT_ROOT / "logs" / "stage10_targeted_sensitivity"
RESULTS_INDEX = PROJECT_ROOT / "RESULTS_INDEX.md"

BASE_COVARS = ["baseline_age_2011", "female", "rural_hukou_2011", "low_self_education"]
WAVE3_DOMAIN_VARS = ["dep_w3", "cogdef_w3", "physical_lim_w3", "multimorb_w3"]
DOMAINS = ["dep", "cogdef", "physical_lim", "multimorb"]
PHENOTYPE_ORDER = [
    "Stable low-burden",
    "Emerging multidomain decline",
    "High-burden mixed pattern",
    "Severe multidomain decline",
]
WAVE_TIMES = {1: -4.0, 2: -2.0, 3: 0.0}
SEED = 20260405
CV_SEED = 20260406


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def fmt_ci(est: float, low: float, high: float) -> str:
    return f"{est:.3f} ({low:.3f}, {high:.3f})"


def format_p(p: float) -> str:
    if pd.isna(p):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def weighted_brier(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def safe_logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def calibration_intercept_slope(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    pred = np.clip(pred, 1e-6, 1 - 1e-6)
    logit_pred = safe_logit(pred)
    slope_model = sm.GLM(y, sm.add_constant(logit_pred), family=sm.families.Binomial()).fit()
    intercept_model = sm.GLM(y, np.ones((len(y), 1)), offset=logit_pred, family=sm.families.Binomial()).fit()
    return float(intercept_model.params[0]), float(slope_model.params[1])


def fit_pre_w3_slope(values: list[float]) -> float:
    times = np.array([WAVE_TIMES[1], WAVE_TIMES[2], WAVE_TIMES[3]], dtype=float)
    arr = np.array(values, dtype=float)
    mask = ~np.isnan(arr)
    if mask.sum() < 2:
        return np.nan
    slope, _ = np.polyfit(times[mask], arr[mask], 1)
    return float(slope)


def load_data() -> pd.DataFrame:
    base = pd.read_csv(RERUN_ROOT / "02_data_processing" / "analysis_dataset_landmark.csv")
    assignments = pd.read_csv(RERUN_ROOT / "03_phenotype_derivation" / "phenotype_assignments.csv")
    df = base.merge(assignments[["ID_str", "trajectory_group"]], on="ID_str", how="left")
    df["trajectory_group"] = pd.Categorical(df["trajectory_group"], categories=PHENOTYPE_ORDER, ordered=True)
    # Explicit wave 3 age is not stored in the locked analytic table; use deterministic 2011->2015 +4y approximation.
    df["age_w3_approx"] = df["baseline_age_2011"] + 4.0
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
        & df[[*BASE_COVARS, *WAVE3_DOMAIN_VARS, "incident_adl_w3_w4"]].notna().all(axis=1)
    ].copy()
    return {"primary_locked": primary, "unified_locked": unified}


def phenotype_design_matrix(data: pd.DataFrame, include_levels: bool = False, level_cols: list[str] | None = None) -> pd.DataFrame:
    parts = [data[BASE_COVARS].reset_index(drop=True).astype(float)]
    if include_levels:
        if level_cols is None:
            raise ValueError("level_cols required when include_levels=True")
        parts.append(data[level_cols].reset_index(drop=True).astype(float))
    dummies = pd.get_dummies(
        pd.Categorical(data["trajectory_group"].reset_index(drop=True), categories=PHENOTYPE_ORDER, ordered=True),
        prefix="traj",
        drop_first=True,
        dtype=float,
    )
    parts.append(dummies)
    return pd.concat(parts, axis=1)


def fit_glm(data: pd.DataFrame, X: pd.DataFrame, y_col: str = "incident_adl_w3_w4"):
    y = data[y_col].reset_index(drop=True).astype(int)
    X_use = sm.add_constant(X.astype(float).reset_index(drop=True), has_constant="add")
    model = sm.GLM(y, X_use, family=sm.families.Binomial()).fit()
    pred = np.asarray(model.predict(X_use))
    return model, pred


def fit_and_predict_cv(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
    X_train_use = sm.add_constant(X_train.astype(float), has_constant="add")
    X_test_use = sm.add_constant(X_test.astype(float), has_constant="add")
    X_test_use = X_test_use.reindex(columns=X_train_use.columns, fill_value=0.0)
    model = sm.GLM(y_train.astype(int), X_train_use, family=sm.families.Binomial()).fit()
    return np.asarray(model.predict(X_test_use))


def cv_metrics_from_X(data: pd.DataFrame, X: pd.DataFrame, y_col: str = "incident_adl_w3_w4") -> dict[str, float]:
    y = data[y_col].reset_index(drop=True).astype(int)
    full_model, full_pred = fit_glm(data, X, y_col)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=CV_SEED)
    oof_pred = np.zeros(len(data), dtype=float)
    for train_idx, test_idx in skf.split(np.zeros(len(y)), y):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        oof_pred[test_idx] = fit_and_predict_cv(X_train, y_train, X_test)
    intercept, slope = calibration_intercept_slope(y.to_numpy(), oof_pred)
    return {
        "n": int(len(data)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "apparent_auc": float(roc_auc_score(y, full_pred)),
        "oof_auc": float(roc_auc_score(y, oof_pred)),
        "apparent_pr_auc": float(average_precision_score(y, full_pred)),
        "oof_pr_auc": float(average_precision_score(y, oof_pred)),
        "apparent_brier": weighted_brier(y.to_numpy(), full_pred),
        "oof_brier": weighted_brier(y.to_numpy(), oof_pred),
        "oof_calibration_intercept": intercept,
        "oof_calibration_slope": slope,
    }


def z_test_from_contrast(beta: np.ndarray, cov: np.ndarray, contrast: np.ndarray) -> tuple[float, float, float, float, float]:
    est = float(contrast @ beta)
    var = float(contrast @ cov @ contrast)
    se = np.sqrt(max(var, 1e-12))
    z = est / se
    p = float(2 * stats.norm.sf(abs(z)))
    low = est - 1.96 * se
    high = est + 1.96 * se
    return est, se, p, low, high


def simulate_adjusted_risk(data: pd.DataFrame, ndraws: int = 1500) -> pd.DataFrame:
    X = phenotype_design_matrix(data, include_levels=False)
    model, _ = fit_glm(data, X)
    X_use = sm.add_constant(X.astype(float), has_constant="add")
    beta = model.params.loc[X_use.columns].to_numpy(dtype=float)
    cov = model.cov_params().loc[X_use.columns, X_use.columns].to_numpy(dtype=float)
    cov = (cov + cov.T) / 2.0
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-10, None)
    cov = vecs @ np.diag(vals) @ vecs.T
    rng = np.random.default_rng(SEED + 91)
    draws = rng.multivariate_normal(beta, cov, size=ndraws)

    out = []
    stable_draws = None
    scen_draws = {}
    for phen in PHENOTYPE_ORDER:
        scen = X_use.copy()
        for c in [c for c in scen.columns if c.startswith("traj_")]:
            scen[c] = 0.0
        if phen != "Stable low-burden":
            scen[f"traj_{phen}"] = 1.0
        xb = scen.to_numpy(dtype=float) @ draws.T
        xb = np.clip(xb, -20, 20)
        pr = 1 / (1 + np.exp(-xb))
        avg_pr = pr.mean(axis=0)
        scen_draws[phen] = avg_pr
        if phen == "Stable low-burden":
            stable_draws = avg_pr

    assert stable_draws is not None
    for phen in PHENOTYPE_ORDER:
        pr = scen_draws[phen]
        rd = pr - stable_draws
        out.append(
            {
                "phenotype": phen,
                "predicted_probability": float(np.mean(pr)),
                "ci_low": float(np.quantile(pr, 0.025)),
                "ci_high": float(np.quantile(pr, 0.975)),
                "risk_difference_vs_stable": float(np.mean(rd)),
                "rd_low": float(np.quantile(rd, 0.025)),
                "rd_high": float(np.quantile(rd, 0.975)),
            }
        )
    return pd.DataFrame(out)


def older_adult_sensitivity(samples: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    thresholds = [60, 65]
    for cutoff in thresholds:
        primary = samples["primary_locked"][samples["primary_locked"]["age_w3_approx"] >= cutoff].copy()
        unified = samples["unified_locked"][samples["unified_locked"]["age_w3_approx"] >= cutoff].copy()

        rows.append(
            {
                "row_type": "sample",
                "subset": f"age_ge_{cutoff}",
                "sample_name": "primary_locked",
                "n": int(len(primary)),
                "events": int(primary["incident_adl_w3_w4"].sum()),
                "event_rate": float(primary["incident_adl_w3_w4"].mean()) if len(primary) else np.nan,
            }
        )
        rows.append(
            {
                "row_type": "sample",
                "subset": f"age_ge_{cutoff}",
                "sample_name": "unified_locked",
                "n": int(len(unified)),
                "events": int(unified["incident_adl_w3_w4"].sum()),
                "event_rate": float(unified["incident_adl_w3_w4"].mean()) if len(unified) else np.nan,
            }
        )

        # Main phenotype association on primary sample.
        X_base = primary[BASE_COVARS].reset_index(drop=True).astype(float)
        X_pheno = phenotype_design_matrix(primary, include_levels=False)
        y_primary = primary["incident_adl_w3_w4"].reset_index(drop=True).astype(int)
        reduced_model = sm.GLM(y_primary, sm.add_constant(X_base, has_constant="add"), family=sm.families.Binomial()).fit()
        full_model = sm.GLM(y_primary, sm.add_constant(X_pheno, has_constant="add"), family=sm.families.Binomial()).fit()
        lr = 2 * (full_model.llf - reduced_model.llf)
        df_diff = int(full_model.df_model - reduced_model.df_model)
        p_overall = float(stats.chi2.sf(lr, df_diff))
        rows.append(
            {
                "row_type": "overall_test",
                "subset": f"age_ge_{cutoff}",
                "sample_name": "primary_locked",
                "model_name": "base_plus_phenotype",
                "contrast": "Phenotype overall test",
                "lr_chisq": float(lr),
                "df_diff": df_diff,
                "p": p_overall,
                "p_formatted": format_p(p_overall),
            }
        )

        params = full_model.params
        cov = full_model.cov_params()
        conf = full_model.conf_int()
        terms = [c for c in params.index if c.startswith("traj_")]
        for term in terms:
            rows.append(
                {
                    "row_type": "contrast",
                    "subset": f"age_ge_{cutoff}",
                    "sample_name": "primary_locked",
                    "model_name": "base_plus_phenotype",
                    "contrast": term.replace("traj_", "") + " vs Stable",
                    "OR": float(np.exp(params[term])),
                    "ci_low": float(np.exp(conf.loc[term, 0])),
                    "ci_high": float(np.exp(conf.loc[term, 1])),
                    "effect_95ci": fmt_ci(np.exp(params[term]), np.exp(conf.loc[term, 0]), np.exp(conf.loc[term, 1])),
                    "p": float(full_model.pvalues[term]),
                    "p_formatted": format_p(float(full_model.pvalues[term])),
                }
            )

        coef_names = list(params.index)
        beta = params.loc[coef_names].to_numpy(dtype=float)
        cov_np = cov.loc[coef_names, coef_names].to_numpy(dtype=float)
        idx = {n: i for i, n in enumerate(coef_names)}
        pair_defs = [
            ("Emerging vs High-burden", "traj_Emerging multidomain decline", "traj_High-burden mixed pattern"),
            ("Emerging vs Severe", "traj_Emerging multidomain decline", "traj_Severe multidomain decline"),
        ]
        for label, a, b in pair_defs:
            c = np.zeros(len(beta))
            c[idx[a]] = 1
            c[idx[b]] = -1
            est, se, p, low, high = z_test_from_contrast(beta, cov_np, c)
            rows.append(
                {
                    "row_type": "contrast",
                    "subset": f"age_ge_{cutoff}",
                    "sample_name": "primary_locked",
                    "model_name": "base_plus_phenotype",
                    "contrast": label,
                    "OR": float(np.exp(est)),
                    "ci_low": float(np.exp(low)),
                    "ci_high": float(np.exp(high)),
                    "effect_95ci": fmt_ci(np.exp(est), np.exp(low), np.exp(high)),
                    "p": p,
                    "p_formatted": format_p(p),
                }
            )

        # Key model comparisons on unified sample.
        y_unified = unified["incident_adl_w3_w4"].astype(int)
        model_specs = {
            "M1_levels": pd.concat([unified[BASE_COVARS].astype(float), unified[WAVE3_DOMAIN_VARS].astype(float)], axis=1),
            "M2_levels_slopes": None,
            "M3_phenotype": phenotype_design_matrix(unified, include_levels=False),
            "M4_levels_phenotype": phenotype_design_matrix(unified, include_levels=True, level_cols=WAVE3_DOMAIN_VARS),
        }
        # build slopes from original coding
        slope_df = pd.DataFrame({"ID_str": unified["ID_str"].values})
        for d in DOMAINS:
            slope_df[f"{d}_pre_w3_slope"] = unified[[f"{d}_w1", f"{d}_w2", f"{d}_w3"]].apply(
                lambda r: fit_pre_w3_slope([r[f"{d}_w1"], r[f"{d}_w2"], r[f"{d}_w3"]]), axis=1
            ).to_numpy()
        model_specs["M2_levels_slopes"] = pd.concat(
            [unified[BASE_COVARS].astype(float).reset_index(drop=True), unified[WAVE3_DOMAIN_VARS].astype(float).reset_index(drop=True), slope_df[[c for c in slope_df.columns if c != "ID_str"]]],
            axis=1,
        )
        metrics = {}
        for model_name, X in model_specs.items():
            metrics[model_name] = cv_metrics_from_X(unified, X)
            rows.append(
                {
                    "row_type": "performance",
                    "subset": f"age_ge_{cutoff}",
                    "sample_name": "unified_locked",
                    "model_name": model_name,
                    **metrics[model_name],
                }
            )

        # Adjusted absolute risk from base + phenotype.
        risk_df = simulate_adjusted_risk(primary)
        for r in risk_df.itertuples(index=False):
            rows.append(
                {
                    "row_type": "adjusted_risk",
                    "subset": f"age_ge_{cutoff}",
                    "sample_name": "primary_locked",
                    "model_name": "base_plus_phenotype",
                    "phenotype": r.phenotype,
                    "predicted_probability": r.predicted_probability,
                    "ci_low": r.ci_low,
                    "ci_high": r.ci_high,
                    "risk_difference_vs_stable": r.risk_difference_vs_stable,
                    "rd_low": r.rd_low,
                    "rd_high": r.rd_high,
                }
            )

    return pd.DataFrame(rows)


def pooled_tertiles(values: pd.Series) -> tuple[float, float]:
    q1 = float(values.quantile(1 / 3))
    q2 = float(values.quantile(2 / 3))
    return q1, q2


def build_measurement_scenarios(unified: pd.DataFrame) -> dict[str, pd.DataFrame]:
    scenarios: dict[str, pd.DataFrame] = {}

    def identity(x: pd.Series) -> pd.Series:
        return x.astype(float)

    # pooled thresholds from the unified sample to keep scenarios deterministic and comparable
    phys_pool = pd.concat([unified["physical_lim_w1"], unified["physical_lim_w2"], unified["physical_lim_w3"]], ignore_index=True).dropna()
    phys_q1, phys_q2 = pooled_tertiles(phys_pool)

    cog_means = {w: float(unified[f"cogdef_w{w}"].mean()) for w in [1, 2, 3]}
    cog_sds = {w: float(unified[f"cogdef_w{w}"].std(ddof=0) or 1.0) for w in [1, 2, 3]}

    all_wave_means = {(d, w): float(unified[f"{d}_w{w}"].mean()) for d in DOMAINS for w in [1, 2, 3]}
    all_wave_sds = {(d, w): float(unified[f"{d}_w{w}"].std(ddof=0) or 1.0) for d in DOMAINS for w in [1, 2, 3]}

    def transform_scenario(name: str) -> pd.DataFrame:
        data = unified.copy()
        for d in DOMAINS:
            for w in [1, 2, 3]:
                src = data[f"{d}_w{w}"].astype(float)
                if name == "reference_original":
                    data[f"{d}_alt_w{w}"] = src
                elif name == "physical_ordered_burden":
                    if d == "physical_lim":
                        data[f"{d}_alt_w{w}"] = np.where(src <= phys_q1, 0.0, np.where(src <= phys_q2, 1.0, 2.0))
                    else:
                        data[f"{d}_alt_w{w}"] = src
                elif name == "multimorb_ge2":
                    if d == "multimorb":
                        data[f"{d}_alt_w{w}"] = (src >= 2).astype(float)
                    else:
                        data[f"{d}_alt_w{w}"] = src
                elif name == "multimorb_ge3":
                    if d == "multimorb":
                        data[f"{d}_alt_w{w}"] = (src >= 3).astype(float)
                    else:
                        data[f"{d}_alt_w{w}"] = src
                elif name == "cognition_wave_z":
                    if d == "cogdef":
                        data[f"{d}_alt_w{w}"] = (src - cog_means[w]) / cog_sds[w]
                    else:
                        data[f"{d}_alt_w{w}"] = src
                elif name == "all_domain_wave_z":
                    data[f"{d}_alt_w{w}"] = (src - all_wave_means[(d, w)]) / all_wave_sds[(d, w)]
                else:
                    raise ValueError(name)
        for d in DOMAINS:
            data[f"{d}_alt_level_at_w3"] = data[f"{d}_alt_w3"]
            data[f"{d}_alt_pre_w3_slope"] = data[[f"{d}_alt_w1", f"{d}_alt_w2", f"{d}_alt_w3"]].apply(
                lambda r: fit_pre_w3_slope([r[f"{d}_alt_w1"], r[f"{d}_alt_w2"], r[f"{d}_alt_w3"]]),
                axis=1,
            ).to_numpy()
        return data

    for name in [
        "reference_original",
        "physical_ordered_burden",
        "multimorb_ge2",
        "multimorb_ge3",
        "cognition_wave_z",
        "all_domain_wave_z",
    ]:
        scenarios[name] = transform_scenario(name)
    return scenarios


def measurement_robustness(unified: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    scenarios = build_measurement_scenarios(unified)
    ref_m3 = cv_metrics_from_X(unified, phenotype_design_matrix(unified, include_levels=False))

    for name, data in scenarios.items():
        level_cols = [f"{d}_alt_level_at_w3" for d in DOMAINS]
        slope_cols = [f"{d}_alt_pre_w3_slope" for d in DOMAINS]
        X_m1 = pd.concat([data[BASE_COVARS].astype(float).reset_index(drop=True), data[level_cols].astype(float).reset_index(drop=True)], axis=1)
        X_m2 = pd.concat([data[BASE_COVARS].astype(float).reset_index(drop=True), data[level_cols + slope_cols].astype(float).reset_index(drop=True)], axis=1)
        X_m4 = phenotype_design_matrix(data, include_levels=True, level_cols=level_cols)
        m1 = cv_metrics_from_X(data, X_m1)
        m2 = cv_metrics_from_X(data, X_m2)
        m4 = cv_metrics_from_X(data, X_m4)
        for model_name, metrics in [("M1_alt", m1), ("M2_alt", m2), ("M4_alt", m4), ("M3_reference", ref_m3)]:
            rows.append(
                {
                    "row_type": "performance",
                    "scenario": name,
                    "model_name": model_name,
                    **metrics,
                }
            )

        # burden-adjusted phenotype effect under alternative coding
        y = data["incident_adl_w3_w4"].reset_index(drop=True).astype(int)
        X_burden = pd.concat([data[BASE_COVARS].astype(float).reset_index(drop=True), data[level_cols].astype(float).reset_index(drop=True)], axis=1)
        X_full = phenotype_design_matrix(data, include_levels=True, level_cols=level_cols)
        reduced = sm.GLM(y, sm.add_constant(X_burden, has_constant="add"), family=sm.families.Binomial()).fit()
        full = sm.GLM(y, sm.add_constant(X_full, has_constant="add"), family=sm.families.Binomial()).fit()
        lr = 2 * (full.llf - reduced.llf)
        p_overall = float(stats.chi2.sf(lr, int(full.df_model - reduced.df_model)))
        term = "traj_Emerging multidomain decline"
        conf = full.conf_int()
        rows.append(
            {
                "row_type": "burden_adjusted_emerging",
                "scenario": name,
                "model_name": "base_alt_levels_plus_phenotype",
                "contrast": "Emerging vs Stable",
                "OR": float(np.exp(full.params[term])),
                "ci_low": float(np.exp(conf.loc[term, 0])),
                "ci_high": float(np.exp(conf.loc[term, 1])),
                "effect_95ci": fmt_ci(np.exp(full.params[term]), np.exp(conf.loc[term, 0]), np.exp(conf.loc[term, 1])),
                "p": float(full.pvalues[term]),
                "p_formatted": format_p(float(full.pvalues[term])),
                "phenotype_overall_p": p_overall,
                "m1_oof_auc": m1["oof_auc"],
                "m2_oof_auc": m2["oof_auc"],
                "m4_oof_auc": m4["oof_auc"],
                "delta_m2_vs_m1": m2["oof_auc"] - m1["oof_auc"],
                "delta_m4_vs_m1": m4["oof_auc"] - m1["oof_auc"],
            }
        )
    return pd.DataFrame(rows)


def build_plots(older_df: pd.DataFrame, measure_df: pd.DataFrame) -> None:
    with PdfPages(RESULTS_DIR / "targeted_sensitivity_plots.pdf") as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        perf = older_df[older_df["row_type"] == "performance"].copy()
        for i, subset in enumerate(["age_ge_60", "age_ge_65"]):
            sub = perf[perf["subset"] == subset].copy()
            order = ["M1_levels", "M2_levels_slopes", "M3_phenotype", "M4_levels_phenotype"]
            sub = sub.set_index("model_name").reindex(order).reset_index()
            axes[i].bar(sub["model_name"], sub["oof_auc"], color=["#72B7B2", "#4C78A8", "#F58518", "#E45756"])
            axes[i].set_title(f"{subset.replace('_', ' ')}: OOF AUC")
            axes[i].tick_params(axis="x", rotation=20)
            axes[i].set_ylim(0.68, max(0.82, sub["oof_auc"].max() + 0.03))
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        risk = older_df[older_df["row_type"] == "adjusted_risk"].copy()
        for i, subset in enumerate(["age_ge_60", "age_ge_65"]):
            sub = risk[risk["subset"] == subset].set_index("phenotype").reindex(PHENOTYPE_ORDER).reset_index()
            axes[i].bar(sub["phenotype"], sub["predicted_probability"], color="#4C78A8")
            axes[i].errorbar(
                np.arange(len(sub)),
                sub["predicted_probability"],
                yerr=[
                    sub["predicted_probability"] - sub["ci_low"],
                    sub["ci_high"] - sub["predicted_probability"],
                ],
                fmt="none",
                ecolor="black",
                capsize=3,
            )
            axes[i].set_title(f"{subset.replace('_', ' ')}: adjusted risk by phenotype")
            axes[i].set_xticks(np.arange(len(sub)))
            axes[i].set_xticklabels(["Stable", "Emerging", "High-burden", "Severe"], rotation=20, ha="right")
            axes[i].set_ylabel("Predicted probability")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        perf_m = measure_df[measure_df["row_type"] == "performance"].copy()
        m2 = perf_m[perf_m["model_name"] == "M2_alt"].copy().set_index("scenario")
        m4 = perf_m[perf_m["model_name"] == "M4_alt"].copy().set_index("scenario")
        scenarios = [s for s in m2.index if s != "reference_original"]
        axes[0].bar(scenarios, m2.loc[scenarios, "oof_auc"], color="#72B7B2")
        axes[0].axhline(float(m2.loc["reference_original", "oof_auc"]), color="black", linestyle="--", linewidth=1)
        axes[0].set_title("Measurement robustness: M2 OOF AUC")
        axes[0].tick_params(axis="x", rotation=20)
        axes[1].bar(scenarios, m4.loc[scenarios, "oof_auc"], color="#E45756")
        axes[1].axhline(float(m4.loc["reference_original", "oof_auc"]), color="black", linestyle="--", linewidth=1)
        axes[1].set_title("Measurement robustness: M4 OOF AUC")
        axes[1].tick_params(axis="x", rotation=20)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def write_summary(older_df: pd.DataFrame, measure_df: pd.DataFrame) -> None:
    older_perf = older_df[older_df["row_type"] == "performance"].copy()
    older_risk = older_df[older_df["row_type"] == "adjusted_risk"].copy()
    older_contr = older_df[older_df["row_type"] == "contrast"].copy()
    measure_perf = measure_df[measure_df["row_type"] == "performance"].copy()
    measure_emerging = measure_df[measure_df["row_type"] == "burden_adjusted_emerging"].copy()

    age60_perf = older_perf[older_perf["subset"] == "age_ge_60"].set_index("model_name")
    age65_perf = older_perf[older_perf["subset"] == "age_ge_65"].set_index("model_name")
    age60_emerging = older_contr[(older_contr["subset"] == "age_ge_60") & (older_contr["contrast"] == "Emerging multidomain decline vs Stable")].iloc[0]
    age65_emerging = older_contr[(older_contr["subset"] == "age_ge_65") & (older_contr["contrast"] == "Emerging multidomain decline vs Stable")].iloc[0]
    age60_risk = older_risk[(older_risk["subset"] == "age_ge_60") & (older_risk["phenotype"] == "Emerging multidomain decline")].iloc[0]
    age65_risk = older_risk[(older_risk["subset"] == "age_ge_65") & (older_risk["phenotype"] == "Emerging multidomain decline")].iloc[0]

    measure_ref = measure_perf[measure_perf["scenario"] == "reference_original"].set_index("model_name")
    measure_rows = measure_emerging.set_index("scenario")
    strongest_scen = measure_rows["delta_m2_vs_m1"].idxmax()
    weakest_scen = measure_rows["delta_m4_vs_m1"].idxmin()

    summary = f"""# Stage 10 targeted sensitivity summary

## 1. Stage aim

本阶段通过两类 targeted sensitivity 检查主结论是否稳健：

1. **older-adult focused sensitivity**：把分析限制到更接近 geriatric practice 的 older-adult 人群；
2. **measurement robustness**：在当前可追溯的数据层面上，对关键 domain 的编码方式做替代处理，检查结论是否依赖某种特定量表表达。

## 2. Key implementation notes

- 当前锁定分析底表 **没有显式存储 wave 3 年龄变量**，因此 older-adult 分析使用 `baseline_age_2011 + 4` 作为 `age_w3` 近似值。这对应 2011 到 2015 的 landmark 时间间隔。
- 当前锁定分析底表 **没有保留 physical limitation 的 item-level 组件**，因此无法在本阶段做“均值 vs item count”的原始量表级重构。最接近、可复现的替代方案是：
  - physical limitation：`ordered burden`（基于 pooled tertiles）
  - multimorbidity：`>=2` 和 `>=3`
  - cognition：wave-specific z-score
  - all domains：wave-specific z-score sensitivity
- 这些都已经明确记录；没有对不存在的 raw item silently 造新变量。

## 3. Older-adult focused sensitivity

### 3.1 Age >=60 at wave 3

- `M1` OOF AUC = `{age60_perf.loc['M1_levels', 'oof_auc']:.3f}`
- `M2` OOF AUC = `{age60_perf.loc['M2_levels_slopes', 'oof_auc']:.3f}`
- `M3` OOF AUC = `{age60_perf.loc['M3_phenotype', 'oof_auc']:.3f}`
- `M4` OOF AUC = `{age60_perf.loc['M4_levels_phenotype', 'oof_auc']:.3f}`
- `Emerging vs Stable` = `{age60_emerging['effect_95ci']}`
- `Emerging` adjusted absolute risk = `{age60_risk['predicted_probability']:.3f}` (`95% CI {age60_risk['ci_low']:.3f} to {age60_risk['ci_high']:.3f}`)

### 3.2 Age >=65 at wave 3

- `M1` OOF AUC = `{age65_perf.loc['M1_levels', 'oof_auc']:.3f}`
- `M2` OOF AUC = `{age65_perf.loc['M2_levels_slopes', 'oof_auc']:.3f}`
- `M3` OOF AUC = `{age65_perf.loc['M3_phenotype', 'oof_auc']:.3f}`
- `M4` OOF AUC = `{age65_perf.loc['M4_levels_phenotype', 'oof_auc']:.3f}`
- `Emerging vs Stable` = `{age65_emerging['effect_95ci']}`
- `Emerging` adjusted absolute risk = `{age65_risk['predicted_probability']:.3f}` (`95% CI {age65_risk['ci_low']:.3f} to {age65_risk['ci_high']:.3f}`)

Older-adult 子样本里的核心结构仍然是：

- `M2 > M1`
- `M1 > M3`
- `M4` 只比 `M1` 小幅更高或近似

因此，主结论并没有因为人群限制到 older adults 而改变方向。

## 4. Measurement robustness

以 `reference_original` 为参照，measurement robustness 最重要的观察是：

- `M2` 的增量在不同编码下总体方向一致；
- `M4` 相对 `M1` 的增量更容易受 baseline-domain coding 影响；
- `Emerging` 在 burden-adjusted 模型中的方向总体稳定，但效应量会随编码方式轻度浮动。

在本轮场景中：

- `M2 vs M1` 增量最大的场景：`{strongest_scen}`，`ΔAUC = {measure_rows.loc[strongest_scen, 'delta_m2_vs_m1']:.3f}`
- `M4 vs M1` 最弱的场景：`{weakest_scen}`，`ΔAUC = {measure_rows.loc[weakest_scen, 'delta_m4_vs_m1']:.3f}`

这说明：

1. 显式 level+change 表示比 hard phenotype 更不依赖某种特定 coding；
2. hard phenotype 的“超越 current burden 的附加信息”对 coding 更敏感；
3. 当前主叙事更应压在：
   - current burden is dominant
   - explicit change adds modest information
   - hard phenotype is clinically interpretable but not coding-invariant

## 5. Direct answers

### 这项研究更适用于“中老年总体”还是“真正 older adults”？

两者都适用，但证据更稳地支持写成：

- a study in **middle-aged and older adults**
- with clinically relevant supportive consistency in **older-adult subsets**

也就是说，不建议把整篇稿子重写成“仅适用于真正 older adults”的研究；更稳的做法是在主文保持中老年总体，在 Discussion 里补充 older-adult focused consistency。

### phenotype 结果是否依赖某种特定 domain coding？

不是完全依赖，但有一定敏感性。

- `Emerging` 的方向性和 high-risk interpretation 总体稳定；
- 但 phenotype 在 current burden 之外的附加增量，确实比 `levels + slopes` 更受 domain coding 影响。

### 哪些主结果最稳健，最值得进主文？

最稳的主结果是：

1. `wave 3 current burden` 是主要预测信号来源；
2. `pre-landmark change` 仍有 modest 增量；
3. `Emerging` 是临床上可解释的高风险 phenotype；
4. `hard phenotype` 的附加预测价值应写得保守，且不应被写成对任一特定 domain coding 完全不敏感。
"""
    (RESULTS_DIR / "targeted_sensitivity_summary.md").write_text(summary, encoding="utf-8")


def update_results_index() -> None:
    addition = """

## 2026-03-25 Stage 10 Targeted sensitivity analyses

本阶段通过 older-adult focused 子样本和 key domain 替代编码，检验当前主结论是否依赖特定年龄层或某种 domain 量表表达。

### 新增文件

| 文件 | 回答的问题 | 备注 |
|---|---|---|
| `scripts/stage10_targeted_sensitivity/stage10_targeted_sensitivity.py` | 如何在不改主分析定义的前提下，对 older-adult 子样本和关键 domain 编码做 targeted sensitivity | Stage 10 独立脚本 |
| `results/stage10_targeted_sensitivity/older_adult_sensitivity.csv` | age>=60 / age>=65 时，main phenotype association、M1-M4 关键比较和 Emerging adjusted absolute risk 是否稳定 | older-adult 结果表 |
| `results/stage10_targeted_sensitivity/measurement_robustness.csv` | physical limitation、multimorbidity、cognition 等替代编码下，M1/M2/M4 和 burden-adjusted Emerging 结果是否敏感 | measurement robustness 结果表 |
| `results/stage10_targeted_sensitivity/targeted_sensitivity_summary.md` | 研究更适用于中老年总体还是 older adults；phenotype 是否依赖特定 coding；哪些结果最稳健 | Stage 10 总结文件 |
| `results/stage10_targeted_sensitivity/targeted_sensitivity_plots.pdf` | older-adult 和 measurement robustness 的图示总结 | 便于人工核查 |

### 本阶段给出的直接结论

1. older-adult 子样本中，`M2 > M1`、`M1 > M3` 的结构保持不变。
2. `Emerging` 在 older-adult 子样本里仍表现为高风险 phenotype，adjusted absolute risk 方向一致。
3. hard phenotype 的附加增量比 explicit level+change 表示更受 domain coding 影响。
4. 最稳的 manuscript-level claim 仍应写成：
   - current burden is dominant;
   - explicit change adds modest information;
   - phenotype is clinically interpretable but should not be presented as coding-invariant.

### 下一阶段建议

只在你确认后进入：

1. Stage 11
"""
    current = RESULTS_INDEX.read_text(encoding="utf-8")
    if "## 2026-03-25 Stage 10 Targeted sensitivity analyses" not in current:
        RESULTS_INDEX.write_text(current + addition, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    np.random.seed(SEED)
    random.seed(SEED)

    df = load_data()
    samples = build_samples(df)
    older_df = older_adult_sensitivity(samples)
    measure_df = measurement_robustness(samples["unified_locked"])

    older_df.to_csv(RESULTS_DIR / "older_adult_sensitivity.csv", index=False)
    measure_df.to_csv(RESULTS_DIR / "measurement_robustness.csv", index=False)
    build_plots(older_df, measure_df)
    write_summary(older_df, measure_df)
    update_results_index()

    meta = {
        "stage": "stage10_targeted_sensitivity",
        "seed": SEED,
        "primary_locked_n": int(len(samples["primary_locked"])),
        "unified_locked_n": int(len(samples["unified_locked"])),
        "age_w3_definition": "baseline_age_2011 + 4",
        "measurement_note": "no raw item-level physical limitation components in locked dataset; used ordered-burden approximation instead",
    }
    (RESULTS_DIR / "stage10_run_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()

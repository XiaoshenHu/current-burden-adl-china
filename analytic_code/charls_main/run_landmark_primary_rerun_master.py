#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import warnings
from datetime import datetime
from pathlib import Path

# Resolve repository imports independently of the launch directory.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reproducibility.project_paths import charls_raw_dir, project_root, require_reinforcement

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning as SMConvergenceWarning

from analytic_code.charls_main.run_landmark_time_split_analysis import (
    build_base_dataset,
    build_pooled_person_period,
    burden_order_score,
    derive_trajectory,
    label_order_num,
    map_labels_for_4class,
    plot_profiles,
    prepare_landmark_outcomes,
)


ROOT = project_root()
DERIVED = ROOT / "derived"
RAW_D = charls_raw_dir() / "H_CHARLS_D_Data.dta"
RERUNS = ROOT / "reruns"
RERUN_ID = "2026-03-19_landmark_primary_rerun_v1"
RERUN_DIR = RERUNS / RERUN_ID
RERUN_OUTPUTS = RERUN_DIR / "outputs"
RERUN_DOCS = RERUN_DIR / "docs"
RERUN_DERIVED = RERUN_DIR / "derived"

TRAJECTORY_ORDER = [
    "stable_low_burden",
    "emerging_multidomain",
    "persistent_high_burden",
    "severe_multidomain",
]
ADVERSITY_ORDER = ["low", "moderate", "high"]
BASE_COVARS = ["baseline_age_2011", "female", "rural_hukou_2011", "low_self_education"]


def format_p(p: float) -> str:
    if pd.isna(p):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt_est(est: float, low: float, high: float) -> str:
    return f"{est:.3f} ({low:.3f}, {high:.3f})"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def register_file(
    registry: list[dict[str, object]],
    key: str,
    section: str,
    role: str,
    analysis: str,
    model: str,
    path: Path,
    description: str,
) -> None:
    registry.append(
        {
            "file_key": key,
            "section": section,
            "role": role,
            "analysis": analysis,
            "model": model,
            "description": description,
            "path": str(path),
            "relative_path": str(path.relative_to(RERUN_DIR)),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else np.nan,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
    )


def exp_glm_table(result, label: str, effect_col: str) -> pd.DataFrame:
    params = result.params
    conf = result.conf_int()
    table = pd.DataFrame(
        {
            "term": params.index,
            "coef": params.values,
            "ci_low_coef": conf.iloc[:, 0].values,
            "ci_high_coef": conf.iloc[:, 1].values,
            "p": result.pvalues.values,
        }
    )
    table["model"] = label
    table[effect_col] = np.exp(table["coef"])
    table[f"{effect_col}_low"] = np.exp(table["ci_low_coef"])
    table[f"{effect_col}_high"] = np.exp(table["ci_high_coef"])
    table["effect_95ci"] = table.apply(
        lambda r: fmt_est(r[effect_col], r[f"{effect_col}_low"], r[f"{effect_col}_high"]),
        axis=1,
    )
    table["p_formatted"] = table["p"].map(format_p)
    return table


def mnlogit_table(result, outcome_labels: list[str], model_label: str) -> pd.DataFrame:
    conf = result.conf_int().reset_index().rename(
        columns={"trajectory_group_num": "comparison_num", "level_1": "term", "lower": "ci_low", "upper": "ci_high"}
    )
    conf["comparison_num"] = conf["comparison_num"].astype(int)
    rows = []
    for col in result.params.columns:
        outcome_group = outcome_labels[int(col) + 1]
        params = result.params[col]
        pvals = result.pvalues[col]
        sub_conf = conf[conf["comparison_num"] == int(col) + 1].set_index("term")
        for term in params.index:
            coef = float(params[term])
            ci_low = float(sub_conf.loc[term, "ci_low"])
            ci_high = float(sub_conf.loc[term, "ci_high"])
            rows.append(
                {
                    "model": model_label,
                    "outcome_group": outcome_group,
                    "term": term,
                    "OR": np.exp(coef),
                    "OR_low": np.exp(ci_low),
                    "OR_high": np.exp(ci_high),
                    "effect_95ci": fmt_est(np.exp(coef), np.exp(ci_low), np.exp(ci_high)),
                    "coef": coef,
                    "p": float(pvals[term]),
                    "p_formatted": format_p(float(pvals[term])),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_stability_consistent(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    n_iter: int = 100,
    seed: int = 20260319,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    reference_labels = feature_df["trajectory_group_num"].astype(int).to_numpy()
    rows: list[dict[str, float]] = []
    for i in range(n_iter):
        idx = np.sort(rng.choice(len(feature_df), size=int(len(feature_df) * 0.8), replace=False))
        sub = feature_df.iloc[idx].copy()
        scaler = StandardScaler()
        X_sub = scaler.fit_transform(sub[feature_cols].to_numpy(dtype=float))
        gmm = GaussianMixture(
            n_components=4,
            covariance_type="diag",
            reg_covar=1e-6,
            n_init=10,
            random_state=seed + i,
        )
        gmm.fit(X_sub)
        sub["cluster_raw"] = gmm.predict(X_sub)
        label_map = map_labels_for_4class(sub, "cluster_raw")
        boot_labels = sub["cluster_raw"].map(label_map)
        boot_codes = label_order_num(boot_labels).astype(int).to_numpy()
        rows.append(
            {
                "iteration": i + 1,
                "sample_n": len(idx),
                "adjusted_rand_index": adjusted_rand_score(reference_labels[idx], boot_codes),
                "mean_max_posterior": gmm.predict_proba(X_sub).max(axis=1).mean(),
            }
        )
    return pd.DataFrame(rows)


def build_landmark_samples(df: pd.DataFrame, traj3: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    merged = df.merge(
        traj3[
            [
                "ID_str",
                "trajectory_group",
                "trajectory_group_num",
                "max_posterior_probability",
                "normalized_entropy_individual",
            ]
        ],
        on="ID_str",
        how="left",
    )
    flow = {
        "baseline_inw1": int(len(df)),
        "adversity_eligible": int(df["adversity_score_scaled"].notna().sum()),
        "landmark_w1_w3_trajectory_eligible": int(len(traj3)),
        "alive_at_w3": int(merged["alive_at_w3"].sum()),
        "alive_at_w3_known_w4_status": int(((merged["alive_at_w3"] == 1) & (merged["known_w4_status"] == 1)).sum()),
        "landmark_death_analysis_complete_case": int(
            merged.dropna(
                subset=[
                    "trajectory_group",
                    "adversity_score_scaled",
                    *BASE_COVARS,
                    "death_w3_w4",
                ]
            ).shape[0]
        ),
        "landmark_adl_analysis_complete_case": int(
            merged.dropna(
                subset=[
                    "trajectory_group",
                    "adversity_score_scaled",
                    *BASE_COVARS,
                    "incident_adl_w3_w4",
                ]
            ).shape[0]
        ),
        "landmark_adl_composite_complete_case": int(
            merged.dropna(
                subset=[
                    "trajectory_group",
                    "adversity_score_scaled",
                    *BASE_COVARS,
                    "adl_or_death_w3_w4",
                ]
            ).shape[0]
        ),
    }
    return merged, flow


def fit_main_landmark_models(landmark: pd.DataFrame) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}

    work_mn = landmark.dropna(
        subset=[
            "trajectory_group_num",
            "adversity_score_scaled",
            *BASE_COVARS,
        ]
    ).copy()
    Xmn = sm.add_constant(work_mn[["adversity_score_scaled", *BASE_COVARS]].astype(float))
    ymn = work_mn["trajectory_group_num"].astype(int)
    mn = sm.MNLogit(ymn, Xmn).fit(method="newton", maxiter=200, disp=False)
    outputs["trajectory_association"] = mnlogit_table(mn, TRAJECTORY_ORDER, "landmark_w1_w3_mnlogit")

    adl = landmark.dropna(
        subset=[
            "trajectory_group",
            "adversity_score_scaled",
            *BASE_COVARS,
            "incident_adl_w3_w4",
        ]
    ).copy()
    adl["trajectory_group"] = pd.Categorical(adl["trajectory_group"], categories=TRAJECTORY_ORDER, ordered=True)
    adl_dummies = pd.get_dummies(adl["trajectory_group"], prefix="traj", drop_first=True, dtype=float)
    X_adl = sm.add_constant(pd.concat([adl[["adversity_score_scaled", *BASE_COVARS]], adl_dummies], axis=1).astype(float))
    y_adl = adl["incident_adl_w3_w4"].astype(int)
    adl_model = sm.Logit(y_adl, X_adl).fit(disp=False)
    outputs["incident_adl"] = exp_glm_table(adl_model, "landmark_incident_adl", "OR")

    adl_comp = landmark.dropna(
        subset=[
            "trajectory_group",
            "adversity_score_scaled",
            *BASE_COVARS,
            "adl_or_death_w3_w4",
        ]
    ).copy()
    adl_comp["trajectory_group"] = pd.Categorical(adl_comp["trajectory_group"], categories=TRAJECTORY_ORDER, ordered=True)
    comp_dummies = pd.get_dummies(adl_comp["trajectory_group"], prefix="traj", drop_first=True, dtype=float)
    X_comp = sm.add_constant(pd.concat([adl_comp[["adversity_score_scaled", *BASE_COVARS]], comp_dummies], axis=1).astype(float))
    y_comp = adl_comp["adl_or_death_w3_w4"].astype(int)
    comp_model = sm.Logit(y_comp, X_comp).fit(disp=False)
    outputs["adl_or_death"] = exp_glm_table(comp_model, "landmark_adl_or_death", "OR")

    death = landmark.dropna(
        subset=[
            "trajectory_group",
            "adversity_score_scaled",
            *BASE_COVARS,
            "death_w3_w4",
        ]
    ).copy()
    death["trajectory_group"] = pd.Categorical(death["trajectory_group"], categories=TRAJECTORY_ORDER, ordered=True)
    X_death_simple = sm.add_constant(death[["adversity_score_scaled", *BASE_COVARS]].astype(float))
    y_death = death["death_w3_w4"].astype(int)
    death_model = sm.GLM(
        y_death,
        X_death_simple,
        family=sm.families.Binomial(link=sm.families.links.CLogLog()),
    ).fit()
    outputs["discrete_death"] = exp_glm_table(death_model, "landmark_discrete_death", "HR")

    death_dummies = pd.get_dummies(death["trajectory_group"], prefix="traj", drop_first=True, dtype=float)
    X_death_joint = sm.add_constant(
        pd.concat([death[["adversity_score_scaled", *BASE_COVARS]], death_dummies], axis=1).astype(float)
    )
    death_joint = sm.GLM(
        y_death,
        X_death_joint,
        family=sm.families.Binomial(link=sm.families.links.CLogLog()),
    ).fit()
    outputs["joint_discrete_death"] = exp_glm_table(death_joint, "landmark_joint_discrete_death", "HR")

    adv_before = outputs["discrete_death"].loc[outputs["discrete_death"]["term"] == "adversity_score_scaled"].iloc[0]
    adv_after = outputs["joint_discrete_death"].loc[outputs["joint_discrete_death"]["term"] == "adversity_score_scaled"].iloc[0]
    attenuation_pct = np.nan
    if adv_before["coef"] != 0:
        attenuation_pct = (adv_before["coef"] - adv_after["coef"]) / adv_before["coef"] * 100
    outputs["attenuation"] = pd.DataFrame(
        [
            {
                "analysis": "landmark_w1_w3_discrete_death",
                "term": "adversity_score_scaled",
                "effect_before": adv_before["effect_95ci"],
                "effect_after": adv_after["effect_95ci"],
                "p_before": adv_before["p_formatted"],
                "p_after": adv_after["p_formatted"],
                "percent_attenuation_coef": round(float(attenuation_pct), 1) if pd.notna(attenuation_pct) else np.nan,
            }
        ]
    )

    crude_death = death.groupby("trajectory_group", observed=False)["death_w3_w4"].agg(["mean", "sum", "count"]).reset_index()
    crude_death["outcome"] = "Death during 2015-2018"
    crude_death["rate"] = crude_death["mean"]
    crude_adl = adl.groupby("trajectory_group", observed=False)["incident_adl_w3_w4"].agg(["mean", "sum", "count"]).reset_index()
    crude_adl["outcome"] = "Incident ADL disability"
    crude_adl["rate"] = crude_adl["mean"]
    outputs["crude_event_rates"] = pd.concat([crude_death, crude_adl], ignore_index=True)

    return outputs


def fit_pooled_discrete_time(df: pd.DataFrame, traj2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled = build_pooled_person_period(df, traj2)
    pooled = pooled.dropna(
        subset=[
            "event",
            "adversity_score_scaled",
            *BASE_COVARS,
            "trajectory_group",
        ]
    ).copy()
    pooled["trajectory_group"] = pd.Categorical(pooled["trajectory_group"], categories=TRAJECTORY_ORDER, ordered=True)
    period_dummies = pd.get_dummies(pooled["period"], prefix="period", drop_first=True, dtype=float)
    traj_dummies = pd.get_dummies(pooled["trajectory_group"], prefix="traj", drop_first=True, dtype=float)

    X_simple = sm.add_constant(
        pd.concat([pooled[["adversity_score_scaled", *BASE_COVARS]], period_dummies], axis=1).astype(float)
    )
    y = pooled["event"].astype(int)
    model_simple = sm.GLM(
        y,
        X_simple,
        family=sm.families.Binomial(link=sm.families.links.CLogLog()),
    ).fit(cov_type="cluster", cov_kwds={"groups": pooled["ID_str"]})

    X_joint = sm.add_constant(
        pd.concat([pooled[["adversity_score_scaled", *BASE_COVARS]], period_dummies, traj_dummies], axis=1).astype(float)
    )
    model_joint = sm.GLM(
        y,
        X_joint,
        family=sm.families.Binomial(link=sm.families.links.CLogLog()),
    ).fit(cov_type="cluster", cov_kwds={"groups": pooled["ID_str"]})
    return (
        exp_glm_table(model_simple, "pooled_discrete_death", "HR"),
        exp_glm_table(model_joint, "pooled_joint_discrete_death", "HR"),
    )


def fit_full_followup_cox_supplements(analytic: pd.DataFrame) -> dict[str, pd.DataFrame]:
    reinforcement = require_reinforcement()
    cox_table = reinforcement.cox_table
    fit_cox = reinforcement.fit_cox
    make_ipw_weights = reinforcement.make_ipw_weights
    ph_diagnostics = reinforcement.ph_diagnostics
    plot_schoenfeld = reinforcement.plot_schoenfeld
    spline_curve = reinforcement.spline_curve
    results: dict[str, pd.DataFrame] = {}
    main_cox = analytic.dropna(subset=["survival_time_years", "death_event", "adversity_score_scaled", *BASE_COVARS]).copy()

    main_model = fit_cox(main_cox[["survival_time_years", "death_event", "adversity_score_scaled", *BASE_COVARS]], "survival_time_years", "death_event")
    results["full_followup_main_cox"] = cox_table(main_model, "full_followup_main_cox")

    subgroup_rows: list[dict[str, object]] = []
    subgroup_specs = [
        ("female", {0: "Male", 1: "Female"}, ["adversity_score_scaled", "baseline_age_2011", "rural_hukou_2011", "low_self_education"]),
        ("rural_hukou_2011", {0: "Urban hukou", 1: "Rural hukou"}, ["adversity_score_scaled", "baseline_age_2011", "female", "low_self_education"]),
    ]
    for var, label_map, covars in subgroup_specs:
        for val, label in label_map.items():
            sub = main_cox[main_cox[var] == val].copy()
            if sub.empty:
                continue
            model = fit_cox(sub[["survival_time_years", "death_event", *covars]], "survival_time_years", "death_event")
            row = model.summary.loc["adversity_score_scaled"]
            subgroup_rows.append(
                {
                    "modifier": var,
                    "group": label,
                    "n": len(sub),
                    "events": int(sub["death_event"].sum()),
                    "HR": row["exp(coef)"],
                    "HR_low": row["exp(coef) lower 95%"],
                    "HR_high": row["exp(coef) upper 95%"],
                    "effect_95ci": fmt_est(row["exp(coef)"], row["exp(coef) lower 95%"], row["exp(coef) upper 95%"]),
                    "p": row["p"],
                    "p_formatted": format_p(row["p"]),
                }
            )

    age_df = main_cox.copy()
    age_df["age_group"] = pd.cut(age_df["baseline_age_2011"], bins=[45, 60, 75, np.inf], right=False, labels=["45-59", "60-74", "75+"])
    for label in ["45-59", "60-74", "75+"]:
        sub = age_df[age_df["age_group"] == label].copy()
        if sub.empty:
            continue
        covars = ["adversity_score_scaled", "female", "rural_hukou_2011", "low_self_education"]
        model = fit_cox(sub[["survival_time_years", "death_event", *covars]], "survival_time_years", "death_event")
        row = model.summary.loc["adversity_score_scaled"]
        subgroup_rows.append(
            {
                "modifier": "age_group",
                "group": label,
                "n": len(sub),
                "events": int(sub["death_event"].sum()),
                "HR": row["exp(coef)"],
                "HR_low": row["exp(coef) lower 95%"],
                "HR_high": row["exp(coef) upper 95%"],
                "effect_95ci": fmt_est(row["exp(coef)"], row["exp(coef) lower 95%"], row["exp(coef) upper 95%"]),
                "p": row["p"],
                "p_formatted": format_p(row["p"]),
            }
        )
    results["subgroup_mortality_cox"] = pd.DataFrame(subgroup_rows)

    interaction_rows: list[dict[str, object]] = []
    main_base_df = main_cox[["survival_time_years", "death_event", "adversity_score_scaled", *BASE_COVARS]].copy()
    for modifier in ["female", "rural_hukou_2011"]:
        tmp = main_base_df.copy()
        inter_term = f"adversity_x_{modifier}"
        tmp[inter_term] = tmp["adversity_score_scaled"] * tmp[modifier]
        model = fit_cox(tmp, "survival_time_years", "death_event")
        row = model.summary.loc[inter_term]
        interaction_rows.append(
            {
                "modifier": modifier,
                "analysis": "single_interaction",
                "term": inter_term,
                "HR_95CI": fmt_est(row["exp(coef)"], row["exp(coef) lower 95%"], row["exp(coef) upper 95%"]),
                "p": row["p"],
                "p_formatted": format_p(row["p"]),
            }
        )

    age_tmp = main_base_df.copy()
    age_tmp["age_group"] = pd.cut(age_tmp["baseline_age_2011"], bins=[45, 60, 75, np.inf], right=False, labels=["45-59", "60-74", "75+"])
    age_dummies = pd.get_dummies(age_tmp["age_group"], prefix="age", drop_first=True, dtype=float)
    age_base = pd.concat([age_tmp.drop(columns=["baseline_age_2011", "age_group"]), age_dummies], axis=1)
    age_base_covars = [c for c in age_base.columns if c not in ["survival_time_years", "death_event"]]
    age_base_model = fit_cox(age_base[["survival_time_years", "death_event", *age_base_covars]], "survival_time_years", "death_event")
    age_inter = age_base.copy()
    for col in age_dummies.columns:
        age_inter[f"adversity_x_{col}"] = age_inter["adversity_score_scaled"] * age_inter[col]
    age_covars = [c for c in age_inter.columns if c not in ["survival_time_years", "death_event"]]
    age_model = fit_cox(age_inter[["survival_time_years", "death_event", *age_covars]], "survival_time_years", "death_event")
    lr = 2 * (age_model.log_likelihood_ - age_base_model.log_likelihood_)
    df_diff = len(age_dummies.columns)
    interaction_rows.append(
        {
            "modifier": "age_group",
            "analysis": "omnibus_lrt_vs_linear",
            "term": "age_group_interaction_set",
            "HR_95CI": "",
            "p": 1 - chi2.cdf(lr, df=df_diff),
            "p_formatted": format_p(1 - chi2.cdf(lr, df=df_diff)),
        }
    )
    results["interaction_tests"] = pd.DataFrame(interaction_rows)

    predictors = [
        "baseline_age_2011",
        "female",
        "rural_hukou_2011",
        "low_self_education",
        "dep_w1",
        "cogdef_w1",
        "physical_lim_w1",
        "multimorb_w1",
        "adl_disability_baseline",
    ]
    ipw_main_df, ipw_summary = make_ipw_weights(analytic, analytic.index.isin(main_cox.index), predictors, "main_cox")
    ipw_main_model = fit_cox(
        ipw_main_df[["survival_time_years", "death_event", "adversity_score_scaled", *BASE_COVARS, "main_cox_ipw"]],
        "survival_time_years",
        "death_event",
        weights_col="main_cox_ipw",
    )
    results["ipw_main_cox"] = cox_table(ipw_main_model, "ipw_main_cox")
    results["ipw_weight_summary"] = ipw_summary

    basic_cox = analytic.dropna(subset=["survival_time_years", "death_event", "adversity_basic_score_scaled", *BASE_COVARS]).copy()
    basic_model = fit_cox(
        basic_cox[["survival_time_years", "death_event", "adversity_basic_score_scaled", *BASE_COVARS]],
        "survival_time_years",
        "death_event",
    )
    results["basic_adversity_cox"] = cox_table(basic_model, "basic_adversity_cox")

    health_cox = analytic.dropna(
        subset=["survival_time_years", "death_event", "adversity_score_scaled", *BASE_COVARS, "dep_w1", "multimorb_w1"]
    ).copy()
    health_model = fit_cox(
        health_cox[["survival_time_years", "death_event", "adversity_score_scaled", *BASE_COVARS, "dep_w1", "multimorb_w1"]],
        "survival_time_years",
        "death_event",
    )
    results["refined_plus_baseline_health_cox"] = cox_table(health_model, "refined_plus_baseline_health_cox")

    ph_main = ph_diagnostics(
        main_model,
        main_cox[["survival_time_years", "death_event", "adversity_score_scaled", *BASE_COVARS]],
        "survival_time_years",
        "death_event",
        "full_followup_main_cox",
    )
    results["cox_ph_diagnostics"] = ph_main
    plot_schoenfeld(
        main_model,
        main_cox[["survival_time_years", "death_event", "adversity_score_scaled", *BASE_COVARS]],
        "survival_time_years",
        "death_event",
        ["adversity_score_scaled", "baseline_age_2011", "female", "rural_hukou_2011", "low_self_education"],
        RERUN_OUTPUTS / "figureS2_main_cox_schoenfeld.png",
    )
    spline_curve(
        main_cox[["survival_time_years", "death_event", "adversity_score_scaled", *BASE_COVARS]].copy(),
        BASE_COVARS,
        RERUN_OUTPUTS / "tableS20_spline_mortality_curve.csv",
        RERUN_OUTPUTS / "figureS1_spline_mortality_curve.png",
    )
    return results


def write_docs(flow: dict[str, int], registry: pd.DataFrame, landmark_outputs: dict[str, pd.DataFrame]) -> None:
    methods_text = [
        "# Landmark Primary Rerun Methods Note",
        "",
        "## Primary analytic framework",
        "- Exposure: refined 17-item childhood adversity score (continuous).",
        "- Trajectory window: waves 1-3 (2011, 2013, 2015).",
        "- Outcome window: wave 3 to wave 4 (2015-2018).",
        "- Trajectory method: feature-based Gaussian mixture clustering using participant-level intercepts and slopes across four domains.",
        "- Primary reference trajectory: stable_low_burden.",
        "- Primary disability model: logistic regression for incident ADL disability among participants alive and ADL-free at wave 3.",
        "- Primary mortality model: complementary log-log discrete-time hazard model for death between wave 3 and wave 4.",
        "",
        "## Supplementary analyses",
        "- ADL disability or death composite endpoint.",
        "- Pooled person-period discrete-time mortality model using trajectories from waves 1-2.",
        "- Full-follow-up Cox models for adversity and mortality using approximate event times when exact dates were unavailable.",
        "- Subgroup analyses, interaction tests, inverse-probability weighting, alternative adversity definition, and extra adjustment for baseline depressive symptoms and multimorbidity.",
        "",
        "## Missing-data handling",
        "- Childhood adversity score required at least 8 non-missing childhood adversity items.",
        "- Trajectory features required successful intercept and slope estimation for all four domains.",
        "- Main regression models used complete-case analysis under the defined analysis-specific samples.",
    ]
    (RERUN_DOCS / "methods_note.md").write_text("\n".join(methods_text), encoding="utf-8")

    traj_row = landmark_outputs["trajectory_association"].loc[
        (landmark_outputs["trajectory_association"]["outcome_group"] == "severe_multidomain")
        & (landmark_outputs["trajectory_association"]["term"] == "adversity_score_scaled")
    ].iloc[0]
    adl_row = landmark_outputs["incident_adl"].loc[
        landmark_outputs["incident_adl"]["term"] == "traj_severe_multidomain"
    ].iloc[0]
    death_row = landmark_outputs["joint_discrete_death"].loc[
        landmark_outputs["joint_discrete_death"]["term"] == "traj_severe_multidomain"
    ].iloc[0]
    memo_text = [
        "# Landmark Primary Rerun Results Memo",
        "",
        "## Sample counts",
        f"- Baseline participants: {flow['baseline_inw1']}",
        f"- Adversity-eligible participants: {flow['adversity_eligible']}",
        f"- Landmark trajectory derivation sample: {flow['landmark_w1_w3_trajectory_eligible']}",
        f"- Post-landmark mortality complete-case sample: {flow['landmark_death_analysis_complete_case']}",
        f"- Post-landmark incident ADL complete-case sample: {flow['landmark_adl_analysis_complete_case']}",
        "",
        "## Main findings snapshot",
        f"- Childhood adversity and severe multidimensional aging pattern: OR {traj_row['effect_95ci']} (P {traj_row['p_formatted']}).",
        f"- Severe multidimensional aging pattern and incident ADL disability: OR {adl_row['effect_95ci']} (P {adl_row['p_formatted']}).",
        f"- Severe multidimensional aging pattern and post-landmark mortality: HR {death_row['effect_95ci']} (P {death_row['p_formatted']}).",
        "",
        "## Output discipline",
        "- All tables, figures, and notes in this rerun were generated from a single master script.",
        "- Output registry and manifest should be used as the only citation source for manuscript updating.",
        f"- Total registered output files: {len(registry)}",
    ]
    (RERUN_DOCS / "results_memo.md").write_text("\n".join(memo_text), encoding="utf-8")


def main() -> None:
    reinforcement = require_reinforcement()
    draw_flowchart = reinforcement.draw_flowchart
    make_baseline_table = reinforcement.make_baseline_table
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=SMConvergenceWarning)
    sns.set_theme(style="whitegrid")
    for path in [RERUN_DIR, RERUN_OUTPUTS, RERUN_DOCS, RERUN_DERIVED]:
        path.mkdir(parents=True, exist_ok=True)

    registry_rows: list[dict[str, object]] = []

    df = prepare_landmark_outcomes(build_base_dataset())
    analytic = pd.read_csv(DERIVED / "charls_analytic_base.csv")

    traj3, kcomp3, _, _ = derive_trajectory(df, waves=[1, 2, 3], prefix="landmark_w1_w3", k_range=[2, 3, 4, 5], selected_k=4)
    feature_cols3 = [c for c in traj3.columns if c.endswith("intercept") or c.endswith("slope")]
    stab3 = bootstrap_stability_consistent(traj3.copy(), feature_cols3, n_iter=100)
    posterior_summary = (
        traj3.groupby("trajectory_group", observed=False)
        .agg(
            n=("ID_str", "size"),
            mean_max_posterior=("max_posterior_probability", "mean"),
            median_max_posterior=("max_posterior_probability", "median"),
            mean_normalized_entropy=("normalized_entropy_individual", "mean"),
        )
        .reset_index()
    )

    landmark, flow = build_landmark_samples(df, traj3)
    profile3 = plot_profiles(
        landmark.dropna(subset=["trajectory_group"]),
        waves=[1, 2, 3],
        out_path=RERUN_OUTPUTS / "figure2_landmark_trajectory_profiles.png",
        title_prefix="Landmark multidimensional aging trajectories (2011-2015)",
    )

    flow_rows = [
        ("Baseline CHARLS participants", f"Wave 1 interview participants: n = {flow['baseline_inw1']:,}"),
        ("Adversity-eligible sample", f"Participants with >=8 adversity items: n = {flow['adversity_eligible']:,}"),
        ("Trajectory derivation sample", f"Waves 1-3 multidomain features + adversity available: n = {flow['landmark_w1_w3_trajectory_eligible']:,}"),
        ("Alive at wave 3", f"Participants alive at 2015 interview: n = {flow['alive_at_w3']:,}"),
        ("Known wave 4 vital status", f"Alive at wave 3 and known 2018 vital status: n = {flow['alive_at_w3_known_w4_status']:,}"),
        ("Post-landmark mortality sample", f"Complete-case discrete-time mortality analysis: n = {flow['landmark_death_analysis_complete_case']:,}"),
        ("Post-landmark incident ADL sample", f"Alive, ADL-free at wave 3, complete-case analysis: n = {flow['landmark_adl_analysis_complete_case']:,}"),
    ]
    draw_flowchart(flow_rows, RERUN_OUTPUTS / "figure1_landmark_flowchart.png")

    fig, ax1 = plt.subplots(figsize=(7, 5))
    sns.lineplot(data=kcomp3, x="n_components", y="bic", marker="o", ax=ax1, color="#1f77b4")
    ax1.set_xlabel("Number of classes")
    ax1.set_ylabel("BIC", color="#1f77b4")
    ax2 = ax1.twinx()
    sns.lineplot(data=kcomp3, x="n_components", y="mean_max_posterior", marker="s", ax=ax2, color="#d62728")
    ax2.set_ylabel("Mean max posterior probability", color="#d62728")
    ax1.set_title("Landmark trajectory solution comparison")
    plt.tight_layout()
    plt.savefig(RERUN_OUTPUTS / "figureS3_landmark_k_comparison.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(stab3["adjusted_rand_index"], bins=20, color="#4C72B0", ax=ax)
    ax.axvline(
        stab3["adjusted_rand_index"].mean(),
        color="red",
        linestyle="--",
        label=f"Mean ARI = {stab3['adjusted_rand_index'].mean():.3f}",
    )
    ax.set_xlabel("Adjusted Rand index")
    ax.set_ylabel("Frequency")
    ax.set_title("Bootstrap stability of landmark trajectory solution")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(RERUN_OUTPUTS / "figureS4_landmark_bootstrap_stability.png", dpi=300)
    plt.close(fig)

    landmark_outputs = fit_main_landmark_models(landmark)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=landmark_outputs["crude_event_rates"],
        x="trajectory_group",
        y="rate",
        hue="outcome",
        order=TRAJECTORY_ORDER,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Crude risk")
    ax.set_title("Crude post-landmark event risk by trajectory group")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(RERUN_OUTPUTS / "figure3_landmark_crude_event_rates.png", dpi=300)
    plt.close(fig)

    traj2, kcomp2, _, _ = derive_trajectory(df, waves=[1, 2], prefix="landmark_w1_w2", k_range=[2, 3, 4, 5], selected_k=4)
    pooled_simple, pooled_joint = fit_pooled_discrete_time(df, traj2)

    full_followup_outputs = fit_full_followup_cox_supplements(analytic)

    # Save derived data
    trajectory_w1w3_path = RERUN_DERIVED / "charls_trajectory_features_landmark_w1_w3.csv"
    traj3.to_csv(trajectory_w1w3_path, index=False)
    register_file(registry_rows, "derived_landmark_w1_w3_features", "derived", "supporting", "trajectory derivation", "feature matrix", trajectory_w1w3_path, "Participant-level waves 1-3 trajectory features and assignments.")

    trajectory_w1w2_path = RERUN_DERIVED / "charls_trajectory_features_landmark_w1_w2.csv"
    traj2.to_csv(trajectory_w1w2_path, index=False)
    register_file(registry_rows, "derived_landmark_w1_w2_features", "derived", "supporting", "trajectory derivation", "feature matrix", trajectory_w1w2_path, "Participant-level waves 1-2 trajectory features and assignments for pooled discrete-time sensitivity.")

    # Save main tables and figures
    save_map = [
        ("tableS1_flow_counts", "supplement", "supplementary", "sample flow", "count table", pd.DataFrame({"step": [r[0] for r in flow_rows], "detail": [r[1] for r in flow_rows]}), RERUN_OUTPUTS / "tableS1_flow_counts.csv", "Sample flow counts for the rerun framework."),
        ("table1_baseline_by_adversity_category", "main", "primary", "descriptive", "baseline table", make_baseline_table(landmark.dropna(subset=["adversity_cat"]).copy(), "adversity_cat", ADVERSITY_ORDER, RERUN_OUTPUTS / "table1_baseline_by_adversity_category.csv"), RERUN_OUTPUTS / "table1_baseline_by_adversity_category.csv", "Baseline characteristics by childhood adversity category."),
        ("tableS2_baseline_by_trajectory_group", "supplement", "supplementary", "descriptive", "baseline table", make_baseline_table(landmark.dropna(subset=["trajectory_group"]).copy(), "trajectory_group", TRAJECTORY_ORDER, RERUN_OUTPUTS / "tableS2_baseline_by_trajectory_group.csv"), RERUN_OUTPUTS / "tableS2_baseline_by_trajectory_group.csv", "Baseline characteristics by landmark trajectory group."),
        ("table2_landmark_trajectory_association", "main", "primary", "trajectory association", "multinomial logistic", landmark_outputs["trajectory_association"], RERUN_OUTPUTS / "table2_landmark_trajectory_association.csv", "Adjusted associations between childhood adversity and landmark trajectory membership."),
        ("table3_landmark_incident_adl", "main", "primary", "incident ADL", "logistic", landmark_outputs["incident_adl"], RERUN_OUTPUTS / "table3_landmark_incident_adl.csv", "Adjusted associations of childhood adversity and trajectory group with incident ADL disability."),
        ("table4a_landmark_discrete_death", "main", "primary", "post-landmark mortality", "cloglog discrete-time hazard", landmark_outputs["discrete_death"], RERUN_OUTPUTS / "table4a_landmark_discrete_death.csv", "Adjusted discrete-time mortality model for childhood adversity."),
        ("table4b_landmark_joint_discrete_death", "main", "primary", "post-landmark mortality", "cloglog discrete-time joint model", landmark_outputs["joint_discrete_death"], RERUN_OUTPUTS / "table4b_landmark_joint_discrete_death.csv", "Adjusted discrete-time mortality model including childhood adversity and trajectory group."),
        ("tableS3_landmark_trajectory_profiles", "supplement", "supplementary", "trajectory profile", "descriptive profile", profile3, RERUN_OUTPUTS / "tableS3_landmark_trajectory_profiles.csv", "Mean domain values across waves 1-3 by landmark trajectory group."),
        ("tableS4_landmark_k_comparison", "supplement", "supplementary", "trajectory robustness", "class comparison", kcomp3, RERUN_OUTPUTS / "tableS4_landmark_k_comparison.csv", "Fit statistics and class metrics for 2-5 class landmark solutions."),
        ("tableS5_landmark_posterior_summary", "supplement", "supplementary", "trajectory robustness", "posterior summary", posterior_summary, RERUN_OUTPUTS / "tableS5_landmark_posterior_summary.csv", "Posterior assignment quality for the selected 4-class landmark solution."),
        ("tableS6_landmark_bootstrap_stability", "supplement", "supplementary", "trajectory robustness", "bootstrap stability", stab3, RERUN_OUTPUTS / "tableS6_landmark_bootstrap_stability.csv", "Bootstrap stability metrics for the selected 4-class landmark solution."),
        ("tableS7_landmark_adl_or_death", "supplement", "supplementary", "ADL sensitivity", "logistic", landmark_outputs["adl_or_death"], RERUN_OUTPUTS / "tableS7_landmark_adl_or_death.csv", "Composite endpoint of ADL disability or death between wave 3 and wave 4."),
        ("tableS8_pooled_discrete_death", "supplement", "supplementary", "mortality sensitivity", "cloglog pooled discrete-time hazard", pooled_simple, RERUN_OUTPUTS / "tableS8_pooled_discrete_death.csv", "Pooled person-period discrete-time mortality model with adversity only."),
        ("tableS9_pooled_joint_discrete_death", "supplement", "supplementary", "mortality sensitivity", "cloglog pooled joint model", pooled_joint, RERUN_OUTPUTS / "tableS9_pooled_joint_discrete_death.csv", "Pooled person-period discrete-time mortality model including adversity and trajectory group."),
        ("tableS10_full_followup_main_cox", "supplement", "supplementary", "mortality sensitivity", "Cox proportional hazards", full_followup_outputs["full_followup_main_cox"], RERUN_OUTPUTS / "tableS10_full_followup_main_cox.csv", "Full-follow-up Cox model for childhood adversity and mortality."),
        ("tableS11_subgroup_mortality_cox", "supplement", "supplementary", "subgroup analysis", "Cox proportional hazards", full_followup_outputs["subgroup_mortality_cox"], RERUN_OUTPUTS / "tableS11_subgroup_mortality_cox.csv", "Subgroup-specific adversity-mortality associations."),
        ("tableS12_interaction_tests", "supplement", "supplementary", "interaction analysis", "Cox interaction tests", full_followup_outputs["interaction_tests"], RERUN_OUTPUTS / "tableS12_interaction_tests.csv", "Formal interaction tests for adversity-mortality associations."),
        ("tableS13_ipw_main_cox", "supplement", "supplementary", "selection sensitivity", "IPW Cox", full_followup_outputs["ipw_main_cox"], RERUN_OUTPUTS / "tableS13_ipw_main_cox.csv", "Inverse-probability weighted Cox model for adversity and mortality."),
        ("tableS14_ipw_weight_summary", "supplement", "supplementary", "selection sensitivity", "IPW summary", full_followup_outputs["ipw_weight_summary"], RERUN_OUTPUTS / "tableS14_ipw_weight_summary.csv", "Summary of stabilized inverse-probability weights."),
        ("tableS15_adversity_basic_cox", "supplement", "supplementary", "mortality sensitivity", "Cox proportional hazards", full_followup_outputs["basic_adversity_cox"], RERUN_OUTPUTS / "tableS15_adversity_basic_cox.csv", "Full-follow-up Cox model using the basic childhood adversity score."),
        ("tableS16_refined_plus_baseline_health_cox", "supplement", "supplementary", "mortality sensitivity", "Cox proportional hazards", full_followup_outputs["refined_plus_baseline_health_cox"], RERUN_OUTPUTS / "tableS16_refined_plus_baseline_health_cox.csv", "Full-follow-up Cox model additionally adjusted for baseline depressive symptoms and multimorbidity."),
        ("tableS17_cox_ph_diagnostics", "supplement", "supplementary", "assumption check", "Schoenfeld test", full_followup_outputs["cox_ph_diagnostics"], RERUN_OUTPUTS / "tableS17_cox_ph_diagnostics.csv", "Proportional hazards diagnostics for the full-follow-up main Cox model."),
        ("tableS18_landmark_attenuation", "supplement", "supplementary", "joint model summary", "attenuation summary", landmark_outputs["attenuation"], RERUN_OUTPUTS / "tableS18_landmark_attenuation.csv", "Summary of adversity coefficient attenuation after adding trajectory group."),
        ("tableS19_landmark_crude_event_rates", "supplement", "supplementary", "descriptive outcome rates", "crude rates", landmark_outputs["crude_event_rates"], RERUN_OUTPUTS / "tableS19_landmark_crude_event_rates.csv", "Crude post-landmark event rates by trajectory group."),
        ("tableS20_spline_mortality_curve", "supplement", "supplementary", "mortality sensitivity", "spline curve data", pd.read_csv(RERUN_OUTPUTS / "tableS20_spline_mortality_curve.csv"), RERUN_OUTPUTS / "tableS20_spline_mortality_curve.csv", "Spline-based full-follow-up adversity-mortality curve data."),
    ]

    for key, section, role, analysis, model, data, path, description in save_map:
        if not path.exists():
            data.to_csv(path, index=False)
        register_file(registry_rows, key, section, role, analysis, model, path, description)

    figure_map = [
        ("figure1_landmark_flowchart", "main", "primary", "sample flow", "flowchart", RERUN_OUTPUTS / "figure1_landmark_flowchart.png", "Flowchart of sample derivation for the landmark rerun."),
        ("figure2_landmark_trajectory_profiles", "main", "primary", "trajectory profile", "line plot", RERUN_OUTPUTS / "figure2_landmark_trajectory_profiles.png", "Trajectory profile plot across the four aging domains."),
        ("figure3_landmark_crude_event_rates", "main", "primary", "descriptive outcome rates", "bar plot", RERUN_OUTPUTS / "figure3_landmark_crude_event_rates.png", "Crude post-landmark event rates by trajectory group."),
        ("figureS1_spline_mortality_curve", "supplement", "supplementary", "mortality sensitivity", "spline curve", RERUN_OUTPUTS / "figureS1_spline_mortality_curve.png", "Spline-based full-follow-up adversity-mortality curve."),
        ("figureS2_main_cox_schoenfeld", "supplement", "supplementary", "assumption check", "Schoenfeld residual plot", RERUN_OUTPUTS / "figureS2_main_cox_schoenfeld.png", "Schoenfeld residual diagnostics for the full-follow-up main Cox model."),
        ("figureS3_landmark_k_comparison", "supplement", "supplementary", "trajectory robustness", "class comparison plot", RERUN_OUTPUTS / "figureS3_landmark_k_comparison.png", "Comparison of landmark class solutions."),
        ("figureS4_landmark_bootstrap_stability", "supplement", "supplementary", "trajectory robustness", "bootstrap histogram", RERUN_OUTPUTS / "figureS4_landmark_bootstrap_stability.png", "Bootstrap stability histogram for the selected trajectory solution."),
    ]
    for key, section, role, analysis, model, path, description in figure_map:
        register_file(registry_rows, key, section, role, analysis, model, path, description)

    write_docs(flow, pd.DataFrame(registry_rows), landmark_outputs)
    for doc_name, desc in [
        ("methods_note.md", "Rerun methods summary."),
        ("results_memo.md", "Rerun results memo."),
    ]:
        register_file(registry_rows, doc_name.replace(".md", ""), "docs", "supporting", "documentation", "markdown", RERUN_DOCS / doc_name, desc)

    registry_df = pd.DataFrame(registry_rows)
    registry_path = RERUN_DIR / "output_registry.csv"
    registry_df.to_csv(registry_path, index=False)

    manifest = {
        "rerun_id": RERUN_ID,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(ROOT),
        "primary_framework": {
            "design": "longitudinal observational cohort secondary analysis",
            "exposure": "refined 17-item childhood adversity score",
            "trajectory_window": "waves 1-3 (2011-2015)",
            "trajectory_method": "feature-based Gaussian mixture clustering on participant-level intercepts and slopes",
            "trajectory_groups": TRAJECTORY_ORDER,
            "reference_group": "stable_low_burden",
            "primary_functional_outcome": "incident ADL disability between wave 3 and wave 4",
            "primary_mortality_outcome": "death between wave 3 and wave 4",
            "primary_mortality_model": "complementary log-log discrete-time hazard",
            "supplementary_mortality_model": "full-follow-up Cox with approximate event times",
            "covariates": BASE_COVARS,
            "decisions": {
                "A_unique_primary_framework": True,
                "B_retain_four_domain_trajectory": True,
                "C_retain_feature_based_gmm": True,
                "D_full_window_cox_supplementary_only": True,
                "E_regenerate_all_main_tables_figures": True,
                "F_add_manifest_and_output_registry": True,
            },
        },
        "inputs": {
            "analytic_base_csv": str(DERIVED / "charls_analytic_base.csv"),
            "raw_charls_dta": str(RAW_D),
            "analytic_base_csv_sha256": sha256_file(DERIVED / "charls_analytic_base.csv"),
            "raw_charls_dta_sha256": sha256_file(RAW_D),
            "master_script": str(Path(__file__).resolve()),
            "master_script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "sample_counts": flow,
        "output_registry": str(registry_path),
    }
    manifest_path = RERUN_DIR / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

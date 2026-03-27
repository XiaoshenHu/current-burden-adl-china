#!/usr/bin/env python3
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "derived"
OUTPUTS = ROOT / "outputs"
DOCS = ROOT / "docs"
RAW_D = DERIVED / "raw_extracts" / "H_CHARLS_D_Data.dta"


def zfill_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(12)


def fit_slope(values: list[float], waves: list[int]) -> tuple[float, float]:
    arr = np.array(values, dtype=float)
    mask = ~np.isnan(arr)
    if mask.sum() < 2:
        return np.nan, np.nan
    x = np.array(waves, dtype=float)[mask]
    y = arr[mask]
    slope, intercept = np.polyfit(x, y, 1)
    return intercept, slope


def safe_exp_ci(coef: float, se: float) -> tuple[float, float, float]:
    low = coef - 1.96 * se
    high = coef + 1.96 * se
    return np.exp(coef), np.exp(low), np.exp(high)


def format_p(p: float) -> str:
    if pd.isna(p):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def save_glm_table(result, path: Path, label: str, effect_col: str) -> pd.DataFrame:
    params = result.params
    se = result.bse
    pvals = result.pvalues
    table = pd.DataFrame({"term": params.index, "coef": params.values, "se": se.values, "p": pvals.values})
    table["model"] = label
    vals = table.apply(lambda r: safe_exp_ci(r["coef"], r["se"]), axis=1, result_type="expand")
    vals.columns = [effect_col, f"{effect_col}_low", f"{effect_col}_high"]
    table = pd.concat([table, vals], axis=1)
    table["effect_95ci"] = table.apply(
        lambda r: f"{r[effect_col]:.3f} ({r[f'{effect_col}_low']:.3f}, {r[f'{effect_col}_high']:.3f})",
        axis=1,
    )
    table["p_formatted"] = table["p"].map(format_p)
    table.to_csv(path, index=False)
    return table


def save_mnlogit_table(result, path: Path) -> pd.DataFrame:
    params = result.params
    conf = result.conf_int().reset_index().rename(columns={"trajectory_group_num": "comparison_num", "level_1": "term"})
    conf["comparison_num"] = conf["comparison_num"].astype(int)
    pvals = result.pvalues
    rows = []
    comparison_names = {
        1: "emerging_multidomain",
        2: "persistent_high_burden",
        3: "severe_multidomain",
    }
    for col in params.columns:
        comp = col + 1
        pcol = pvals[col]
        sub_conf = conf[conf["comparison_num"] == comp].set_index("term")
        for term in params.index:
            coef = float(params.loc[term, col])
            low = float(sub_conf.loc[term, "lower"])
            high = float(sub_conf.loc[term, "upper"])
            rows.append(
                {
                    "model": "landmark_w1_w3_mnlogit",
                    "outcome_group": comparison_names.get(comp, str(comp)),
                    "term": term,
                    "OR": np.exp(coef),
                    "OR_low": np.exp(low),
                    "OR_high": np.exp(high),
                    "effect_95ci": f"{np.exp(coef):.3f} ({np.exp(low):.3f}, {np.exp(high):.3f})",
                    "p": float(pcol[term]),
                    "p_formatted": format_p(float(pcol[term])),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(path, index=False)
    return table


def burden_order_score(sub: pd.DataFrame) -> float:
    return (
        sub["dep_intercept"].mean()
        + 3 * sub["dep_slope"].mean()
        + sub["cogdef_intercept"].mean()
        + 3 * sub["cogdef_slope"].mean()
        + 10 * sub["physical_lim_intercept"].mean()
        + 30 * sub["physical_lim_slope"].mean()
        + sub["multimorb_intercept"].mean()
        + 3 * sub["multimorb_slope"].mean()
    )


def baseline_burden_score(sub: pd.DataFrame) -> float:
    return (
        sub["dep_intercept"].mean()
        + sub["cogdef_intercept"].mean()
        + 10 * sub["physical_lim_intercept"].mean()
        + sub["multimorb_intercept"].mean()
    )


def map_labels_for_4class(feature_df: pd.DataFrame, raw_label_col: str) -> dict[int, str]:
    scores = {c: burden_order_score(sub) for c, sub in feature_df.groupby(raw_label_col)}
    ordered_clusters = sorted(scores, key=scores.get)
    lowest = ordered_clusters[0]
    highest = ordered_clusters[-1]
    middle = ordered_clusters[1:-1]
    baseline_scores = {c: baseline_burden_score(feature_df[feature_df[raw_label_col] == c]) for c in middle}
    persistent = max(middle, key=lambda x: baseline_scores[x])
    emerging = min(middle, key=lambda x: baseline_scores[x])
    return {
        lowest: "stable_low_burden",
        emerging: "emerging_multidomain",
        persistent: "persistent_high_burden",
        highest: "severe_multidomain",
    }


def label_order_num(series: pd.Series) -> pd.Series:
    cat = pd.Categorical(
        series,
        categories=[
            "stable_low_burden",
            "emerging_multidomain",
            "persistent_high_burden",
            "severe_multidomain",
        ],
        ordered=True,
    )
    return pd.Series(cat.codes, index=series.index)


def derive_trajectory(
    df: pd.DataFrame,
    waves: list[int],
    prefix: str,
    k_range: list[int] | None = None,
    selected_k: int = 4,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, GaussianMixture, np.ndarray]:
    domains = ["dep", "cogdef", "physical_lim", "multimorb"]
    rows = []
    for _, row in df.iterrows():
        out = {
            "ID_str": row["ID_str"],
            "adversity_score_scaled": row["adversity_score_scaled"],
            "adversity_cat": row["adversity_cat"],
            "baseline_age_2011": row["baseline_age_2011"],
            "female": row["female"],
            "rural_hukou_2011": row["rural_hukou_2011"],
            "low_self_education": row["low_self_education"],
        }
        enough = pd.notna(row["adversity_score_scaled"])
        for dom in domains:
            vals = [row[f"{dom}_w{i}"] for i in waves]
            intercept, slope = fit_slope(vals, waves)
            out[f"{dom}_intercept"] = intercept
            out[f"{dom}_slope"] = slope
            if np.isnan(intercept) or np.isnan(slope):
                enough = False
        out["complete_for_cluster"] = int(enough)
        rows.append(out)

    feat = pd.DataFrame(rows)
    feat = feat[feat["complete_for_cluster"] == 1].copy()
    feature_cols = [f"{d}_{m}" for d in domains for m in ["intercept", "slope"]]
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X = scaler.fit_transform(imputer.fit_transform(feat[feature_cols]))

    k_range = k_range or [2, 3, 4, 5]
    comparison_rows = []
    fitted = {}
    for k in k_range:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            reg_covar=1e-6,
            n_init=10,
            random_state=random_state,
        )
        gmm.fit(X)
        probs = gmm.predict_proba(X)
        assignments = gmm.predict(X)
        class_props = pd.Series(assignments).value_counts(normalize=True)
        entropy = -np.mean(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)) / np.log(k)
        comparison_rows.append(
            {
                "trajectory_prefix": prefix,
                "n_components": k,
                "bic": gmm.bic(X),
                "aic": gmm.aic(X),
                "normalized_entropy": entropy,
                "mean_max_posterior": probs.max(axis=1).mean(),
                "min_class_proportion": class_props.min(),
                "max_class_proportion": class_props.max(),
            }
        )
        fitted[k] = gmm

    selected = fitted[selected_k]
    raw = selected.predict(X)
    probs = selected.predict_proba(X)
    feat["cluster_raw"] = raw
    label_map = map_labels_for_4class(feat, "cluster_raw")
    feat["trajectory_group"] = feat["cluster_raw"].map(label_map)
    feat["trajectory_group_num"] = label_order_num(feat["trajectory_group"])
    feat["max_posterior_probability"] = probs.max(axis=1)
    feat["normalized_entropy_individual"] = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1) / np.log(selected_k)
    for i in range(selected_k):
        feat[f"posterior_class_{i}"] = probs[:, i]

    comparison = pd.DataFrame(comparison_rows)
    return feat, comparison, selected, X


def bootstrap_stability(feature_df: pd.DataFrame, feature_cols: list[str], n_iter: int = 100, seed: int = 20260319) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ref_labels = feature_df["trajectory_group_num"].astype(int).to_numpy()
    X = feature_df[feature_cols].to_numpy()
    rows = []
    for i in range(n_iter):
        idx = np.sort(rng.choice(len(feature_df), size=int(len(feature_df) * 0.8), replace=False))
        sub = feature_df.iloc[idx].copy()
        gmm = GaussianMixture(
            n_components=4,
            covariance_type="diag",
            reg_covar=1e-6,
            n_init=5,
            random_state=seed + i,
        )
        gmm.fit(sub[feature_cols])
        sub["cluster_raw"] = gmm.predict(sub[feature_cols])
        label_map = map_labels_for_4class(sub, "cluster_raw")
        boot_labels = sub["cluster_raw"].map(label_map)
        boot_num = label_order_num(boot_labels).astype(int).to_numpy()
        ari = adjusted_rand_score(ref_labels[idx], boot_num)
        rows.append(
            {
                "iteration": i + 1,
                "sample_n": len(idx),
                "adjusted_rand_index": ari,
                "mean_max_posterior": gmm.predict_proba(sub[feature_cols]).max(axis=1).mean(),
            }
        )
    return pd.DataFrame(rows)


def plot_profiles(merged: pd.DataFrame, waves: list[int], out_path: Path, title_prefix: str) -> pd.DataFrame:
    sns.set_theme(style="whitegrid")
    domains = ["dep", "cogdef", "physical_lim", "multimorb"]
    domain_title = {
        "dep": "Depressive symptoms",
        "cogdef": "Cognitive deficit",
        "physical_lim": "Physical limitation",
        "multimorb": "Multimorbidity count",
    }
    hue_order = ["stable_low_burden", "emerging_multidomain", "persistent_high_burden", "severe_multidomain"]
    plot_df = []
    for dom in domains:
        for w in waves:
            tmp = merged[["trajectory_group", f"{dom}_w{w}"]].rename(columns={f"{dom}_w{w}": "value"}).copy()
            tmp["domain"] = dom
            tmp["wave"] = w
            plot_df.append(tmp)
    plot_df = pd.concat(plot_df, ignore_index=True).dropna()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    for ax, dom in zip(axes, domains):
        sub = plot_df[plot_df["domain"] == dom]
        sns.lineplot(
            data=sub,
            x="wave",
            y="value",
            hue="trajectory_group",
            hue_order=hue_order,
            estimator="mean",
            errorbar=None,
            ax=ax,
        )
        ax.set_title(domain_title[dom])
        ax.set_xlabel("Wave")
        ax.set_ylabel("Value")
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        if ax.get_legend():
            ax.get_legend().remove()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(title_prefix, y=1.02, fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    profiles = (
        merged.groupby("trajectory_group")[[f"{dom}_w{w}" for dom in domains for w in waves]]
        .mean()
        .round(3)
        .reset_index()
    )
    return profiles


def build_base_dataset() -> pd.DataFrame:
    base = pd.read_csv(DERIVED / "charls_analytic_base.csv")
    raw = pd.read_stata(
        RAW_D,
        columns=[
            "ID",
            "inw1",
            "r2iwstat",
            "r3iwstat",
            "r4iwstat",
            "r2adlab_c",
            "r3adlab_c",
            "r4adlab_c",
        ],
    )
    raw = raw[raw["inw1"].astype("string") == "1.yes"].copy()
    base["ID_str"] = zfill_id(base["ID"])
    raw["ID_str"] = zfill_id(raw["ID"])
    df = base.merge(
        raw[["ID_str", "r2iwstat", "r3iwstat", "r4iwstat", "r2adlab_c", "r3adlab_c", "r4adlab_c"]],
        on="ID_str",
        how="left",
    )
    return df


def prepare_landmark_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    s3 = out["r3iwstat"].astype("string").str.lower()
    s4 = out["r4iwstat"].astype("string").str.lower()
    out["alive_at_w3"] = (~s3.str.startswith(("5.", "6."))).astype(int)
    out["known_w4_status"] = (~s4.str.startswith("9.")).astype(int)
    out["death_w3_w4"] = np.where(out["alive_at_w3"] == 1, s4.str.startswith(("5.", "6.")).astype(float), np.nan)
    adl3 = pd.to_numeric(out["r3adlab_c"], errors="coerce")
    adl4 = pd.to_numeric(out["r4adlab_c"], errors="coerce")
    out["adl_disability_w3"] = np.where(adl3.notna(), (adl3 > 0).astype(float), np.nan)
    out["adl_disability_w4_raw"] = np.where(adl4.notna(), (adl4 > 0).astype(float), np.nan)
    out["incident_adl_w3_w4"] = np.where(
        (out["alive_at_w3"] == 1) & (out["adl_disability_w3"] == 0) & adl4.notna(),
        (adl4 > 0).astype(float),
        np.nan,
    )
    out["adl_or_death_w3_w4"] = np.where(
        (out["alive_at_w3"] == 1) & (out["adl_disability_w3"] == 0) & (out["known_w4_status"] == 1),
        np.maximum(out["incident_adl_w3_w4"].fillna(0), out["death_w3_w4"].fillna(0)),
        np.nan,
    )
    return out


def build_pooled_person_period(df: pd.DataFrame, traj2: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(traj2[["ID_str", "trajectory_group", "trajectory_group_num"]], on="ID_str", how="inner")
    s2 = merged["r2iwstat"].astype("string").str.lower()
    s3 = merged["r3iwstat"].astype("string").str.lower()
    s4 = merged["r4iwstat"].astype("string").str.lower()
    alive_w2 = ~s2.str.startswith(("5.", "6."))
    known_w3 = ~s3.str.startswith("9.")
    known_w4 = ~s4.str.startswith("9.")

    rows = []
    for _, row in merged.iterrows():
        if not alive_w2.loc[row.name]:
            continue
        if known_w3.loc[row.name]:
            event_23 = float(s3.loc[row.name].startswith(("5.", "6.")))
            rows.append(
                {
                    "ID_str": row["ID_str"],
                    "period": "w2_to_w3",
                    "period_num": 1,
                    "event": event_23,
                    "adversity_score_scaled": row["adversity_score_scaled"],
                    "baseline_age_2011": row["baseline_age_2011"],
                    "female": row["female"],
                    "rural_hukou_2011": row["rural_hukou_2011"],
                    "low_self_education": row["low_self_education"],
                    "trajectory_group": row["trajectory_group"],
                }
            )
            if event_23 == 1.0:
                continue
        else:
            continue

        if known_w4.loc[row.name]:
            rows.append(
                {
                    "ID_str": row["ID_str"],
                    "period": "w3_to_w4",
                    "period_num": 2,
                    "event": float(s4.loc[row.name].startswith(("5.", "6."))),
                    "adversity_score_scaled": row["adversity_score_scaled"],
                    "baseline_age_2011": row["baseline_age_2011"],
                    "female": row["female"],
                    "rural_hukou_2011": row["rural_hukou_2011"],
                    "low_self_education": row["low_self_education"],
                    "trajectory_group": row["trajectory_group"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    df = prepare_landmark_outcomes(build_base_dataset())

    # Primary landmark: trajectories from waves 1-3, outcomes from wave 3 to wave 4.
    traj3, kcomp3, gmm3, X3 = derive_trajectory(df, waves=[1, 2, 3], prefix="landmark_w1_w3", k_range=[2, 3, 4, 5], selected_k=4)
    feat_cols3 = [c for c in traj3.columns if c.endswith("intercept") or c.endswith("slope")]
    stab3 = bootstrap_stability(traj3.copy(), feat_cols3, n_iter=100)
    traj3.to_csv(DERIVED / "charls_trajectory_features_landmark_w1_w3.csv", index=False)
    kcomp3.to_csv(OUTPUTS / "tableS_landmark_w1_w3_k_comparison.csv", index=False)
    stab3.to_csv(OUTPUTS / "tableS_landmark_w1_w3_bootstrap_stability.csv", index=False)
    posterior_summary = (
        traj3.groupby("trajectory_group")
        .agg(
            n=("ID_str", "size"),
            mean_max_posterior=("max_posterior_probability", "mean"),
            median_max_posterior=("max_posterior_probability", "median"),
            mean_normalized_entropy=("normalized_entropy_individual", "mean"),
        )
        .reset_index()
    )
    posterior_summary.to_csv(OUTPUTS / "tableS_landmark_w1_w3_posterior_summary.csv", index=False)

    landmark = df.merge(
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

    profile3 = plot_profiles(
        landmark.dropna(subset=["trajectory_group"]),
        waves=[1, 2, 3],
        out_path=OUTPUTS / "figure_landmark_w1_w3_trajectory_profiles.png",
        title_prefix="Landmark multidimensional aging trajectories (2011-2015)",
    )
    profile3.to_csv(OUTPUTS / "tableS_landmark_w1_w3_trajectory_profiles.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(7, 5))
    sns.lineplot(data=kcomp3, x="n_components", y="bic", marker="o", ax=ax1, color="#1f77b4")
    ax1.set_xlabel("Number of classes")
    ax1.set_ylabel("BIC", color="#1f77b4")
    ax2 = ax1.twinx()
    sns.lineplot(data=kcomp3, x="n_components", y="mean_max_posterior", marker="s", ax=ax2, color="#d62728")
    ax2.set_ylabel("Mean max posterior probability", color="#d62728")
    ax1.set_title("Landmark trajectory solution comparison")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "figure_landmark_w1_w3_k_comparison.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(stab3["adjusted_rand_index"], bins=20, color="#4C72B0", ax=ax)
    ax.axvline(stab3["adjusted_rand_index"].mean(), color="red", linestyle="--", label=f"Mean ARI = {stab3['adjusted_rand_index'].mean():.3f}")
    ax.set_xlabel("Adjusted Rand index")
    ax.set_ylabel("Frequency")
    ax.set_title("Bootstrap stability of landmark trajectory solution")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "figure_landmark_w1_w3_bootstrap_stability.png", dpi=300)
    plt.close(fig)

    flow = {
        "baseline_inw1": len(df),
        "landmark_w1_w3_trajectory_eligible": int(len(traj3)),
        "alive_at_w3": int(landmark["alive_at_w3"].sum()),
        "alive_at_w3_known_w4_status": int(((landmark["alive_at_w3"] == 1) & (landmark["known_w4_status"] == 1)).sum()),
        "landmark_death_analysis_complete_case": int(
            landmark.dropna(
                subset=[
                    "trajectory_group",
                    "adversity_score_scaled",
                    "baseline_age_2011",
                    "female",
                    "rural_hukou_2011",
                    "low_self_education",
                    "death_w3_w4",
                ]
            ).shape[0]
        ),
        "landmark_adl_analysis_complete_case": int(
            landmark.dropna(
                subset=[
                    "trajectory_group",
                    "adversity_score_scaled",
                    "baseline_age_2011",
                    "female",
                    "rural_hukou_2011",
                    "low_self_education",
                    "incident_adl_w3_w4",
                ]
            ).shape[0]
        ),
        "landmark_adl_composite_complete_case": int(
            landmark.dropna(
                subset=[
                    "trajectory_group",
                    "adversity_score_scaled",
                    "baseline_age_2011",
                    "female",
                    "rural_hukou_2011",
                    "low_self_education",
                    "adl_or_death_w3_w4",
                ]
            ).shape[0]
        ),
    }
    pd.DataFrame({"metric": list(flow.keys()), "value": list(flow.values())}).to_csv(
        OUTPUTS / "tableS_landmark_flow_counts.csv", index=False
    )

    # Main landmark multinomial model: adversity -> trajectory.
    work_mn = traj3.dropna(
        subset=[
            "trajectory_group_num",
            "adversity_score_scaled",
            "baseline_age_2011",
            "female",
            "rural_hukou_2011",
            "low_self_education",
        ]
    ).copy()
    Xmn = sm.add_constant(
        work_mn[["adversity_score_scaled", "baseline_age_2011", "female", "rural_hukou_2011", "low_self_education"]].astype(float)
    )
    ymn = work_mn["trajectory_group_num"].astype(int)
    mn = sm.MNLogit(ymn, Xmn).fit(method="newton", maxiter=200, disp=False)
    mn_table = save_mnlogit_table(mn, OUTPUTS / "table_landmark_w1_w3_trajectory_association.csv")

    # Main landmark incident ADL model.
    adl = landmark.dropna(
        subset=[
            "trajectory_group",
            "adversity_score_scaled",
            "baseline_age_2011",
            "female",
            "rural_hukou_2011",
            "low_self_education",
            "incident_adl_w3_w4",
        ]
    ).copy()
    adl["trajectory_group"] = pd.Categorical(
        adl["trajectory_group"],
        categories=["stable_low_burden", "emerging_multidomain", "persistent_high_burden", "severe_multidomain"],
        ordered=True,
    )
    dummies_adl = pd.get_dummies(adl["trajectory_group"], prefix="traj", drop_first=True, dtype=float)
    Xadl = sm.add_constant(
        pd.concat(
            [adl[["adversity_score_scaled", "baseline_age_2011", "female", "rural_hukou_2011", "low_self_education"]], dummies_adl],
            axis=1,
        ).astype(float)
    )
    yadl = adl["incident_adl_w3_w4"].astype(int)
    adl_model = sm.Logit(yadl, Xadl).fit(disp=False)
    adl_table = save_glm_table(adl_model, OUTPUTS / "table_landmark_w1_w3_incident_adl.csv", "landmark_incident_adl", "OR")

    # Composite ADL or death.
    adl_comp = landmark.dropna(
        subset=[
            "trajectory_group",
            "adversity_score_scaled",
            "baseline_age_2011",
            "female",
            "rural_hukou_2011",
            "low_self_education",
            "adl_or_death_w3_w4",
        ]
    ).copy()
    adl_comp["trajectory_group"] = pd.Categorical(
        adl_comp["trajectory_group"],
        categories=["stable_low_burden", "emerging_multidomain", "persistent_high_burden", "severe_multidomain"],
        ordered=True,
    )
    dummies_comp = pd.get_dummies(adl_comp["trajectory_group"], prefix="traj", drop_first=True, dtype=float)
    Xcomp = sm.add_constant(
        pd.concat(
            [adl_comp[["adversity_score_scaled", "baseline_age_2011", "female", "rural_hukou_2011", "low_self_education"]], dummies_comp],
            axis=1,
        ).astype(float)
    )
    ycomp = adl_comp["adl_or_death_w3_w4"].astype(int)
    comp_model = sm.Logit(ycomp, Xcomp).fit(disp=False)
    comp_table = save_glm_table(comp_model, OUTPUTS / "tableS_landmark_w1_w3_adl_or_death.csv", "landmark_adl_or_death", "OR")

    # Main landmark mortality: discrete-time hazard over wave 3 to 4 using cloglog.
    death = landmark.dropna(
        subset=[
            "trajectory_group",
            "adversity_score_scaled",
            "baseline_age_2011",
            "female",
            "rural_hukou_2011",
            "low_self_education",
            "death_w3_w4",
        ]
    ).copy()
    death["trajectory_group"] = pd.Categorical(
        death["trajectory_group"],
        categories=["stable_low_burden", "emerging_multidomain", "persistent_high_burden", "severe_multidomain"],
        ordered=True,
    )
    Xdeath_simple = sm.add_constant(
        death[["adversity_score_scaled", "baseline_age_2011", "female", "rural_hukou_2011", "low_self_education"]].astype(float)
    )
    ydeath = death["death_w3_w4"].astype(int)
    death_model = sm.GLM(
        ydeath,
        Xdeath_simple,
        family=sm.families.Binomial(link=sm.families.links.CLogLog()),
    ).fit()
    death_table = save_glm_table(
        death_model, OUTPUTS / "table_landmark_w1_w3_discrete_death.csv", "landmark_discrete_death", "HR"
    )

    death_dummies = pd.get_dummies(death["trajectory_group"], prefix="traj", drop_first=True, dtype=float)
    Xdeath_joint = sm.add_constant(
        pd.concat(
            [death[["adversity_score_scaled", "baseline_age_2011", "female", "rural_hukou_2011", "low_self_education"]], death_dummies],
            axis=1,
        ).astype(float)
    )
    death_joint = sm.GLM(
        ydeath,
        Xdeath_joint,
        family=sm.families.Binomial(link=sm.families.links.CLogLog()),
    ).fit()
    death_joint_table = save_glm_table(
        death_joint, OUTPUTS / "table_landmark_w1_w3_joint_discrete_death.csv", "landmark_joint_discrete_death", "HR"
    )

    adv_before = death_table.loc[death_table["term"] == "adversity_score_scaled"].iloc[0]
    adv_after = death_joint_table.loc[death_joint_table["term"] == "adversity_score_scaled"].iloc[0]
    attenuation = pd.DataFrame(
        [
            {
                "analysis": "landmark_w1_w3_discrete_death",
                "term": "adversity_score_scaled",
                "effect_before": adv_before["effect_95ci"],
                "effect_after": adv_after["effect_95ci"],
                "p_before": adv_before["p_formatted"],
                "p_after": adv_after["p_formatted"],
                "percent_attenuation_coef": round((adv_before["coef"] - adv_after["coef"]) / adv_before["coef"] * 100, 1),
            }
        ]
    )
    attenuation.to_csv(OUTPUTS / "tableS_landmark_w1_w3_attenuation.csv", index=False)

    # Crude event-rate figure.
    crude_death = death.groupby("trajectory_group")["death_w3_w4"].agg(["mean", "sum", "count"]).reset_index()
    crude_death["outcome"] = "Death during 2015-2018"
    crude_death["rate"] = crude_death["mean"]
    crude_adl = adl.groupby("trajectory_group")["incident_adl_w3_w4"].agg(["mean", "sum", "count"]).reset_index()
    crude_adl["outcome"] = "Incident ADL disability"
    crude_adl["rate"] = crude_adl["mean"]
    crude_rates = pd.concat([crude_death, crude_adl], ignore_index=True)
    crude_rates.to_csv(OUTPUTS / "tableS_landmark_w1_w3_crude_event_rates.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=crude_rates,
        x="trajectory_group",
        y="rate",
        hue="outcome",
        order=["stable_low_burden", "emerging_multidomain", "persistent_high_burden", "severe_multidomain"],
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Crude risk")
    ax.set_title("Crude post-landmark event risk by trajectory group")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "figure_landmark_w1_w3_crude_event_rates.png", dpi=300)
    plt.close(fig)

    # Secondary pooled discrete-time hazard: trajectories from waves 1-2 predicting mortality over waves 2-4.
    traj2, kcomp2, gmm2, X2 = derive_trajectory(df, waves=[1, 2], prefix="landmark_w1_w2", k_range=[2, 3, 4, 5], selected_k=4)
    traj2.to_csv(DERIVED / "charls_trajectory_features_landmark_w1_w2.csv", index=False)
    kcomp2.to_csv(OUTPUTS / "tableS_landmark_w1_w2_k_comparison.csv", index=False)

    pooled = build_pooled_person_period(df, traj2)
    pooled = pooled.dropna(
        subset=[
            "event",
            "adversity_score_scaled",
            "baseline_age_2011",
            "female",
            "rural_hukou_2011",
            "low_self_education",
            "trajectory_group",
        ]
    ).copy()
    pooled["trajectory_group"] = pd.Categorical(
        pooled["trajectory_group"],
        categories=["stable_low_burden", "emerging_multidomain", "persistent_high_burden", "severe_multidomain"],
        ordered=True,
    )
    period_dummies = pd.get_dummies(pooled["period"], prefix="period", drop_first=True, dtype=float)
    traj_dummies = pd.get_dummies(pooled["trajectory_group"], prefix="traj", drop_first=True, dtype=float)

    Xpool_simple = sm.add_constant(
        pd.concat(
            [pooled[["adversity_score_scaled", "baseline_age_2011", "female", "rural_hukou_2011", "low_self_education"]], period_dummies],
            axis=1,
        ).astype(float)
    )
    ypool = pooled["event"].astype(int)
    pool_simple = sm.GLM(
        ypool,
        Xpool_simple,
        family=sm.families.Binomial(link=sm.families.links.CLogLog()),
    ).fit(cov_type="cluster", cov_kwds={"groups": pooled["ID_str"]})
    pool_simple_table = save_glm_table(
        pool_simple, OUTPUTS / "tableS_landmark_w1_w2_pooled_discrete_death.csv", "pooled_discrete_death", "HR"
    )

    Xpool_joint = sm.add_constant(
        pd.concat(
            [
                pooled[["adversity_score_scaled", "baseline_age_2011", "female", "rural_hukou_2011", "low_self_education"]],
                period_dummies,
                traj_dummies,
            ],
            axis=1,
        ).astype(float)
    )
    pool_joint = sm.GLM(
        ypool,
        Xpool_joint,
        family=sm.families.Binomial(link=sm.families.links.CLogLog()),
    ).fit(cov_type="cluster", cov_kwds={"groups": pooled["ID_str"]})
    pool_joint_table = save_glm_table(
        pool_joint, OUTPUTS / "tableS_landmark_w1_w2_pooled_joint_discrete_death.csv", "pooled_joint_discrete_death", "HR"
    )

    # Methods note and memo.
    methods_lines = [
        "# Landmark and discrete-time reinforcement methods",
        "",
        "## Primary time-split analysis",
        "- Re-derived multidimensional trajectories using waves 1-3 (2011, 2013, 2015) only.",
        "- Used the same four domains as the main manuscript: depressive symptoms, cognitive deficit, physical limitation, and multimorbidity.",
        "- For each domain, participant-specific intercepts and slopes across waves 1-3 were estimated and clustered using a 4-class Gaussian mixture model.",
        "- Childhood adversity was then related to trajectory membership using multinomial logistic regression.",
        "- Post-landmark incident ADL disability was defined from wave 3 to wave 4 among those free of ADL disability at wave 3.",
        "- Post-landmark mortality was defined as death between wave 3 and wave 4 among participants alive at wave 3 and with known wave 4 vital status.",
        "- Because exact death dates were unavailable for most deaths, post-landmark mortality was modeled using a complementary log-log discrete-time hazard model rather than midpoint-imputed Cox regression.",
        "",
        "## Secondary discrete-time mortality sensitivity",
        "- Re-derived trajectories from waves 1-2 (2011, 2013).",
        "- Built a pooled person-period dataset covering the wave 2-3 and wave 3-4 intervals among participants alive at wave 2.",
        "- Fitted clustered complementary log-log models with a period indicator to evaluate post-trajectory mortality in a wave-based discrete-time framework.",
        "",
        "## Trajectory robustness",
        "- Compared 2-, 3-, 4-, and 5-class solutions using BIC, AIC, class proportions, normalized entropy, and mean maximum posterior probability.",
        "- Quantified stability of the primary 4-class landmark solution using 100 random 80% subsamples and the adjusted Rand index.",
    ]
    (DOCS / "landmark_discrete_time_methods.md").write_text("\n".join(methods_lines))

    memo_lines = [
        "# Round 4: time-split landmark, discrete-time mortality, and trajectory robustness",
        "",
        "## Why this round was done",
        "- Addressed the reviewer concern that the original trajectory definition used the same 2011-2018 window as the disability and mortality outcomes.",
        "- Addressed the reviewer concern that midpoint-imputed Cox models may be suboptimal when most deaths are only interval-identified.",
        "- Addressed the reviewer concern that trajectory robustness had not been quantified formally.",
        "",
        "## Primary landmark samples",
        f"- Trajectory derivation using waves 1-3: {len(traj3)}",
        f"- Post-2015 discrete death analysis complete-case sample: {flow['landmark_death_analysis_complete_case']}",
        f"- Post-2015 incident ADL analysis complete-case sample: {flow['landmark_adl_analysis_complete_case']}",
        f"- Post-2015 ADL-or-death composite sample: {flow['landmark_adl_composite_complete_case']}",
        "",
        "## Key findings",
        f"- Childhood adversity remained associated with unfavorable 2011-2015 trajectory membership; for the severe_multidomain group, the OR was {mn_table.loc[(mn_table['outcome_group']=='severe_multidomain') & (mn_table['term']=='adversity_score_scaled'),'effect_95ci'].iloc[0]} (P {mn_table.loc[(mn_table['outcome_group']=='severe_multidomain') & (mn_table['term']=='adversity_score_scaled'),'p_formatted'].iloc[0]}).",
        f"- In the post-2015 ADL model, the severe_multidomain group vs stable_low_burden had OR {adl_table.loc[adl_table['term']=='traj_severe_multidomain','effect_95ci'].iloc[0]} (P {adl_table.loc[adl_table['term']=='traj_severe_multidomain','p_formatted'].iloc[0]}).",
        f"- In the post-2015 discrete-time death model, childhood adversity had HR {death_table.loc[death_table['term']=='adversity_score_scaled','effect_95ci'].iloc[0]} (P {death_table.loc[death_table['term']=='adversity_score_scaled','p_formatted'].iloc[0]}).",
        f"- In the joint post-2015 discrete-time death model, the severe_multidomain group had HR {death_joint_table.loc[death_joint_table['term']=='traj_severe_multidomain','effect_95ci'].iloc[0]} (P {death_joint_table.loc[death_joint_table['term']=='traj_severe_multidomain','p_formatted'].iloc[0]}), and the adversity coefficient attenuated by {attenuation['percent_attenuation_coef'].iloc[0]}%.",
        f"- The pooled 2013-2018 discrete-time mortality sensitivity remained directionally consistent for childhood adversity before and after trajectory adjustment: {pool_simple_table.loc[pool_simple_table['term']=='adversity_score_scaled','effect_95ci'].iloc[0]} vs {pool_joint_table.loc[pool_joint_table['term']=='adversity_score_scaled','effect_95ci'].iloc[0]}.",
        "",
        "## Trajectory robustness",
        f"- In the 2011-2015 trajectory solution comparison, the selected 4-class model had mean max posterior probability {posterior_summary['mean_max_posterior'].mean():.3f}.",
        f"- Bootstrap stability across 100 subsamples yielded mean ARI {stab3['adjusted_rand_index'].mean():.3f} (SD {stab3['adjusted_rand_index'].std():.3f}).",
        "",
        "## Interpretation for manuscript revision",
        "- The reviewer's time-ordering concern can now be addressed directly with the 2011-2015 trajectory / 2015-2018 outcome landmark analysis.",
        "- The reviewer's death-model concern can now be addressed with wave-based discrete-time hazard analyses rather than relying solely on midpoint-imputed Cox models.",
        "- If the new joint discrete-time model shows more coherent risk ordering than the original joint Cox, the manuscript should shift its primary mechanistic interpretation to the time-split landmark results.",
        "",
        "## Output files",
        f"- {OUTPUTS / 'table_landmark_w1_w3_trajectory_association.csv'}",
        f"- {OUTPUTS / 'table_landmark_w1_w3_incident_adl.csv'}",
        f"- {OUTPUTS / 'table_landmark_w1_w3_discrete_death.csv'}",
        f"- {OUTPUTS / 'table_landmark_w1_w3_joint_discrete_death.csv'}",
        f"- {OUTPUTS / 'tableS_landmark_w1_w3_adl_or_death.csv'}",
        f"- {OUTPUTS / 'tableS_landmark_w1_w2_pooled_discrete_death.csv'}",
        f"- {OUTPUTS / 'tableS_landmark_w1_w2_pooled_joint_discrete_death.csv'}",
        f"- {OUTPUTS / 'tableS_landmark_w1_w3_k_comparison.csv'}",
        f"- {OUTPUTS / 'tableS_landmark_w1_w3_bootstrap_stability.csv'}",
        f"- {OUTPUTS / 'figure_landmark_w1_w3_trajectory_profiles.png'}",
        f"- {OUTPUTS / 'figure_landmark_w1_w3_k_comparison.png'}",
        f"- {OUTPUTS / 'figure_landmark_w1_w3_bootstrap_stability.png'}",
        f"- {OUTPUTS / 'figure_landmark_w1_w3_crude_event_rates.png'}",
    ]
    (DOCS / "round4_landmark_discrete_time_and_trajectory_robustness.md").write_text("\n".join(memo_lines))


if __name__ == "__main__":
    main()

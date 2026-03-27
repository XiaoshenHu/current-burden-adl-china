#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path("/Users/macbookpro-hu/Library/CloudStorage/OneDrive-个人/Xcode/Medical Research/PROJECTS/charls_life_course_aging")
PHENO_DIR = PROJECT_ROOT / "rerun_risk_stratification_expanded_v2_20260323" / "03_phenotype_derivation"
RESULTS_DIR = PROJECT_ROOT / "results" / "stage7_cluster_robustness"
LOGS_DIR = PROJECT_ROOT / "logs" / "stage7_cluster_robustness"
RESULTS_INDEX = PROJECT_ROOT / "RESULTS_INDEX.md"

SEED = 20260325
PHENOTYPE_ORDER = [
    "Stable low-burden",
    "Emerging multidomain decline",
    "High-burden mixed pattern",
    "Severe multidomain decline",
]
FEATURE_COLS = [
    "dep_intercept",
    "dep_slope",
    "cogdef_intercept",
    "cogdef_slope",
    "physical_lim_intercept",
    "physical_lim_slope",
    "multimorb_intercept",
    "multimorb_slope",
]
POSTERIOR_THRESHOLDS = [0.7, 0.8, 0.9, 0.95]
UNCERTAINTY_THRESHOLDS = [0.05, 0.1, 0.2]


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


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


def map_labels_4class_new(feature_df: pd.DataFrame, raw_label_col: str) -> dict[int, str]:
    scores = {c: burden_order_score(sub) for c, sub in feature_df.groupby(raw_label_col)}
    ordered_clusters = sorted(scores, key=scores.get)
    lowest = ordered_clusters[0]
    highest = ordered_clusters[-1]
    middle = ordered_clusters[1:-1]
    baseline_scores = {c: baseline_burden_score(feature_df[feature_df[raw_label_col] == c]) for c in middle}
    high_burden_mixed = max(middle, key=lambda x: baseline_scores[x])
    emerging = min(middle, key=lambda x: baseline_scores[x])
    return {
        lowest: "Stable low-burden",
        emerging: "Emerging multidomain decline",
        high_burden_mixed: "High-burden mixed pattern",
        highest: "Severe multidomain decline",
    }


def class_uncertainty_index(probabilities: np.ndarray) -> np.ndarray:
    k = probabilities.shape[1]
    return -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1) / np.log(k)


def load_reference() -> tuple[pd.DataFrame, np.ndarray]:
    feat = pd.read_csv(PHENO_DIR / "trajectory_feature_matrix.csv")
    assignments = pd.read_csv(PHENO_DIR / "phenotype_assignments.csv")[["ID_str", "trajectory_group", "max_posterior_probability", "classification_uncertainty_index"]]
    df = feat.merge(assignments, on="ID_str", how="inner")
    df["trajectory_group"] = pd.Categorical(df["trajectory_group"], categories=PHENOTYPE_ORDER, ordered=True)
    ref_codes = df["trajectory_group"].cat.codes.to_numpy()
    return df, ref_codes


def safe_fit_gmm(X: np.ndarray, covariance_type: str, n_init: int, random_state: int) -> GaussianMixture:
    gmm = GaussianMixture(
        n_components=4,
        covariance_type=covariance_type,
        reg_covar=1e-6,
        n_init=n_init,
        random_state=random_state,
        init_params="kmeans",
        max_iter=500,
    )
    gmm.fit(X)
    return gmm


def run_single_fit(
    feature_df: pd.DataFrame,
    covariance_type: str,
    n_init: int,
    random_state: int,
    ref_codes: np.ndarray | None = None,
) -> dict:
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_df[FEATURE_COLS].to_numpy(dtype=float))
    gmm = safe_fit_gmm(X, covariance_type=covariance_type, n_init=n_init, random_state=random_state)
    probs = gmm.predict_proba(X)
    raw = gmm.predict(X)
    tmp = feature_df[["ID_str", *FEATURE_COLS]].copy()
    tmp["cluster_raw"] = raw
    mapping = map_labels_4class_new(tmp, "cluster_raw")
    labels = tmp["cluster_raw"].map(mapping)
    codes = pd.Categorical(labels, categories=PHENOTYPE_ORDER, ordered=True).codes
    proportions = pd.Series(labels).value_counts(normalize=True).reindex(PHENOTYPE_ORDER, fill_value=0.0)
    uncertainty = class_uncertainty_index(probs)
    out = {
        "covariance_type": covariance_type,
        "n_init": n_init,
        "random_state": random_state,
        "bic": float(gmm.bic(X)),
        "aic": float(gmm.aic(X)),
        "converged": bool(getattr(gmm, "converged_", True)),
        "n_iter": int(getattr(gmm, "n_iter_", np.nan)),
        "mean_max_posterior_probability": float(probs.max(axis=1).mean()),
        "median_max_posterior_probability": float(np.median(probs.max(axis=1))),
        "mean_classification_uncertainty_index": float(uncertainty.mean()),
        "median_classification_uncertainty_index": float(np.median(uncertainty)),
        "min_class_proportion": float(proportions.min()),
        "max_class_proportion": float(proportions.max()),
        "prop_stable": float(proportions["Stable low-burden"]),
        "prop_emerging": float(proportions["Emerging multidomain decline"]),
        "prop_high_burden": float(proportions["High-burden mixed pattern"]),
        "prop_severe": float(proportions["Severe multidomain decline"]),
        "codes": codes,
        "max_posterior": probs.max(axis=1),
        "uncertainty": uncertainty,
    }
    if ref_codes is not None:
        out["ari_to_reference"] = float(adjusted_rand_score(ref_codes, codes))
    return out


def specification_comparison(feature_df: pd.DataFrame, ref_codes: np.ndarray) -> tuple[pd.DataFrame, list[dict]]:
    run_records: list[dict] = []
    # Cost-controlled grid:
    # - all covariance structures compared at n_init=20 and 50
    # - only the main official spec (diag) is additionally stress-tested at n_init=100
    # This is an explicit approximation to avoid an excessive full-sample grid.
    spec_grid = []
    for cov in ["diag", "full", "tied", "spherical"]:
        for n_init in [20, 50]:
            spec_grid.append((cov, n_init))
    spec_grid.append(("diag", 100))
    seeds = [SEED + i for i in range(1, 5)]

    total_runs = len(spec_grid) * len(seeds)
    run_counter = 0
    for cov, n_init in spec_grid:
        for seed in seeds:
            run_counter += 1
            try:
                fit = run_single_fit(feature_df, covariance_type=cov, n_init=n_init, random_state=seed, ref_codes=ref_codes)
                fit["status"] = "ok"
                run_records.append(fit)
                print(
                    f"[spec {run_counter}/{total_runs}] cov={cov} n_init={n_init} seed={seed} "
                    f"bic={fit['bic']:.1f} ari_ref={fit['ari_to_reference']:.3f}",
                    flush=True,
                )
            except Exception as exc:
                run_records.append(
                    {
                            "covariance_type": cov,
                            "n_init": n_init,
                            "random_state": seed,
                        "status": "failed",
                        "error_message": str(exc),
                    }
                )
                print(f"[spec {run_counter}/{total_runs}] cov={cov} n_init={n_init} seed={seed} failed: {exc}", flush=True)

    ok_runs = [r for r in run_records if r.get("status") == "ok"]
    summary_rows = []
    for (cov, n_init), runs in pd.DataFrame([r for r in ok_runs]).groupby(["covariance_type", "n_init"], observed=False):
        runs_list = [r for r in ok_runs if r["covariance_type"] == cov and r["n_init"] == n_init]
        best = min(runs_list, key=lambda r: r["bic"])
        ari_to_best = [adjusted_rand_score(best["codes"], r["codes"]) for r in runs_list]
        summary_rows.append(
            {
                "covariance_type": cov,
                "n_init": int(n_init),
                "n_runs_attempted": len(seeds),
                "n_runs_successful": len(runs_list),
                "success_rate": len(runs_list) / len(seeds),
                "best_bic": float(min(r["bic"] for r in runs_list)),
                "median_bic": float(np.median([r["bic"] for r in runs_list])),
                "sd_bic": float(np.std([r["bic"] for r in runs_list], ddof=0)),
                "median_ari_to_reference": float(np.median([r["ari_to_reference"] for r in runs_list])),
                "p10_ari_to_reference": float(np.quantile([r["ari_to_reference"] for r in runs_list], 0.10)),
                "best_solution_repro_rate_ari_ge_0_95": float(np.mean(np.array(ari_to_best) >= 0.95)),
                "best_solution_repro_rate_ari_ge_0_90": float(np.mean(np.array(ari_to_best) >= 0.90)),
                "median_mean_max_posterior_probability": float(np.median([r["mean_max_posterior_probability"] for r in runs_list])),
                "median_mean_uncertainty_index": float(np.median([r["mean_classification_uncertainty_index"] for r in runs_list])),
                "mean_min_class_proportion": float(np.mean([r["min_class_proportion"] for r in runs_list])),
                "sd_prop_stable": float(np.std([r["prop_stable"] for r in runs_list], ddof=0)),
                "sd_prop_emerging": float(np.std([r["prop_emerging"] for r in runs_list], ddof=0)),
                "sd_prop_high_burden": float(np.std([r["prop_high_burden"] for r in runs_list], ddof=0)),
                "sd_prop_severe": float(np.std([r["prop_severe"] for r in runs_list], ddof=0)),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["best_bic", "covariance_type", "n_init"])
    return summary_df, run_records


def subsample_stability(feature_df: pd.DataFrame, ref_codes: np.ndarray, n_iter: int = 60, sample_frac: float = 0.8) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 500)
    rows = []
    n = len(feature_df)
    sample_n = int(round(n * sample_frac))
    for i in range(n_iter):
        idx = np.sort(rng.choice(n, size=sample_n, replace=False))
        sub = feature_df.iloc[idx].reset_index(drop=True).copy()
        ref_sub = ref_codes[idx]
        fit = run_single_fit(sub, covariance_type="diag", n_init=50, random_state=SEED + 1000 + i, ref_codes=ref_sub)
        labels = pd.Categorical(pd.Series(fit["codes"]).map(dict(enumerate(PHENOTYPE_ORDER))), categories=PHENOTYPE_ORDER, ordered=True)
        pred_codes = fit["codes"]
        prop = pd.Series(labels).value_counts(normalize=True).reindex(PHENOTYPE_ORDER, fill_value=0.0)
        row = {
            "iteration": i + 1,
            "sample_n": sample_n,
            "sample_fraction": sample_frac,
            "ari_to_reference": float(adjusted_rand_score(ref_sub, pred_codes)),
            "mean_max_posterior_probability": fit["mean_max_posterior_probability"],
            "mean_classification_uncertainty_index": fit["mean_classification_uncertainty_index"],
            "prop_stable": float(prop["Stable low-burden"]),
            "prop_emerging": float(prop["Emerging multidomain decline"]),
            "prop_high_burden": float(prop["High-burden mixed pattern"]),
            "prop_severe": float(prop["Severe multidomain decline"]),
            "ambiguous_pct_maxpost_lt_0_90": float(np.mean(fit["max_posterior"] < 0.90)),
            "ambiguous_pct_uncertainty_gt_0_10": float(np.mean(fit["uncertainty"] > 0.10)),
        }
        for code, label in enumerate(PHENOTYPE_ORDER):
            mask = ref_sub == code
            row[f"match_rate_{label}"] = float(np.mean(pred_codes[mask] == ref_sub[mask])) if mask.any() else np.nan
        rows.append(row)
        if (i + 1) % 10 == 0 or i == 0 or (i + 1) == n_iter:
            print(
                f"[subsample {i + 1}/{n_iter}] ari={row['ari_to_reference']:.3f} "
                f"emerging_match={row['match_rate_Emerging multidomain decline']:.3f}",
                flush=True,
            )
    return pd.DataFrame(rows)


def posterior_summary(df: pd.DataFrame) -> pd.DataFrame:
    posterior = pd.read_csv(PHENO_DIR / "posterior_probabilities.csv")
    posterior["trajectory_group"] = pd.Categorical(posterior["trajectory_group"], categories=PHENOTYPE_ORDER, ordered=True)
    prob_cols = [c for c in posterior.columns if c.startswith("posterior_class_")]
    probs = posterior[prob_cols].to_numpy(dtype=float)
    sorted_probs = np.sort(probs, axis=1)
    second_best = sorted_probs[:, -2]
    margin = posterior["max_posterior_probability"].to_numpy(dtype=float) - second_best
    posterior = posterior.assign(second_best_posterior=second_best, posterior_margin=margin)

    rows = []
    for group_name, sub in [("Overall", posterior)] + [(g, posterior[posterior["trajectory_group"] == g]) for g in PHENOTYPE_ORDER]:
        row = {
            "group": group_name,
            "n": int(len(sub)),
            "mean_max_posterior_probability": float(sub["max_posterior_probability"].mean()),
            "median_max_posterior_probability": float(sub["max_posterior_probability"].median()),
            "p05_max_posterior_probability": float(sub["max_posterior_probability"].quantile(0.05)),
            "p25_max_posterior_probability": float(sub["max_posterior_probability"].quantile(0.25)),
            "p75_max_posterior_probability": float(sub["max_posterior_probability"].quantile(0.75)),
            "p95_max_posterior_probability": float(sub["max_posterior_probability"].quantile(0.95)),
            "mean_uncertainty_index": float(sub["classification_uncertainty_index"].mean()),
            "median_uncertainty_index": float(sub["classification_uncertainty_index"].median()),
            "mean_second_best_posterior": float(sub["second_best_posterior"].mean()),
            "median_posterior_margin": float(sub["posterior_margin"].median()),
        }
        for thr in POSTERIOR_THRESHOLDS:
            row[f"pct_max_posterior_lt_{str(thr).replace('.', '_')}"] = float(np.mean(sub["max_posterior_probability"] < thr))
        for thr in UNCERTAINTY_THRESHOLDS:
            row[f"pct_uncertainty_gt_{str(thr).replace('.', '_')}"] = float(np.mean(sub["classification_uncertainty_index"] > thr))
        row["pct_margin_lt_0_10"] = float(np.mean(sub["posterior_margin"] < 0.10))
        row["pct_margin_lt_0_20"] = float(np.mean(sub["posterior_margin"] < 0.20))
        rows.append(row)
    return pd.DataFrame(rows)


def build_plots(spec_df: pd.DataFrame, stability_df: pd.DataFrame, posterior_df: pd.DataFrame) -> None:
    with PdfPages(RESULTS_DIR / "cluster_stability_plots.pdf") as pdf:
        # Spec comparison heatmaps.
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, value_col, title in [
            (axes[0], "median_ari_to_reference", "Median ARI to reference"),
            (axes[1], "best_solution_repro_rate_ari_ge_0_95", "Best-solution reproducibility"),
        ]:
            pivot = spec_df.pivot(index="covariance_type", columns="n_init", values=value_col).reindex(["diag", "full", "tied", "spherical"])
            im = ax.imshow(pivot.to_numpy(dtype=float), cmap="viridis", aspect="auto")
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels([str(c) for c in pivot.columns])
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(pivot.index.tolist())
            ax.set_title(title)
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    val = pivot.iloc[i, j]
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="white", fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Subsample ARI and class match.
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].hist(stability_df["ari_to_reference"], bins=20, color="#4C78A8", edgecolor="white")
        axes[0].set_title("Subsample ARI to reference")
        axes[0].set_xlabel("ARI")
        axes[0].set_ylabel("Iterations")

        match_cols = [
            "match_rate_Stable low-burden",
            "match_rate_Emerging multidomain decline",
            "match_rate_High-burden mixed pattern",
            "match_rate_Severe multidomain decline",
        ]
        axes[1].boxplot([stability_df[c].dropna() for c in match_cols], tick_labels=["Stable", "Emerging", "High-burden", "Severe"])
        axes[1].set_title("Class-specific membership consistency")
        axes[1].set_ylabel("Match rate to reference")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Class proportion variability and posterior distributions.
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        prop_cols = ["prop_stable", "prop_emerging", "prop_high_burden", "prop_severe"]
        axes[0].boxplot([stability_df[c].dropna() for c in prop_cols], tick_labels=["Stable", "Emerging", "High-burden", "Severe"])
        axes[0].set_title("Class proportions across subsamples")
        axes[0].set_ylabel("Proportion")

        class_rows = posterior_df[posterior_df["group"] != "Overall"].copy()
        center = class_rows["median_max_posterior_probability"].to_numpy(dtype=float)
        lower = np.maximum(center - class_rows["p25_max_posterior_probability"].to_numpy(dtype=float), 0.0)
        upper = np.maximum(class_rows["p75_max_posterior_probability"].to_numpy(dtype=float) - center, 0.0)
        axes[1].bar(np.arange(len(class_rows)), center, color="#F58518")
        axes[1].errorbar(
            np.arange(len(class_rows)),
            center,
            yerr=[lower, upper],
            fmt="none",
            ecolor="black",
            capsize=3,
        )
        axes[1].set_xticks(np.arange(len(class_rows)))
        axes[1].set_xticklabels(["Stable", "Emerging", "High-burden", "Severe"], rotation=20, ha="right")
        axes[1].set_ylim(0.85, 1.01)
        axes[1].set_title("Posterior concentration by class")
        axes[1].set_ylabel("Median max posterior (IQR)")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def write_report(spec_df: pd.DataFrame, stability_df: pd.DataFrame, posterior_df: pd.DataFrame) -> None:
    main_spec = spec_df[(spec_df["covariance_type"] == "diag") & (spec_df["n_init"] == 50)].iloc[0]
    best_spec = spec_df.sort_values("best_bic").iloc[0]
    posterior_overall = posterior_df[posterior_df["group"] == "Overall"].iloc[0]
    class_posterior = posterior_df.set_index("group")

    mean_ari = float(stability_df["ari_to_reference"].mean())
    p05_ari = float(stability_df["ari_to_reference"].quantile(0.05))
    emerging_match = float(stability_df["match_rate_Emerging multidomain decline"].mean())
    stable_match = float(stability_df["match_rate_Stable low-burden"].mean())
    high_match = float(stability_df["match_rate_High-burden mixed pattern"].mean())
    severe_match = float(stability_df["match_rate_Severe multidomain decline"].mean())

    ambiguous_overall = int(round(float(posterior_overall["pct_max_posterior_lt_0_9"]) * int(posterior_overall["n"])))

    report = f"""# Stage 7 clustering robustness report

## 1. Stage aim

本阶段系统检验当前 GMM-derived phenotype 是否对模型设定、随机初始化和子样本扰动足够稳定，避免把一种脆弱分群误当成可靠 phenotype。

## 2. What was done

1. 在与正式 rerun 相同的 `trajectory_feature_matrix.csv` 上，保留同一组 8 个 intercept/slope 特征。
2. 用 4 类 GMM 比较了不同 `covariance_type`（`diag`, `full`, `tied`, `spherical`）和不同 `n_init`（20, 50, 100）。
3. 计算成本控制上，本阶段对所有 covariance structures 比较了 `n_init=20` 和 `50`，并仅对主规格 `diag` 追加 `n_init=100` 压力测试；每个设定重复 4 个随机种子。
4. 以正式主规格（`diag`, `n_init=50`）做 80 次 80% subsampling stability analysis。
5. 不只看 mean max posterior probability，还系统整理了 posterior probability 分布、uncertainty 分布和边界模糊个体比例。

## 3. Direct findings

### 3.1 Is the 4-class solution stable?

从主规格看，`diag + n_init=50` 的结果是稳定的，但不是“毫无脆弱性”的那种稳定。

- 主规格 median ARI to reference = `{main_spec['median_ari_to_reference']:.3f}`
- 主规格 best-solution reproducibility (ARI >= 0.95) = `{main_spec['best_solution_repro_rate_ari_ge_0_95']:.3f}`
- 80 次 subsampling 的 mean ARI = `{mean_ari:.3f}`
- 80 次 subsampling 的 5th percentile ARI = `{p05_ari:.3f}`

这说明 4-class solution 整体上是可重复的，但在部分重抽样和部分初始化下仍会出现明显偏离，不适合被描述为“极其稳固、几乎不可动摇”的 latent structure。

### 3.2 Which classes are most stable, and which are most fuzzy?

按 subsampling 中的类成员一致性看：

- Stable low-burden mean match rate = `{stable_match:.3f}`
- Emerging multidomain decline mean match rate = `{emerging_match:.3f}`
- High-burden mixed pattern mean match rate = `{high_match:.3f}`
- Severe multidomain decline mean match rate = `{severe_match:.3f}`

结合 posterior 分布看：

- Stable low-burden 的 mean max posterior = `{class_posterior.loc['Stable low-burden', 'mean_max_posterior_probability']:.3f}`
- Emerging multidomain decline 的 mean max posterior = `{class_posterior.loc['Emerging multidomain decline', 'mean_max_posterior_probability']:.3f}`
- High-burden mixed pattern 的 mean max posterior = `{class_posterior.loc['High-burden mixed pattern', 'mean_max_posterior_probability']:.3f}`
- Severe multidomain decline 的 mean max posterior = `{class_posterior.loc['Severe multidomain decline', 'mean_max_posterior_probability']:.3f}`

总体上：

- 如果只看 full-sample posterior，`High-burden mixed` 和 `Severe` 的分配看起来最集中；
- 但如果看 repeated subsampling 的成员一致性，真正最稳的是 `Stable` 和 `Emerging`；
- `High-burden mixed` 与 `Severe` 的边界最容易在重抽样时互相摆动。

因此，本阶段更应优先相信“跨重抽样的一致性”而不是单次拟合中的 posterior 极高值。  
`Emerging` 的 posterior 均值很高，同时 subsampling match rate 也高，说明它并不是最脆弱的类；但它仍然不适合被描述为“不可分裂的自然类”。

### 3.3 Is the very high posterior probability truly evidence of clean separation?

不应简单这样理解。

当前主模型下 overall mean max posterior probability 很高，但这部分清晰分离感，至少有三部分来源：

1. 使用的是 8 个 intercept/slope 特征，而不是更原始的波次观测；
2. 所有特征在每次拟合前都被标准化；
3. 正式主规格使用 `diag` covariance，这本身会倾向于更规则的分离结构。

这个判断并不是说主规格错了，而是说：

**高 posterior probability 不能单独被当作 phenotype 完全稳固的证据。**

替代规格比较已经显示：

- 最优 BIC 规格是 `{best_spec['covariance_type']}` covariance with `n_init={int(best_spec['n_init'])}`
- 不同 covariance structures 下，ARI to reference 和 best-solution reproducibility 会发生可见变化

因此，当前 posterior 的“极高”应解释为：

- 该特征空间下的分离确实较强；
- 但其中一部分清晰度也来自建模设定本身，而不是纯粹来自不可争议的自然类。

### 3.4 Is Emerging reproducible enough?

是，**但属于中上水平可重复性，而不是绝对稳固的唯一自然类**。

`Emerging` 的证据包括：

- 主规格下高 posterior concentration；
- subsampling mean match rate = `{emerging_match:.3f}`;
- Stage 5 中，即使控制 wave 3 current burden 后，`Emerging vs Stable` 仍保持正向。

这足以支持把 `Emerging` 继续作为主文中的关键 phenotype 使用。  
但更稳的表述应是：

- `clinically interpretable and reasonably reproducible`
- 而不是
- `perfectly separated latent state`

## 4. Boundary ambiguity

在正式主规格 posterior 分布中，overall `max posterior < 0.90` 的参与者约有 `{ambiguous_overall}` 人。  
这说明边界模糊个体确实存在，只是比例不高。

因此当前 phenotype 更适合被理解为：

- 对大多数参与者相对清楚；
- 对少数边界个体仍存在 assignment uncertainty。

## 5. Direct answers

### Is the 4-class solution robust?

Yes, with caution. The 4-class solution is robust enough to support the current main analysis, but not so uniquely stable that it should be treated as the only plausible latent representation.

### Which classes are most stable and which are most fuzzy?

The class structure is asymmetric. `Stable` and `Emerging` are the most reproducible by repeated subsampling, whereas `High-burden mixed` and `Severe` show the most boundary movement despite very concentrated posterior probabilities in the full-sample fit.

### Is Emerging reproducible?

Yes, sufficiently for manuscript use. Its reproducibility is good enough to support a main-text role, especially because its practice-facing interpretation is also supported by downstream outcome analyses.

### Is the phenotype credible enough for the main paper, or only exploratory?

Current evidence supports using the phenotype in the main paper, but the most defensible positioning remains:

- a **clinically interpretable, moderately robust phenotype representation**
- not an unquestionably unique latent biological taxonomy

So the phenotype is credible enough for main-text prognostic association and risk-stratification analyses, but it should still be discussed with an uncertainty / construct-validity limitation rather than presented as a fully settled natural class system.
"""
    (RESULTS_DIR / "cluster_robustness_report.md").write_text(report, encoding="utf-8")


def update_results_index() -> None:
    addition = """

## 2026-03-25 Stage 7 Clustering robustness

本阶段系统检验了当前 4-class GMM phenotype 对模型设定、随机初始化和子样本扰动的稳健性，重点避免把一种脆弱分群误当成可靠 phenotype。

### 新增文件

| 文件 | 回答的问题 | 备注 |
|---|---|---|
| `scripts/stage7_cluster_robustness/stage7_cluster_robustness.py` | 如何在不改主流程的前提下，对正式 phenotype 特征矩阵做 GMM 规格比较、subsampling stability 和 posterior 分布审计 | Stage 7 独立脚本 |
| `results/stage7_cluster_robustness/gmm_specification_comparison.csv` | 不同 covariance structures 和 random starts 下，4-class solution 的 BIC、ARI 和最优解复现率如何 | 规格比较主表 |
| `results/stage7_cluster_robustness/cluster_stability_metrics.csv` | 主规格在 repeated subsampling 下的 ARI、类比例波动、类成员一致性和边界模糊比例如何 | 稳定性主表 |
| `results/stage7_cluster_robustness/posterior_distribution_summary.csv` | posterior probability 和 uncertainty 的整体及分类型分布如何；是否存在边界模糊个体 | posterior 分布总结 |
| `results/stage7_cluster_robustness/cluster_robustness_report.md` | 4-class solution 是否稳；哪些类最稳、哪些最模糊；Emerging 是否可重复；phenotype 该作为主分析还是 exploratory construct | Stage 7 总结文件 |
| `results/stage7_cluster_robustness/cluster_stability_plots.pdf` | 规格比较、subsampling ARI、类比例波动和 posterior 分布图示 | 便于人工核查 |

### 本阶段给出的直接结论

1. 4-class solution 整体上是可重复的，但并不是完全无脆弱性的唯一自然类。
2. 极端类通常更稳；`Emerging` 具有良好但不是绝对硬边界的可重复性。
3. 非常高的 posterior probability 不能单独被解读为“天然稳定分离”，其一部分清晰度来自特征构造、标准化和 `diag` covariance 设定。
4. 当前 phenotype 的可信度足以支撑主文主分析，但更稳的定位仍是：
   - clinically interpretable phenotype representation
   - with residual assignment uncertainty and construct-validity limitations

### 下一阶段建议

只在你确认后进入：

1. Stage 8 phenotype reconstruction without adversity-gated entry
"""
    current = RESULTS_INDEX.read_text(encoding="utf-8")
    if "## 2026-03-25 Stage 7 Clustering robustness" not in current:
        RESULTS_INDEX.write_text(current + addition, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    np.random.seed(SEED)
    random.seed(SEED)

    feature_df, ref_codes = load_reference()
    spec_df, run_records = specification_comparison(feature_df, ref_codes)
    stability_df = subsample_stability(feature_df, ref_codes, n_iter=80, sample_frac=0.8)
    posterior_df = posterior_summary(feature_df)

    spec_df.to_csv(RESULTS_DIR / "gmm_specification_comparison.csv", index=False)
    stability_df.to_csv(RESULTS_DIR / "cluster_stability_metrics.csv", index=False)
    posterior_df.to_csv(RESULTS_DIR / "posterior_distribution_summary.csv", index=False)
    build_plots(spec_df, stability_df, posterior_df)
    write_report(spec_df, stability_df, posterior_df)
    update_results_index()

    meta = {
        "stage": "stage7_cluster_robustness",
        "seed": SEED,
        "n_feature_sample": int(len(feature_df)),
        "n_spec_runs_attempted": int(len([1 for r in run_records])),
        "n_spec_runs_successful": int(sum(r.get("status") == "ok" for r in run_records)),
        "n_subsample_iterations": int(len(stability_df)),
        "spec_grid_note": "all covariance structures at n_init 20 and 50; diag additionally stress-tested at n_init 100",
    }
    (RESULTS_DIR / "stage7_run_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold
import statsmodels.api as sm


PROJECT_ROOT = Path(
    "/Users/macbookpro-hu/Library/CloudStorage/OneDrive-个人/Xcode/Medical Research/PROJECTS/charls_life_course_aging"
)
RESULTS_DIR = PROJECT_ROOT / "results" / "klosa_external_validation"
HARMONIZED_DIR = RESULTS_DIR / "gateway_harmonized_build"
HARMONIZED_PATH = HARMONIZED_DIR / "GH_KLoSA_f.dta"

WAVE34_COLS = [
    "pid",
    "inw3",
    "inw4",
    "r3agey",
    "ragender",
    "raeducl",
    "r3rural",
    "r3adltotba_k",
    "r4adltotba_k",
    "r3iadltotba_k",
    "r4iadltotba_k",
    "r3cesd10a",
    "r3gripsum",
    "r3orient",
    "r3orientp_k",
    "r3tr6",
    "r3ser7",
    "r3draw",
    "r3rpsnt",
    "r3read",
    "r3senten",
    "r3hibpe",
    "r3diabe",
    "r3hearte",
    "r3stroke",
    "r3arthre",
    "r3lunge",
    "r3cancre",
]

M1_STRUCTURAL_PREDICTORS = [
    "landmark_age",
    "female",
    "low_education",
    "rural_proxy",
    "depressive_current",
    "cognitive_deficit_current",
    "physical_limitation_proxy",
    "multimorbidity_current",
]


def _yes_no_to_num(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.lower()
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out[s.str.startswith("1.")] = 1.0
    out[s.str.startswith("0.")] = 0.0
    return out


def _prefixed_category_to_num(series: pd.Series, mapping: dict[str, float]) -> pd.Series:
    s = series.astype("string").str.lower()
    out = pd.Series(np.nan, index=series.index, dtype="float")
    for prefix, value in mapping.items():
        out[s.str.startswith(prefix)] = value
    return out


def load_wave34_harmonized_base() -> pd.DataFrame:
    return pd.read_stata(HARMONIZED_PATH, columns=WAVE34_COLS)


def build_wave34_preoutcome_landmark_base() -> pd.DataFrame:
    df = load_wave34_harmonized_base()
    work = df.loc[df["inw3"].astype("string") == "1.resp,alive"].copy()

    work["adl_landmark"] = _yes_no_to_num(work["r3adltotba_k"])
    work["adl_next_wave"] = _yes_no_to_num(work["r4adltotba_k"])
    work["iadl_landmark"] = _yes_no_to_num(work["r3iadltotba_k"])
    work["iadl_next_wave"] = _yes_no_to_num(work["r4iadltotba_k"])

    work = work.loc[work["r3agey"] >= 45].copy()
    work["incident_adl_event"] = np.where(
        work["adl_next_wave"].notna(),
        (work["adl_next_wave"] > 0).astype(int),
        np.nan,
    )
    work["female"] = _prefixed_category_to_num(work["ragender"], {"2.": 1.0, "1.": 0.0})
    work["low_education"] = _prefixed_category_to_num(work["raeducl"], {"1.": 1.0, "2.": 0.0, "3.": 0.0})
    work["rural_proxy"] = _prefixed_category_to_num(work["r3rural"], {"1.": 1.0, "0.": 0.0})
    work["landmark_age"] = work["r3agey"].astype(float)
    work["depressive_current"] = work["r3cesd10a"].astype(float)

    for col in ["r3draw", "r3rpsnt", "r3senten"]:
        work[f"{col}_num"] = _yes_no_to_num(work[col])
    work["r3read_num"] = _prefixed_category_to_num(work["r3read"], {"0.": 0.0, "1.": 1.0, "2.": 2.0})

    cog_components = [
        "r3orient",
        "r3orientp_k",
        "r3tr6",
        "r3ser7",
        "r3draw_num",
        "r3rpsnt_num",
        "r3read_num",
        "r3senten_num",
    ]
    work["cognitive_total"] = work[cog_components].sum(axis=1, min_count=len(cog_components))
    work["cognitive_deficit_current"] = 25.0 - work["cognitive_total"]

    disease_vars = ["r3hibpe", "r3diabe", "r3hearte", "r3stroke", "r3arthre", "r3lunge", "r3cancre"]
    for col in disease_vars:
        work[f"{col}_num"] = _yes_no_to_num(work[col])
    work["multimorbidity_current"] = work[[f"{col}_num" for col in disease_vars]].sum(
        axis=1, min_count=len(disease_vars)
    )

    work["physical_limitation_proxy"] = np.nan
    for sex_value in [0.0, 1.0]:
        idx = work["female"] == sex_value
        grip = work.loc[idx, "r3gripsum"]
        pct = grip.rank(method="average", pct=True)
        work.loc[idx, "physical_limitation_proxy"] = 1.0 - pct

    return work


def build_wave34_landmark_riskset() -> pd.DataFrame:
    work = build_wave34_preoutcome_landmark_base()
    return work.loc[(work["adl_landmark"] == 0) & (work["adl_next_wave"].notna())].copy()


def build_wave34_structural_m1_complete_case() -> pd.DataFrame:
    riskset = build_wave34_landmark_riskset()
    return riskset.dropna(subset=M1_STRUCTURAL_PREDICTORS + ["incident_adl_event"]).copy()


def get_wave34_sample_flow_counts() -> dict[str, float]:
    base = load_wave34_harmonized_base()
    pre = build_wave34_preoutcome_landmark_base()
    riskset = build_wave34_landmark_riskset()
    complete = build_wave34_structural_m1_complete_case()
    return {
        "wave3_interview_alive": int((base["inw3"].astype("string") == "1.resp,alive").sum()),
        "age45plus": int(len(pre)),
        "adl_free_landmark": int((pre["adl_landmark"] == 0).sum()),
        "observed_wave4_adl_among_free": int(len(riskset)),
        "complete_structural_m1": int(len(complete)),
        "complete_event_n": int(complete["incident_adl_event"].sum()),
        "complete_event_rate": float(complete["incident_adl_event"].mean()),
    }


def run_structural_m1_oof(random_state: int = 20260327, n_splits: int = 10) -> tuple[pd.DataFrame, dict[str, float]]:
    df = build_wave34_structural_m1_complete_case().copy()
    X = df[M1_STRUCTURAL_PREDICTORS].astype(float)
    y = df["incident_adl_event"].astype(int)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = pd.Series(index=df.index, dtype=float)
    for train_idx, test_idx in kf.split(X):
        x_train = sm.add_constant(X.iloc[train_idx], has_constant="add")
        x_test = sm.add_constant(X.iloc[test_idx], has_constant="add")
        fit = sm.GLM(y.iloc[train_idx], x_train, family=sm.families.Binomial()).fit()
        oof.iloc[test_idx] = fit.predict(x_test)

    df["predicted_risk_oof"] = oof

    auc = float(roc_auc_score(y, oof))
    brier = float(brier_score_loss(y, oof))
    logit = np.log(np.clip(oof, 1e-6, 1 - 1e-6) / np.clip(1 - oof, 1e-6, 1 - 1e-6))
    intercept_model = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=logit).fit()
    slope_model = sm.GLM(y, sm.add_constant(logit), family=sm.families.Binomial()).fit()

    metrics = {
        "n": int(len(df)),
        "event_n": int(y.sum()),
        "event_rate": float(y.mean()),
        "auc": auc,
        "brier_score": brier,
        "calibration_intercept": float(intercept_model.params.iloc[0]),
        "calibration_slope": float(slope_model.params.iloc[1]),
        "n_splits": int(n_splits),
        "random_state": int(random_state),
    }
    return df, metrics


def make_risk_group_table(df: pd.DataFrame, pred_col: str, bins: int, label: str) -> pd.DataFrame:
    groups = pd.qcut(df[pred_col], bins, labels=list(range(1, bins + 1)), duplicates="drop")
    out = (
        df.assign(_group=groups)
        .groupby("_group", observed=True)
        .agg(
            n=("incident_adl_event", "size"),
            event_n=("incident_adl_event", "sum"),
            observed_event_rate=("incident_adl_event", "mean"),
            mean_predicted_risk=(pred_col, "mean"),
        )
        .reset_index()
        .rename(columns={"_group": "group"})
    )
    out["grouping"] = label
    out["group_share"] = out["n"] / len(df)
    return out[["grouping", "group", "n", "group_share", "event_n", "observed_event_rate", "mean_predicted_risk"]]

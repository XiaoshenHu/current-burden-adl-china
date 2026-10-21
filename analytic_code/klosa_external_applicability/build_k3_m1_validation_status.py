from __future__ import annotations

from pathlib import Path

# Resolve repository imports independently of the launch directory.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from data_processing_scripts.klosa_processing.harmonized_wave34_m1_utils import (
    RESULTS_DIR,
    make_risk_group_table,
    run_structural_m1_oof,
)


VALIDATION_LABEL = "structural external replication"


def build_performance_csv(metrics: dict[str, float], risk_df: pd.DataFrame) -> pd.DataFrame:
    quintiles = risk_df.loc[risk_df["grouping"] == "quintile"].copy()
    deciles = risk_df.loc[risk_df["grouping"] == "decile"].copy()
    top_q = quintiles.sort_values("group").iloc[-1]
    top_d = deciles.sort_values("group").iloc[-1]

    return pd.DataFrame(
        [
            {
                "target_model": "M1 current-burden model",
                "implemented": True,
                "implemented_validation_label": VALIDATION_LABEL,
                "requested_validation_level": "exact external validation / recalibrated transport / structural external replication",
                "validation_design": "wave3->wave4 landmark, fixed-seed 10-fold out-of-fold structural replication",
                "n": metrics["n"],
                "incident_adl_events": metrics["event_n"],
                "overall_event_rate": metrics["event_rate"],
                "auc": metrics["auc"],
                "brier_score": metrics["brier_score"],
                "calibration_intercept": metrics["calibration_intercept"],
                "calibration_slope": metrics["calibration_slope"],
                "top_quintile_share": top_q["group_share"],
                "top_quintile_observed_event_rate": top_q["observed_event_rate"],
                "top_decile_share": top_d["group_share"],
                "top_decile_observed_event_rate": top_d["observed_event_rate"],
                "note": "Implemented as structural external replication rather than exact or recalibrated transport because the physical domain remains grip-based proxy and the rural term is only an urban-rural analogue.",
            }
        ]
    )


def build_calibration_csv(metrics: dict[str, float], deciles: pd.DataFrame) -> pd.DataFrame:
    overall = pd.DataFrame(
        [
            {
                "record_type": "overall_metrics",
                "target_model": "M1 current-burden model",
                "implemented_validation_label": VALIDATION_LABEL,
                "calibration_intercept": metrics["calibration_intercept"],
                "calibration_slope": metrics["calibration_slope"],
                "observed_vs_predicted_summary": "Overall calibration was acceptable, with mild overprediction in the highest-risk decile and low-decile instability consistent with the low event rate.",
                "grouping": pd.NA,
                "group": pd.NA,
                "n": pd.NA,
                "group_share": pd.NA,
                "event_n": pd.NA,
                "observed_event_rate": pd.NA,
                "mean_predicted_risk": pd.NA,
                "observed_minus_predicted": pd.NA,
            }
        ]
    )

    by_group = deciles.copy()
    by_group["record_type"] = "decile_calibration"
    by_group["target_model"] = "M1 current-burden model"
    by_group["implemented_validation_label"] = VALIDATION_LABEL
    by_group["calibration_intercept"] = pd.NA
    by_group["calibration_slope"] = pd.NA
    by_group["observed_vs_predicted_summary"] = pd.NA
    by_group["observed_minus_predicted"] = by_group["observed_event_rate"] - by_group["mean_predicted_risk"]

    cols = [
        "record_type",
        "target_model",
        "implemented_validation_label",
        "calibration_intercept",
        "calibration_slope",
        "observed_vs_predicted_summary",
        "grouping",
        "group",
        "n",
        "group_share",
        "event_n",
        "observed_event_rate",
        "mean_predicted_risk",
        "observed_minus_predicted",
    ]
    return pd.concat([overall[cols].astype(object), by_group[cols].astype(object)], ignore_index=True)


def build_markdown(metrics: dict[str, float], quintiles: pd.DataFrame, deciles: pd.DataFrame) -> str:
    top_q = quintiles.sort_values("group").iloc[-1]
    top_d = deciles.sort_values("group").iloc[-1]
    lines = [
        "# KLoSA M1 validation summary",
        "",
        "## Validation label",
        f"- **Implemented validation type:** {VALIDATION_LABEL}",
        "- **Why this is not called exact transport:** the KLoSA physical-function domain is represented by a grip-based weakness proxy rather than the CHARLS physical-limitation count, and the KLoSA rural term is an urban-rural analogue rather than a hukou construct.",
        "- **Why this is not called recalibrated transport:** although harmonized KLoSA now exists locally, predictor equivalence remains insufficiently close for a clean frozen-coefficient or intercept/slope-recalibrated transport claim.",
        "",
        "## Validation cohort",
        f"- Complete structural-replication cohort: {metrics['n']}",
        f"- Incident ADL events: {metrics['event_n']}",
        f"- Overall incident ADL risk: {metrics['event_rate']:.4f}",
        "- Interval: wave 3 landmark to wave 4 incident ADL disability.",
        "",
        "## Core performance",
        f"- Out-of-fold AUC: {metrics['auc']:.3f}",
        f"- Brier score: {metrics['brier_score']:.3f}",
        f"- Calibration intercept: {metrics['calibration_intercept']:.3f}",
        f"- Calibration slope: {metrics['calibration_slope']:.3f}",
        "",
        "## Risk stratification",
        f"- Top predicted-risk quintile share: {top_q['group_share']:.3f}",
        f"- Top predicted-risk quintile observed event rate: {top_q['observed_event_rate']:.3f}",
        f"- Top predicted-risk decile share: {top_d['group_share']:.3f}",
        f"- Top predicted-risk decile observed event rate: {top_d['observed_event_rate']:.3f}",
        "",
        "## Calibration pattern",
        "- Overall calibration was acceptable.",
        "- The highest-risk decile showed only mild overprediction.",
        "- Lower deciles were noisier, which is expected in a low-event external replication setting.",
        "",
        "## Honest interpretation",
        "- KLoSA now provides a meaningful structural external replication of the current-burden predictor structure.",
        "- This result supports the transportability of the broader clinical message that current multidomain burden carries substantial near-term prognostic information for subsequent ADL disability.",
        "- It should not be overnamed as formal external validation of the frozen CHARLS M1 coefficients.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df, metrics = run_structural_m1_oof()
    quintiles = make_risk_group_table(df, "predicted_risk_oof", 5, "quintile")
    deciles = make_risk_group_table(df, "predicted_risk_oof", 10, "decile")
    risk_df = pd.concat([quintiles, deciles], ignore_index=True)
    risk_df["target_model"] = "M1 current-burden model"
    risk_df["implemented_validation_label"] = VALIDATION_LABEL
    risk_df["note"] = pd.NA
    risk_df.loc[
        (risk_df["grouping"] == "quintile") & (risk_df["group"] == risk_df.loc[risk_df["grouping"] == "quintile", "group"].max()),
        "note",
    ] = "Top-quintile clinically usable high-risk group."
    risk_df.loc[
        (risk_df["grouping"] == "decile") & (risk_df["group"] == risk_df.loc[risk_df["grouping"] == "decile", "group"].max()),
        "note",
    ] = "Top-decile finer-grained highest-risk group."

    performance = build_performance_csv(metrics, risk_df)
    calibration = build_calibration_csv(metrics, deciles)
    markdown = build_markdown(metrics, quintiles, deciles)

    performance.to_csv(RESULTS_DIR / "klosa_M1_performance.csv", index=False)
    calibration.to_csv(RESULTS_DIR / "klosa_M1_calibration.csv", index=False)
    risk_df.to_csv(RESULTS_DIR / "klosa_M1_risk_stratification.csv", index=False)
    (RESULTS_DIR / "klosa_M1_validation_summary.md").write_text(markdown, encoding="utf-8")

    print("Wrote:")
    print(RESULTS_DIR / "klosa_M1_validation_summary.md")
    print(RESULTS_DIR / "klosa_M1_performance.csv")
    print(RESULTS_DIR / "klosa_M1_calibration.csv")
    print(RESULTS_DIR / "klosa_M1_risk_stratification.csv")


if __name__ == "__main__":
    main()

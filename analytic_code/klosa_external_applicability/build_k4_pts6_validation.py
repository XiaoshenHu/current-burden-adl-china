from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(
    "/Users/macbookpro-hu/Library/CloudStorage/OneDrive-个人/Xcode/Medical Research/PROJECTS/charls_life_course_aging"
)
RAW_DIR = Path(
    "/Users/macbookpro-hu/Library/CloudStorage/OneDrive-个人/Xcode/Medical Research/DATA/public_databases/klosa/KLoSA 1-9th wave (STATA)"
)
RESULTS_DIR = PROJECT_ROOT / "results" / "klosa_external_validation"


def load_wave(wave: int) -> pd.DataFrame:
    path = RAW_DIR / f"str{wave:02d}_e.dta"
    cols = ["pid"]
    for col in [
        f"w{wave:02d}dep1",
        f"w{wave:02d}chronic_sum",
        f"w{wave:02d}mgrip",
        f"w{wave:02d}adl",
        f"w{wave:02d}A002_age",
        f"w{wave:02d}gender1",
    ]:
        if col in pd.read_stata(path, convert_categoricals=False, columns=None).columns:
            cols.append(col)
    return pd.read_stata(path, convert_categoricals=False, columns=cols)


def build_rule_sample() -> pd.DataFrame:
    waves = {wave: load_wave(wave) for wave in [1, 2, 3, 4]}
    df = (
        waves[1]
        .merge(waves[2], on="pid", how="inner")
        .merge(waves[3], on="pid", how="inner")
        .merge(waves[4][["pid", "w04adl"]], on="pid", how="inner")
    )

    risk = df[
        (df["w03A002_age"] >= 45)
        & (df["w03adl"] == 0)
        & (df["w04adl"].notna())
    ].copy()

    needed = [
        "w01dep1",
        "w02dep1",
        "w03dep1",
        "w03chronic_sum",
        "w01mgrip",
        "w02mgrip",
        "w03mgrip",
        "w03gender1",
    ]
    risk = risk[risk[needed].notna().all(axis=1)].copy()

    x = np.array([-4.0, -2.0, 0.0])
    x_centered = x - x.mean()
    var_x = (x_centered**2).sum()

    dep_vals = risk[["w01dep1", "w02dep1", "w03dep1"]].to_numpy(float)
    grip_vals = risk[["w01mgrip", "w02mgrip", "w03mgrip"]].to_numpy(float)
    risk["dep_slope"] = (
        ((dep_vals - dep_vals.mean(axis=1, keepdims=True)) * x_centered).sum(axis=1)
        / var_x
    )
    risk["grip_slope"] = (
        ((grip_vals - grip_vals.mean(axis=1, keepdims=True)) * x_centered).sum(axis=1)
        / var_x
    )

    risk["phys_current_flag"] = 0
    risk["phys_decline_flag"] = 0
    threshold_rows: list[dict] = []
    for sex_value, idx in risk.groupby("w03gender1").groups.items():
        q25_grip = float(risk.loc[idx, "w03mgrip"].quantile(0.25))
        q10_slope = float(risk.loc[idx, "grip_slope"].quantile(0.10))
        risk.loc[idx, "phys_current_flag"] = (risk.loc[idx, "w03mgrip"] <= q25_grip).astype(int)
        risk.loc[idx, "phys_decline_flag"] = (risk.loc[idx, "grip_slope"] <= q10_slope).astype(int)
        threshold_rows.append(
            {
                "component": "physical proxy thresholds",
                "sex_code": sex_value,
                "current_cutoff_definition": "wave-3 grip strength <= within-sex 25th percentile",
                "current_cutoff_value": q25_grip,
                "change_cutoff_definition": "pre-wave-3 grip slope <= within-sex 10th percentile",
                "change_cutoff_value": q10_slope,
            }
        )

    q75_mult = float(risk["w03chronic_sum"].quantile(0.75))
    q75_dep_slope = float(risk["dep_slope"].quantile(0.75))
    risk["dep_current_flag"] = (risk["w03dep1"] >= 1).astype(int)
    risk["mult_current_flag"] = (risk["w03chronic_sum"] >= q75_mult).astype(int)
    risk["dep_change_flag"] = (risk["dep_slope"] >= q75_dep_slope).astype(int)

    risk["score"] = (
        2 * risk["phys_current_flag"]
        + risk["dep_current_flag"]
        + risk["mult_current_flag"]
        + risk["dep_change_flag"]
        + risk["phys_decline_flag"]
    )
    risk["high_risk"] = (risk["score"] >= 4).astype(int)
    risk["low_risk"] = (risk["score"] <= 1).astype(int)
    risk["event"] = (risk["w04adl"] > 0).astype(int)

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.loc[len(threshold_df)] = {
        "component": "multimorbidity current",
        "sex_code": pd.NA,
        "current_cutoff_definition": "wave-3 chronic count >= overall 75th percentile",
        "current_cutoff_value": q75_mult,
        "change_cutoff_definition": pd.NA,
        "change_cutoff_value": pd.NA,
    }
    threshold_df.loc[len(threshold_df)] = {
        "component": "depression change",
        "sex_code": pd.NA,
        "current_cutoff_definition": "wave-3 depression status = 1",
        "current_cutoff_value": 1.0,
        "change_cutoff_definition": "pre-wave-3 depression slope >= overall 75th percentile",
        "change_cutoff_value": q75_dep_slope,
    }
    return risk, threshold_df


def build_outputs(rule_df: pd.DataFrame, threshold_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    total_n = int(len(rule_df))
    total_events = int(rule_df["event"].sum())
    total_risk = float(rule_df["event"].mean())
    high = rule_df[rule_df["high_risk"] == 1].copy()
    low = rule_df[rule_df["low_risk"] == 1].copy()
    mid = rule_df[(rule_df["high_risk"] == 0) & (rule_df["low_risk"] == 0)].copy()
    auc = float(roc_auc_score(rule_df["event"], rule_df["score"]))

    performance = pd.DataFrame(
        [
            {
                "target_rule": "PTS_6_core",
                "validation_label": "adapted external validation",
                "transport_feasibility": "exact transport impossible; adapted transport feasible",
                "sample_definition": "KLoSA wave 3 landmark risk set with wave 1-3 repeated inputs and wave 4 incident ADL outcome",
                "sample_n": total_n,
                "event_n": total_events,
                "overall_event_rate": total_risk,
                "score_auc": auc,
                "high_risk_n": int(len(high)),
                "high_risk_share": float(len(high) / total_n),
                "high_risk_observed_event_rate": float(high["event"].mean()),
                "low_risk_n": int(len(low)),
                "low_risk_share": float(len(low) / total_n),
                "low_risk_observed_event_rate": float(low["event"].mean()),
                "risk_difference_high_vs_low": float(high["event"].mean() - low["event"].mean()),
                "charls_reference_high_risk_share": 0.146,
                "charls_reference_high_risk_rate": 0.300,
                "charls_reference_low_risk_rate": 0.048,
                "charls_reference_auc": 0.710,
                "comparison_with_M1": "Direct KLoSA M1 comparison not yet available because M1 transport/replication remains blocked by unresolved cognition mapping.",
                "calibration_summary": "Not a probability model; no standalone calibration intercept/slope reported for the discrete score.",
            }
        ]
    )

    groups = []
    for group_name, sub in [
        ("Overall", rule_df),
        ("Low risk (score <=1)", low),
        ("Intermediate (score 2-3)", mid),
        ("High risk (score >=4)", high),
    ]:
        groups.append(
            {
                "group": group_name,
                "n": int(len(sub)),
                "share": float(len(sub) / total_n),
                "event_n": int(sub["event"].sum()),
                "observed_event_rate": float(sub["event"].mean()) if len(sub) else pd.NA,
                "mean_score": float(sub["score"].mean()) if len(sub) else pd.NA,
                "mean_wave3_dep1": float(sub["w03dep1"].mean()) if len(sub) else pd.NA,
                "mean_wave3_chronic_sum": float(sub["w03chronic_sum"].mean()) if len(sub) else pd.NA,
                "mean_wave3_mgrip": float(sub["w03mgrip"].mean()) if len(sub) else pd.NA,
            }
        )
    group_df = pd.DataFrame(groups)

    lines = []
    lines.append("# KLoSA PTS_6_core validation summary")
    lines.append("")
    lines.append("## Validation label")
    lines.append("- **Implemented label:** adapted external validation")
    lines.append("- **Exact transport:** not feasible, because the KLoSA physical component is a grip-strength proxy rather than the original CHARLS physical-limitation count, and the depressive component is binary rather than continuous.")
    lines.append("- **Adapted transport:** feasible, because all pragmatic-rule components can be mapped to KLoSA with explicit scale adaptation and percentile-based threshold mapping.")
    lines.append("")
    lines.append("## Adapted rule definition used in KLoSA")
    lines.append("- Landmark interval: waves 1-3 for feature derivation, wave 3 as the landmark, and wave 4 incident ADL disability as the next-wave outcome.")
    lines.append("- Current depression flag: wave-3 depression status (`dep1 = 1`).")
    lines.append("- Current multimorbidity flag: wave-3 chronic disease count at or above the overall 75th percentile.")
    lines.append("- Current physical flag: wave-3 grip strength at or below the within-sex 25th percentile.")
    lines.append("- Depression-change flag: participant-specific pre-wave-3 depression slope at or above the overall 75th percentile.")
    lines.append("- Physical-change flag: participant-specific pre-wave-3 grip slope at or below the within-sex 10th percentile.")
    lines.append("- Score construction preserved the locked CHARLS weight structure: `2*physical_current + depression_current + multimorbidity_current + depression_change + physical_change`.")
    lines.append("- High risk was defined as score >=4 and low risk as score <=1, consistent with the locked CHARLS rule structure.")
    lines.append("")
    lines.append("## Performance and clinical usability")
    lines.append(f"- Complete adapted-rule sample: {total_n} participants with {total_events} incident ADL events (overall event risk {total_risk:.3f}).")
    lines.append(f"- Score-based AUC: {auc:.3f}.")
    lines.append(f"- High-risk group: {len(high)} participants ({len(high) / total_n:.3%}) with observed incident ADL risk {high['event'].mean():.3f}.")
    lines.append(f"- Low-risk group: {len(low)} participants ({len(low) / total_n:.3%}) with observed incident ADL risk {low['event'].mean():.3f}.")
    lines.append(f"- High-versus-low absolute risk difference: {high['event'].mean() - low['event'].mean():.3f}.")
    lines.append("")
    lines.append("## Comparison with locked CHARLS rule performance")
    lines.append("- Locked CHARLS reference values for PTS_6_core were: high-risk share about 14.6%, high-risk observed risk about 0.300, low-risk observed risk about 0.048, and OOF AUC about 0.710.")
    lines.append("- The KLoSA adapted rule preserved a similar high-risk group size but showed a lower absolute event rate, which is consistent with the lower short-horizon incident ADL risk in the KLoSA validation interval.")
    lines.append("- Direct head-to-head comparison with KLoSA M1 is not yet available, because M1 transport/replication remains blocked by unresolved cognition mapping.")
    lines.append("")
    lines.append("## Honest interpretation")
    lines.append("- KLoSA can support a meaningful **adapted external validation** of the pragmatic rule structure.")
    lines.append("- The rule still identifies a smaller subgroup with a materially elevated short-horizon incident ADL risk, even under cross-cohort scale adaptation.")
    lines.append("- This is best interpreted as evidence for pragmatic transportability of simplified high-risk identification, not as strict transport validation of the original CHARLS score thresholds.")
    lines.append("")
    lines.append("## Threshold details used for the adapted validation")
    for _, row in threshold_df.iterrows():
        if pd.notna(row["sex_code"]):
            lines.append(
                f"- {row['component']} (sex code {int(row['sex_code'])}): {row['current_cutoff_definition']} = {row['current_cutoff_value']:.3f}; {row['change_cutoff_definition']} = {row['change_cutoff_value']:.3f}."
            )
        else:
            if pd.notna(row["change_cutoff_definition"]):
                lines.append(
                    f"- {row['component']}: {row['current_cutoff_definition']} = {row['current_cutoff_value']:.3f}; {row['change_cutoff_definition']} = {row['change_cutoff_value']:.3f}."
                )
            else:
                lines.append(
                    f"- {row['component']}: {row['current_cutoff_definition']} = {row['current_cutoff_value']:.3f}."
                )
    lines.append("")

    return performance, group_df, "\n".join(lines)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rule_df, threshold_df = build_rule_sample()
    performance, groups, md = build_outputs(rule_df, threshold_df)

    (RESULTS_DIR / "klosa_PTS6_validation_summary.md").write_text(md, encoding="utf-8")
    performance.to_csv(RESULTS_DIR / "klosa_PTS6_performance.csv", index=False)
    groups.to_csv(RESULTS_DIR / "klosa_PTS6_group_characteristics.csv", index=False)

    print("Wrote:")
    print(RESULTS_DIR / "klosa_PTS6_validation_summary.md")
    print(RESULTS_DIR / "klosa_PTS6_performance.csv")
    print(RESULTS_DIR / "klosa_PTS6_group_characteristics.csv")


if __name__ == "__main__":
    main()

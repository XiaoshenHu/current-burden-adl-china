#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "derived" / "raw_extracts"
RAW_ORIG = RAW / "orig_life_history"
DERIVED = ROOT / "derived"


def load_stata(path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_stata(path, columns=columns)


def zfill_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(12)


def code_yes(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.lower()
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out[s.str.startswith("1.")] = 1.0
    out[s.str.startswith("0.")] = 0.0
    return out


def code_no_is_adverse(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.lower()
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out[s.str.startswith("0.")] = 1.0
    out[s.str.startswith("1.")] = 0.0
    return out


def code_yes_12(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.lower().str.strip()
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out[s.str.startswith("1")] = 1.0
    out[s.str.startswith("2")] = 0.0
    return out


def code_no_is_adverse_12(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.lower().str.strip()
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out[s.str.startswith("2")] = 1.0
    out[s.str.startswith("1")] = 0.0
    return out


def code_low_edu(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.lower()
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out[s.str.startswith("1.")] = 1.0
    out[s.str.startswith("2.")] = 0.0
    out[s.str.startswith("3.")] = 0.0
    return out


def code_farming(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.lower()
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out[s.str.startswith("1.")] = 1.0
    out[s.str.startswith("2")] = 0.0
    return out


def code_poor_child_health(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.lower()
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out[s.str.startswith("1.")] = 0.0
    out[s.str.startswith("2.")] = 0.0
    out[s.str.startswith("3.")] = 0.0
    out[s.str.startswith("4.")] = 1.0
    out[s.str.startswith("5.")] = 1.0
    return out


def code_poor_child_health_12(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.lower().str.strip()
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out[s.str.startswith("1")] = 0.0
    out[s.str.startswith("2")] = 0.0
    out[s.str.startswith("3")] = 0.0
    out[s.str.startswith("4")] = 1.0
    out[s.str.startswith("5")] = 1.0
    return out


def code_rural(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.lower()
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out[s.str.startswith("1.")] = 1.0
    out[s.str.startswith("0.")] = 0.0
    return out


def code_poor_finance(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.lower().str.strip()
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out[s.str.startswith("1")] = 0.0
    out[s.str.startswith("2")] = 0.0
    out[s.str.startswith("3")] = 0.0
    out[s.str.startswith("4")] = 1.0
    out[s.str.startswith("5")] = 1.0
    return out


def code_freq_adverse(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.lower().str.strip()
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out[s.str.startswith("1")] = 1.0
    out[s.str.startswith("2")] = 1.0
    out[s.str.startswith("3")] = 0.0
    out[s.str.startswith("4")] = 0.0
    return out


def code_nonbio_caregiver(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.lower().str.strip()
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out[s.str.startswith("1")] = 0.0
    out[s.str.startswith("2")] = 1.0
    out[s.str.startswith("3")] = 1.0
    out[s.str.startswith("4")] = 1.0
    out[s.str.startswith("5")] = 1.0
    return out


def compute_sum(df: pd.DataFrame, cols: list[str], min_nonmissing: int = 1) -> pd.Series:
    counts = df[cols].notna().sum(axis=1)
    values = df[cols].sum(axis=1, skipna=True)
    values[counts < min_nonmissing] = np.nan
    return values


def compute_mean(df: pd.DataFrame, cols: list[str], min_nonmissing: int = 1) -> pd.Series:
    counts = df[cols].notna().sum(axis=1)
    values = df[cols].mean(axis=1, skipna=True)
    values[counts < min_nonmissing] = np.nan
    return values


def make_date(year_series: pd.Series, month_series: pd.Series, default_month: int = 6) -> pd.Series:
    year = pd.to_numeric(year_series, errors="coerce")
    month = pd.to_numeric(month_series, errors="coerce").fillna(default_month).clip(1, 12)
    return pd.to_datetime({"year": year, "month": month, "day": 15}, errors="coerce")


def midpoint_date(start: pd.Timestamp, end: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(start) or pd.isna(end):
        return pd.NaT
    return start + (end - start) / 2


def main() -> None:
    d_cols = [
        "ID", "inw1", "ragender", "rabyear", "raeducl", "r1rural2",
        "ramomeducl", "radadeducl", "ramomoccup_c", "radadoccup_c",
        "r1iwy", "r2iwy", "r3iwy", "r4iwy", "r1iwm", "r2iwm", "r3iwm", "r4iwm",
        "r1cesd10", "r2cesd10", "r3cesd10", "r4cesd10",
        "r1tr20", "r2tr20", "r3tr20", "r4tr20",
        "r1walk100a", "r2walk100a", "r3walk100a", "r4walk100a",
        "r1walk1kma", "r2walk1kma", "r3walk1kma", "r4walk1kma",
        "r1chaira", "r2chaira", "r3chaira", "r4chaira",
        "r1stoopa", "r2stoopa", "r3stoopa", "r4stoopa",
        "r1lifta", "r2lifta", "r3lifta", "r4lifta",
        "r1hibpe", "r2hibpe", "r3hibpe", "r4hibpe",
        "r1diabe", "r2diabe", "r3diabe", "r4diabe",
        "r1hearte", "r2hearte", "r3hearte", "r4hearte",
        "r1stroke", "r2stroke", "r3stroke", "r4stroke",
        "r1lunge", "r2lunge", "r3lunge", "r4lunge",
        "r1arthre", "r2arthre", "r3arthre", "r4arthre",
        "r1kidneye", "r2kidneye", "r3kidneye", "r4kidneye",
        "r1adlab_c", "r2adlab_c", "r3adlab_c", "r4adlab_c",
        "r2iadlza", "r3iadlza", "r4iadlza",
        "r2iwstat", "r3iwstat", "r4iwstat",
    ]
    health_cols = ["ID", "hs002", "hs003", "hs004", "hs005", "hs006", "hc001"]
    family_cols = [
        "ID", "a1a", "a2a_1", "a1b", "a2b_1", "a7", "a10", "a4a", "a4b",
        "c4", "c5", "c7", "c4_a", "f3", "j10_1", "j10_2", "j10_3",
    ]
    eol_cols = ["ID", "raxt", "raxyear", "raxmonth", "radage", "racod_c", "ragcod"]

    d = load_stata(RAW / "H_CHARLS_D_Data.dta", d_cols)
    health = load_stata(RAW_ORIG / "Health_History.dta", health_cols)
    family = load_stata(RAW_ORIG / "Family_Information.dta", family_cols)
    eol = load_stata(RAW / "H_CHARLS_EOL_a.dta", eol_cols)

    d = d[d["inw1"].astype("string") == "1.yes"].copy()
    d["ID_str"] = zfill_id(d["ID"])
    health["ID_str"] = zfill_id(health["ID"])
    family["ID_str"] = zfill_id(family["ID"])
    eol["ID_str"] = zfill_id(eol["ID"])

    d["female"] = np.where(
        d["ragender"].isna(),
        np.nan,
        (d["ragender"].astype("string") == "2.woman").astype(float),
    )
    d["baseline_age_2011"] = 2011 - d["rabyear"]
    d["low_self_education"] = code_low_edu(d["raeducl"])
    d["rural_hukou_2011"] = code_rural(d["r1rural2"])
    d["low_mother_edu"] = code_low_edu(d["ramomeducl"])
    d["low_father_edu"] = code_low_edu(d["radadeducl"])
    d["mother_farming"] = code_farming(d["ramomoccup_c"])
    d["father_farming"] = code_farming(d["radadoccup_c"])

    health["child_poor_health"] = code_poor_child_health_12(health["hs002"])
    health["child_miss_school_health"] = code_yes_12(health["hs003"]).where(
        ~health["hs003"].astype("string").str.startswith("3"), np.nan
    )
    health["child_bedridden"] = code_yes_12(health["hs004"])
    health["child_hospitalized"] = code_yes_12(health["hs005"])
    health["child_repeated_hospital"] = code_yes_12(health["hs006"])
    health["no_childhood_vaccine"] = code_no_is_adverse_12(health["hc001"])
    health["child_illness_any"] = (
        health[["child_miss_school_health", "child_bedridden", "child_hospitalized", "child_repeated_hospital"]]
        .max(axis=1, skipna=True)
    )

    family["family_starved"] = code_yes_12(family["c4"])
    family["went_out_for_food"] = code_yes_12(family["c5"])
    family["family_starved_death"] = code_yes_12(family["c7"])
    family["poor_financial_status_childhood"] = code_poor_finance(family["c4_a"])
    family["nonbio_female_caregiver"] = code_nonbio_caregiver(family["a4a"])
    family["nonbio_male_caregiver"] = code_nonbio_caregiver(family["a4b"])
    family["school_bullying"] = code_freq_adverse(family["f3"])
    family["parental_quarrel"] = code_freq_adverse(family["j10_1"])
    family["father_beat_mother"] = code_freq_adverse(family["j10_2"])
    family["mother_beat_father"] = code_freq_adverse(family["j10_3"])
    family["parents_divorced"] = code_yes_12(family["a7"])

    df = d.merge(
        health[
            [
                "ID_str", "child_poor_health", "child_miss_school_health", "child_bedridden",
                "child_hospitalized", "child_repeated_hospital", "no_childhood_vaccine",
                "child_illness_any",
            ]
        ],
        on="ID_str",
        how="left",
    )
    df = df.merge(
        family[
            [
                "ID_str", "a1a", "a2a_1", "a1b", "a2b_1", "parents_divorced",
                "nonbio_female_caregiver", "nonbio_male_caregiver",
                "family_starved", "went_out_for_food", "family_starved_death",
                "poor_financial_status_childhood", "school_bullying",
                "parental_quarrel", "father_beat_mother", "mother_beat_father",
            ]
        ],
        on="ID_str",
        how="left",
    )
    df = df.merge(eol.drop(columns=["ID"]), on="ID_str", how="left")

    basic_adversity_items = [
        "child_poor_health",
        "child_illness_any",
        "no_childhood_vaccine",
        "low_mother_edu",
        "low_father_edu",
        "mother_farming",
        "father_farming",
    ]
    mother_alive = df["a1a"].astype("string").str.startswith("1").fillna(False)
    father_alive = df["a1b"].astype("string").str.startswith("1").fillna(False)
    df["maternal_death_childhood"] = np.where(
        mother_alive,
        0.0,
        np.where(
            pd.to_numeric(df["a2a_1"], errors="coerce").notna(),
            ((pd.to_numeric(df["a2a_1"], errors="coerce") - df["rabyear"]) <= 17).astype(float),
            np.nan,
        ),
    )
    df["paternal_death_childhood"] = np.where(
        father_alive,
        0.0,
        np.where(
            pd.to_numeric(df["a2b_1"], errors="coerce").notna(),
            ((pd.to_numeric(df["a2b_1"], errors="coerce") - df["rabyear"]) <= 17).astype(float),
            np.nan,
        ),
    )
    df["nonbio_caregiver_any"] = df[["nonbio_female_caregiver", "nonbio_male_caregiver"]].max(axis=1, skipna=True)
    df["domestic_violence_any"] = df[["father_beat_mother", "mother_beat_father"]].max(axis=1, skipna=True)

    refined_adversity_items = [
        "child_poor_health",
        "child_miss_school_health",
        "child_bedridden",
        "child_hospitalized",
        "child_repeated_hospital",
        "no_childhood_vaccine",
        "family_starved",
        "went_out_for_food",
        "family_starved_death",
        "poor_financial_status_childhood",
        "maternal_death_childhood",
        "paternal_death_childhood",
        "parents_divorced",
        "nonbio_caregiver_any",
        "school_bullying",
        "parental_quarrel",
        "domestic_violence_any",
    ]

    df["adversity_basic_nonmissing"] = df[basic_adversity_items].notna().sum(axis=1)
    df["adversity_basic_score_raw"] = df[basic_adversity_items].sum(axis=1, skipna=True)
    df["adversity_basic_score_scaled"] = np.where(
        df["adversity_basic_nonmissing"] >= 5,
        df["adversity_basic_score_raw"] / df["adversity_basic_nonmissing"] * len(basic_adversity_items),
        np.nan,
    )

    df["adversity_nonmissing"] = df[refined_adversity_items].notna().sum(axis=1)
    df["adversity_score_raw"] = df[refined_adversity_items].sum(axis=1, skipna=True)
    df["adversity_score_scaled"] = np.where(
        df["adversity_nonmissing"] >= 8,
        df["adversity_score_raw"] / df["adversity_nonmissing"] * len(refined_adversity_items),
        np.nan,
    )
    valid_refined = df["adversity_score_scaled"].notna()
    df["adversity_cat"] = pd.Series(pd.NA, index=df.index, dtype="object")
    df.loc[valid_refined, "adversity_cat"] = pd.qcut(
        df.loc[valid_refined, "adversity_score_scaled"],
        q=3,
        labels=["low", "moderate", "high"],
        duplicates="drop",
    ).astype("string")

    for w in [1, 2, 3, 4]:
        df[f"dep_w{w}"] = df[f"r{w}cesd10"]
        df[f"cogdef_w{w}"] = 20 - df[f"r{w}tr20"]
        phys_cols = [f"r{w}walk100a", f"r{w}walk1kma", f"r{w}chaira", f"r{w}stoopa", f"r{w}lifta"]
        df[f"physical_lim_w{w}"] = compute_mean(df.assign(**{c: code_yes(df[c]) for c in phys_cols}), phys_cols, min_nonmissing=3)
        mm_cols = [f"r{w}hibpe", f"r{w}diabe", f"r{w}hearte", f"r{w}stroke", f"r{w}lunge", f"r{w}arthre", f"r{w}kidneye"]
        df = df.assign(**{c: code_yes(df[c]) for c in mm_cols})
        df[f"multimorb_w{w}"] = compute_sum(df, mm_cols, min_nonmissing=5)

    baseline_date = make_date(df["r1iwy"], df["r1iwm"])
    w2_date = make_date(df["r2iwy"], df["r2iwm"])
    w3_date = make_date(df["r3iwy"], df["r3iwm"])
    w4_date = make_date(df["r4iwy"], df["r4iwm"])
    exact_death_date = make_date(df["raxyear"], df["raxmonth"])
    wave2_anchor = pd.Timestamp("2013-07-01")
    wave3_anchor = pd.Timestamp("2015-07-01")
    wave4_anchor = pd.Timestamp("2018-07-01")

    # outcomes
    df["death_by_w4"] = (
        df[["r2iwstat", "r3iwstat", "r4iwstat"]]
        .astype("string")
        .apply(lambda x: x.str.startswith(("5.", "6.")))
        .any(axis=1)
        .astype(int)
    )

    death_date = []
    death_date_source = []
    censor_date = []
    for _, row in df.iterrows():
        bdate = baseline_date.loc[row.name]
        d2 = w2_date.loc[row.name]
        d3 = w3_date.loc[row.name]
        d4 = w4_date.loc[row.name]
        exact = exact_death_date.loc[row.name]
        s2 = str(row["r2iwstat"]).lower()
        s3 = str(row["r3iwstat"]).lower()
        s4 = str(row["r4iwstat"]).lower()

        if row["death_by_w4"] == 1:
            if pd.notna(exact):
                death_date.append(exact)
                death_date_source.append("exact_eol")
            elif s2.startswith("5."):
                death_date.append(midpoint_date(bdate, d2 if pd.notna(d2) else wave2_anchor))
                death_date_source.append("midpoint_w1_w2")
            elif s3.startswith("6."):
                death_date.append(midpoint_date(bdate, d2 if pd.notna(d2) else wave2_anchor))
                death_date_source.append("midpoint_w1_w2_proxy")
            elif s3.startswith("5."):
                death_date.append(
                    midpoint_date(
                        d2 if pd.notna(d2) else wave2_anchor,
                        d3 if pd.notna(d3) else wave3_anchor,
                    )
                )
                death_date_source.append("midpoint_w2_w3")
            elif s4.startswith("6."):
                death_date.append(
                    midpoint_date(
                        d2 if pd.notna(d2) else wave2_anchor,
                        d3 if pd.notna(d3) else wave3_anchor,
                    )
                )
                death_date_source.append("midpoint_w2_w3_proxy")
            elif s4.startswith("5."):
                death_date.append(
                    midpoint_date(
                        d3 if pd.notna(d3) else wave3_anchor,
                        d4 if pd.notna(d4) else wave4_anchor,
                    )
                )
                death_date_source.append("midpoint_w3_w4")
            else:
                death_date.append(pd.NaT)
                death_date_source.append(pd.NA)
        else:
            death_date.append(pd.NaT)
            death_date_source.append(pd.NA)

        alive_dates = [bdate]
        if s2.startswith(("1.", "4.")) and pd.notna(d2):
            alive_dates.append(d2)
        if s3.startswith(("1.", "4.")) and pd.notna(d3):
            alive_dates.append(d3)
        if s4.startswith(("1.", "4.")) and pd.notna(d4):
            alive_dates.append(d4)
        alive_dates = [x for x in alive_dates if pd.notna(x)]
        censor_date.append(max(alive_dates) if alive_dates else pd.NaT)

    df["death_date"] = pd.to_datetime(death_date)
    df["death_date_source"] = death_date_source
    df["censor_date"] = pd.to_datetime(censor_date)
    df["survival_end_date"] = df["censor_date"]
    df.loc[df["death_by_w4"] == 1, "survival_end_date"] = df.loc[df["death_by_w4"] == 1, "death_date"]
    df["death_event"] = df["death_by_w4"]
    df["survival_time_years"] = (df["survival_end_date"] - baseline_date).dt.days / 365.25
    df.loc[(df["death_event"] == 1) & (df["survival_time_years"] <= 0), "survival_time_years"] = 1 / 365.25
    df.loc[(df["death_event"] == 0) & (df["survival_time_years"] <= 0), "survival_time_years"] = np.nan

    df["adl_disability_baseline"] = (df["r1adlab_c"].fillna(0) > 0).astype(int)
    df["adl_disability_w4"] = (df["r4adlab_c"].fillna(0) > 0).astype(int)
    df["incident_adl_by_w4"] = np.where(
        df["adl_disability_baseline"] == 0,
        df["adl_disability_w4"],
        np.nan,
    )

    keep_cols = [
        "ID", "ID_str", "baseline_age_2011", "female", "raeducl", "low_self_education", "rural_hukou_2011",
        "adversity_basic_nonmissing", "adversity_basic_score_raw", "adversity_basic_score_scaled",
        "adversity_nonmissing", "adversity_score_raw", "adversity_score_scaled", "adversity_cat",
        *basic_adversity_items, *refined_adversity_items,
        "low_mother_edu", "low_father_edu", "mother_farming", "father_farming",
        "nonbio_female_caregiver", "nonbio_male_caregiver", "domestic_violence_any",
        "death_by_w4", "death_event", "survival_time_years", "death_date_source",
        "raxt", "raxyear", "raxmonth", "radage", "racod_c", "ragcod",
        "adl_disability_baseline", "adl_disability_w4", "incident_adl_by_w4",
    ] + [f"{x}_w{w}" for x in ["dep", "cogdef", "physical_lim", "multimorb"] for w in [1, 2, 3, 4]]
    keep_cols = list(dict.fromkeys(keep_cols))
    analytic = df[keep_cols].copy()
    analytic.to_csv(DERIVED / "charls_analytic_base.csv", index=False)

    summary_lines = [
        "# CHARLS Analytic Base Summary",
        "",
        f"- Baseline sample (inw1=yes): {len(analytic)}",
        f"- Refined adversity score available (>=8 items): {analytic['adversity_score_scaled'].notna().sum()}",
        f"- Basic adversity score available (>=5 items): {analytic['adversity_basic_score_scaled'].notna().sum()}",
        f"- Death by wave 4: {int(analytic['death_by_w4'].sum())}",
        f"- Survival time available: {analytic['survival_time_years'].notna().sum()}",
        f"- Incident ADL sample: {analytic['incident_adl_by_w4'].notna().sum()}",
        "",
        "## Adversity categories",
        analytic["adversity_cat"].value_counts(dropna=False).to_string(),
        "",
        "## Refined adversity item missingness",
        analytic[refined_adversity_items].isna().mean().round(3).to_string(),
        "",
        "## Missingness (selected)",
        analytic[
            ["adversity_score_scaled", "dep_w1", "dep_w4", "cogdef_w1", "cogdef_w4", "physical_lim_w1", "physical_lim_w4", "multimorb_w1", "multimorb_w4"]
        ].isna().mean().round(3).to_string(),
    ]
    (DERIVED / "charls_analytic_base_summary.md").write_text("\n".join(summary_lines))


if __name__ == "__main__":
    main()

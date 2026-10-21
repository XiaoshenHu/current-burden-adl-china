from __future__ import annotations

from pathlib import Path

# Resolve repository imports independently of the launch directory.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reproducibility.project_paths import find_stata, gateway_do_file, klosa_data_root, klosa_raw_dir, project_root
import pandas as pd
import zipfile
from openpyxl import load_workbook


PROJECT_ROOT = project_root()
DATA_ROOT = klosa_data_root()
RAW_DIR = klosa_raw_dir()
KEIS_DIR = DATA_ROOT / "KEIS_codebooks"
HARMONIZED_DO_FILE = gateway_do_file()
RESULTS_DIR = PROJECT_ROOT / "results" / "klosa_external_validation"


def detect_harmonized_klosa() -> list[Path]:
    candidates = []
    scan_roots = [DATA_ROOT, RESULTS_DIR]
    for root in scan_roots:
        for path in root.rglob("*"):
            if not path.is_dir():
                continue
            lower = path.name.lower()
            if "harmonized" in lower and "klosa" in lower:
                candidates.append(path)
    build_dir = RESULTS_DIR / "gateway_harmonized_build"
    if build_dir.exists():
        candidates.append(build_dir)
    return sorted(set(candidates))


def detect_stata_binary() -> Path | None:
    return find_stata()


def detect_imputation_packages() -> dict:
    outer = DATA_ROOT / "KLoSA_Imputation_E_v2_20210324.zip"
    outer_dup = DATA_ROOT / "KLoSA_Imputation_E_v2_20210324 (1).zip"
    available = outer if outer.exists() else (outer_dup if outer_dup.exists() else None)
    inner_members: list[str] = []
    if available:
        with zipfile.ZipFile(available) as zf:
            if "KLoSA_STATA_Imputation_E_v2_20210324.zip" in zf.namelist():
                inner_members.append("KLoSA_STATA_Imputation_E_v2_20210324.zip")
    return {
        "outer_zip": str(available) if available else "",
        "has_outer_zip": bool(available),
        "has_nested_stata_zip": bool(inner_members),
    }


def detect_exit_sources() -> dict:
    extracted = sorted(
        [
            p.name
            for p in DATA_ROOT.glob("*.dta")
            if "exit" in p.name.lower()
        ]
    )
    zipped = {
        "wave7": DATA_ROOT / "2018 KLoSA 7 wave EXIT.zip",
        "wave8": DATA_ROOT / "KLoSA 8th wave_EXIT.zip",
        "wave9": DATA_ROOT / "KLoSA 9 wave Exit.zip",
    }
    return {
        "extracted": extracted,
        "has_w2_to_w6_extracted": all(
            (DATA_ROOT / fname).exists()
            for fname in [
                "w02_exit_e.dta",
                "w03_exit_e.dta",
                "w04_exit_e.dta",
                "w05_exit_e.dta",
                "w06_Exit_e.dta",
            ]
        ),
        "has_w7_zip": zipped["wave7"].exists(),
        "has_w8_zip": zipped["wave8"].exists(),
        "has_w9_zip": zipped["wave9"].exists(),
    }


def get_row_count(dta_path: Path) -> int:
    return len(pd.read_stata(dta_path, convert_categoricals=False))


def build_wave_summary() -> pd.DataFrame:
    rows = []
    for prefix in ["w", "str", "Lt"]:
        for wave in range(1, 10):
            stem = f"{prefix}{wave:02d}_e.dta"
            path = RAW_DIR / stem
            rows.append(
                {
                    "file_group": prefix,
                    "wave": wave,
                    "file_name": stem,
                    "rows": get_row_count(path),
                }
            )
    extra_path = RAW_DIR / "w05_new_e.dta"
    rows.append(
        {
            "file_group": "w_new",
            "wave": 5,
            "file_name": extra_path.name,
            "rows": get_row_count(extra_path),
        }
    )
    return pd.DataFrame(rows)


def build_crosswalk() -> pd.DataFrame:
    rows = [
        {
            "priority target": "M1 / cohort backbone",
            "CHARLS variable/domain": "Age at landmark",
            "CHARLS role": "Base covariate",
            "Preferred Gateway harmonized KLoSA variable(s)": "r3agey",
            "Preferred raw KLoSA candidate(s)": "w03A002_age",
            "wave availability for primary wave3→4 validation": "Directly available at wave 3",
            "scale/range": "Continuous age in years",
            "direction": "Higher = older",
            "comparability rating": "high",
            "best current validation use": "M1 exact/recalibrated transport or structural replication",
            "most defensible validation level": "Level A/B/C all structurally supported",
            "source evidence": "Gateway do-file June 2025; local raw inspection",
            "notes": "Direct age mapping is one of the cleanest transport components.",
        },
        {
            "priority target": "M1 / cohort backbone",
            "CHARLS variable/domain": "Sex",
            "CHARLS role": "Base covariate",
            "Preferred Gateway harmonized KLoSA variable(s)": "ragender",
            "Preferred raw KLoSA candidate(s)": "w03gender1",
            "wave availability for primary wave3→4 validation": "Directly available at wave 3",
            "scale/range": "Binary categorical",
            "direction": "Female vs male coding mappable",
            "comparability rating": "high",
            "best current validation use": "M1 exact/recalibrated transport or structural replication",
            "most defensible validation level": "Level A/B/C all structurally supported",
            "source evidence": "Gateway do-file June 2025; local raw inspection",
            "notes": "Raw and harmonized routes are both straightforward.",
        },
        {
            "priority target": "M1 / cohort backbone",
            "CHARLS variable/domain": "Low education",
            "CHARLS role": "Base covariate",
            "Preferred Gateway harmonized KLoSA variable(s)": "raeducl (3-level) or raeducel (5-level)",
            "Preferred raw KLoSA candidate(s)": "w03A005 or w03edu",
            "wave availability for primary wave3→4 validation": "Directly available at wave 3; stable across waves",
            "scale/range": "Ordered educational attainment categories",
            "direction": "Lower attainment = higher disadvantage",
            "comparability rating": "high",
            "best current validation use": "M1 exact/recalibrated transport or structural replication",
            "most defensible validation level": "Level A/B/C all structurally supported",
            "source evidence": "Gateway do-file June 2025; KEIS demographics mapping; local raw inspection",
            "notes": "The harmonized build explicitly creates education recodes from wave-specific raw education items.",
        },
        {
            "priority target": "M1 / cohort backbone",
            "CHARLS variable/domain": "Rural hukou",
            "CHARLS role": "Base covariate proxy",
            "Preferred Gateway harmonized KLoSA variable(s)": "r3rural with optional support from r3metrop_k",
            "Preferred raw KLoSA candidate(s)": "w03region2, w03region3, w03enu_type",
            "wave availability for primary wave3→4 validation": "Directly available at wave 3",
            "scale/range": "Village/town and metropolitan/city/town categories",
            "direction": "Need urban-rural proxy mapping",
            "comparability rating": "moderate",
            "best current validation use": "M1 recalibrated transport or structural replication",
            "most defensible validation level": "Level B/C",
            "source evidence": "Gateway do-file June 2025; KEIS demographics mapping",
            "notes": "The official harmonized build creates a rural indicator from region2, but this remains a Korea-specific urban-rural proxy rather than a true hukou analogue.",
        },
        {
            "priority target": "M1 / validation cohort",
            "CHARLS variable/domain": "Community-dwelling restriction at landmark",
            "CHARLS role": "Eligibility refinement",
            "Preferred Gateway harmonized KLoSA variable(s)": "No clear dedicated institutionalisation flag in the detected harmonized script",
            "Preferred raw KLoSA candidate(s)": "w03enu_type",
            "wave availability for primary wave3→4 validation": "Residential type available at wave 3",
            "scale/range": "Residential-type categories",
            "direction": "Context/eligibility indicator",
            "comparability rating": "low",
            "best current validation use": "Sensitivity restriction only, if a defensible community-only cut can be justified",
            "most defensible validation level": "Cautious Level C support only",
            "source evidence": "Gateway do-file June 2025; KEIS demographics mapping; prior local feasibility audit",
            "notes": "This remains weaker than the other base covariates. Primary KLoSA validation may need to be reported without a strict institutional-exclusion analogue unless `enu_type` can be validated empirically.",
        },
        {
            "priority target": "M1 / primary outcome",
            "CHARLS variable/domain": "Incident ADL disability outcome",
            "CHARLS role": "Primary outcome",
            "Preferred Gateway harmonized KLoSA variable(s)": "r3adltotba_k at landmark and r4adltotba_k at next wave",
            "Preferred raw KLoSA candidate(s)": "w03adl / w04adl or raw items w03C201-w03C207 and w04C201-w04C207",
            "wave availability for primary wave3→4 validation": "Directly available for wave 3 and wave 4",
            "scale/range": "Binary any-ADL-help outcome or 0-7 ADL burden summary",
            "direction": "Higher = worse disability burden",
            "comparability rating": "high",
            "best current validation use": "M1 and PTS event definition",
            "most defensible validation level": "Level A/B/C all structurally supported",
            "source evidence": "Gateway do-file June 2025; KEIS health mapping; local raw inspection",
            "notes": "The harmonized script creates both count and binary ADL summaries, which is better aligned with the CHARLS landmark outcome than the coarse raw `adl` index alone.",
        },
        {
            "priority target": "Supportive outcome",
            "CHARLS variable/domain": "Incident IADL disability outcome",
            "CHARLS role": "Supportive outcome",
            "Preferred Gateway harmonized KLoSA variable(s)": "r3iadltotba_k at landmark and r4iadltotba_k at next wave",
            "Preferred raw KLoSA candidate(s)": "w03iadl / w04iadl or raw items w03C208-w03C217 and w04C208-w04C217",
            "wave availability for primary wave3→4 validation": "Directly available for wave 3 and wave 4",
            "scale/range": "Binary any-IADL-help outcome or 0-10 IADL burden summary",
            "direction": "Higher = worse disability burden",
            "comparability rating": "high",
            "best current validation use": "Supportive external replication",
            "most defensible validation level": "Level C immediately; Level B possible on harmonized build",
            "source evidence": "Gateway do-file June 2025; KEIS health mapping; local raw inspection",
            "notes": "Useful as a supportive outcome but not the first transport target.",
        },
        {
            "priority target": "M1 / PTS current burden",
            "CHARLS variable/domain": "Depressive symptoms at landmark",
            "CHARLS role": "Current-burden domain",
            "Preferred Gateway harmonized KLoSA variable(s)": "r3cesd10a (preferred), with r3depresl as one component item",
            "Preferred raw KLoSA candidate(s)": "w03dep1 status summary; if needed, wave-3 imputation CES-D items reconstructed through the official do-file",
            "wave availability for primary wave3→4 validation": "Wave 3 current measure available; repeated waves 1-3 available for change terms after harmonized build",
            "scale/range": "CES-D 10 summary on the harmonized route; binary depression status on the raw-summary fallback route",
            "direction": "Higher = worse depressive burden",
            "comparability rating": "moderate",
            "best current validation use": "M1 recalibrated transport or structural replication; PTS adapted validation",
            "most defensible validation level": "Level B/C for M1; adapted validation for PTS",
            "source evidence": "Gateway do-file June 2025; KEIS health mapping; local raw inspection",
            "notes": "The official harmonized script materially improves depression comparability because it builds CES-D summaries from item-level data. Exact transport still depends on matching the CHARLS burden scaling, but this is no longer a weakly mapped domain.",
        },
        {
            "priority target": "M1 current burden",
            "CHARLS variable/domain": "Cognitive deficit at landmark",
            "CHARLS role": "Current-burden domain",
            "Preferred Gateway harmonized KLoSA variable(s)": "Construct from harmonized cognition components: r3orient, r3orientp_k, r3tr6, r3ser7, r3draw, r3rpsnt, r3read, r3senten",
            "Preferred raw KLoSA candidate(s)": "Official KEIS mapping tables list w03mmse / w03mmseg in str/Lt, but the local raw build path is stronger via imputation items w03C401-w03C419 used by the official do-file",
            "wave availability for primary wave3→4 validation": "Wave 3 current cognition and waves 1-3 repeated cognition components available through the official harmonized build path",
            "scale/range": "Component-based cognition profile rather than one directly matched CHARLS scalar",
            "direction": "Worse cognition = higher risk after direction harmonisation",
            "comparability rating": "moderate",
            "best current validation use": "M1 structural replication immediately after harmonized build; recalibrated transport if a defensible scalar mapping is fixed",
            "most defensible validation level": "Level B contingent on harmonized build; Level C immediately after confirmed reconstruction",
            "source evidence": "Gateway do-file June 2025; KEIS structured and light mapping tables",
            "notes": "This domain is no longer best described as 'absent'. The honest update is that cognition is available, but the preferred route is the official harmonized build rather than ad hoc reliance on raw `mmse/mmseg` summaries with inconsistent labels across KEIS tables.",
        },
        {
            "priority target": "M1 / PTS current burden",
            "CHARLS variable/domain": "Physical limitation at landmark",
            "CHARLS role": "Current-burden domain / pragmatic rule component",
            "Preferred Gateway harmonized KLoSA variable(s)": "r3gripsum (preferred proxy), with support from r3gripcomp and r3hlthlm if needed",
            "Preferred raw KLoSA candidate(s)": "w03mgrip and related grip-test completion variables",
            "wave availability for primary wave3→4 validation": "Wave 3 current proxy and waves 1-3 repeated grip measurements available",
            "scale/range": "Dominant-hand grip strength in kg plus test-completion status",
            "direction": "Lower grip = worse function; sign must be reversed for burden-style modelling",
            "comparability rating": "low",
            "best current validation use": "PTS adapted validation and M1 structural replication with explicit proxy language",
            "most defensible validation level": "Level C for M1; adapted validation for PTS",
            "source evidence": "Gateway do-file June 2025; KEIS health mapping; local raw inspection",
            "notes": "This remains the weakest domain match for strict transport because KLoSA contributes a performance-based grip proxy rather than the CHARLS physical-limitation count. It is still useful for pragmatic transportability, but not as an exact analogue.",
        },
        {
            "priority target": "M1 / PTS current burden",
            "CHARLS variable/domain": "Multimorbidity at landmark",
            "CHARLS role": "Current-burden domain / pragmatic rule component",
            "Preferred Gateway harmonized KLoSA variable(s)": "Derive count from harmonized disease indicators (eg, r3hibpe, r3diabe, r3hearte, r3stroke, r3arthre, etc)",
            "Preferred raw KLoSA candidate(s)": "w03chronic_sum",
            "wave availability for primary wave3→4 validation": "Direct summary available at wave 3; repeated history also derivable",
            "scale/range": "Count of chronic diseases",
            "direction": "Higher = worse",
            "comparability rating": "high",
            "best current validation use": "M1 and PTS transportability work",
            "most defensible validation level": "Level A/B/C structurally supported",
            "source evidence": "Gateway do-file June 2025; KEIS health mapping; local raw inspection",
            "notes": "This remains one of the strongest cross-cohort domain mappings.",
        },
        {
            "priority target": "PTS / M2 change structure",
            "CHARLS variable/domain": "Pre-landmark depressive change",
            "CHARLS role": "Change component",
            "Preferred Gateway harmonized KLoSA variable(s)": "Participant-specific slope over r1cesd10a, r2cesd10a, r3cesd10a",
            "Preferred raw KLoSA candidate(s)": "Fallback slope over wave-specific depression summaries, if harmonized build is unavailable",
            "wave availability for primary wave3→4 validation": "Waves 1-3 available",
            "scale/range": "Participant-specific linear slope using the CHARLS-aligned time coding (-4, -2, 0)",
            "direction": "More adverse increase = higher risk",
            "comparability rating": "moderate",
            "best current validation use": "PTS adapted validation and secondary M2-style sensitivity work",
            "most defensible validation level": "Adapted validation or Level C",
            "source evidence": "Gateway do-file June 2025; CHARLS slope time coding; prior K4 implementation",
            "notes": "This is more defensible on the harmonized route because repeated CES-D summaries are explicitly constructed there.",
        },
        {
            "priority target": "PTS / M2 change structure",
            "CHARLS variable/domain": "Pre-landmark physical change",
            "CHARLS role": "Change component",
            "Preferred Gateway harmonized KLoSA variable(s)": "Participant-specific slope over r1gripsum, r2gripsum, r3gripsum; sex-specific percentile adaptation if needed",
            "Preferred raw KLoSA candidate(s)": "Fallback slope over w01mgrip, w02mgrip, w03mgrip",
            "wave availability for primary wave3→4 validation": "Waves 1-3 available",
            "scale/range": "Participant-specific grip-strength slope using the CHARLS-aligned time coding (-4, -2, 0)",
            "direction": "More negative slope = worse change",
            "comparability rating": "moderate",
            "best current validation use": "PTS adapted validation and secondary M2-style sensitivity work",
            "most defensible validation level": "Adapted validation or Level C",
            "source evidence": "Gateway do-file June 2025; KEIS health mapping; CHARLS PTS rule logic; prior K4 implementation",
            "notes": "This is transportable as a pragmatic change signal, but only with clear threshold adaptation because the underlying construct is grip strength rather than the CHARLS physical-limitation count.",
        },
        {
            "priority target": "Supportive outcome",
            "CHARLS variable/domain": "Post-landmark mortality",
            "CHARLS role": "Supportive outcome",
            "Preferred Gateway harmonized KLoSA variable(s)": "r4iwstat with support from radyear/radmonth and linked exit information",
            "Preferred raw KLoSA candidate(s)": "Wave-specific exit files linked by pid",
            "wave availability for primary wave3→4 validation": "Wave 4 exit ascertainment feasible",
            "scale/range": "Binary death event with date fields",
            "direction": "Death event after landmark",
            "comparability rating": "moderate",
            "best current validation use": "Supportive replication only",
            "most defensible validation level": "Level C immediately; Level B on harmonized build",
            "source evidence": "Gateway do-file June 2025; local exit-file audit",
            "notes": "Useful for disability-versus-mortality pattern checks, but not the first external validation target.",
        },
        {
            "priority target": "Primary target",
            "CHARLS variable/domain": "M1 current-burden model",
            "CHARLS role": "Model-level target",
            "Preferred Gateway harmonized KLoSA variable(s)": "Age + sex + education + rural proxy + harmonized depression burden + harmonized cognition construct + grip-based physical proxy + multimorbidity count",
            "Preferred raw KLoSA candidate(s)": "Same predictor structure using raw age/sex/education/region, w03dep1, cognition fallback if needed, w03mgrip, and w03chronic_sum",
            "wave availability for primary wave3→4 validation": "Wave 3 predictors and wave 4 ADL outcome supported; harmonized build now locally unblocked by bukpmcwp.do",
            "scale/range": "Logistic model",
            "direction": "Higher predicted probability = higher incident ADL risk",
            "comparability rating": "moderate",
            "best current validation use": "First-priority external validation target",
            "most defensible validation level": "Level C already implemented; Level B still conditional",
            "source evidence": "Gateway do-file June 2025; KEIS mappings; local raw audit",
            "notes": "The key update is that M1 is no longer blocked by missing cognition in principle. The honest remaining issue is not availability but whether the final KLoSA constructs are close enough to support frozen-coefficient transport versus recalibrated transport or structural replication.",
        },
        {
            "priority target": "Second-priority target",
            "CHARLS variable/domain": "PTS_6_core pragmatic rule",
            "CHARLS role": "Rule-level target",
            "Preferred Gateway harmonized KLoSA variable(s)": "Current depression burden + multimorbidity count + grip-based physical proxy + pre-landmark depressive and physical slopes",
            "Preferred raw KLoSA candidate(s)": "Depression summary, chronic_sum, mgrip, and participant-specific slopes across waves 1-3",
            "wave availability for primary wave3→4 validation": "Waves 1-3 components available; wave 3 landmark and wave 4 outcome supported",
            "scale/range": "Weighted point rule",
            "direction": "Higher score = higher risk",
            "comparability rating": "moderate",
            "best current validation use": "Second-priority pragmatic transportability target",
            "most defensible validation level": "Adapted external validation",
            "source evidence": "Gateway do-file June 2025; CHARLS PTS rule script; prior K4 implementation",
            "notes": "This is more transportable than the full phenotype because it tolerates percentile-based threshold adaptation while preserving the clinical rule logic.",
        },
        {
            "priority target": "Optional only",
            "CHARLS variable/domain": "4-class phenotype transport",
            "CHARLS role": "Exploratory subgroup resemblance",
            "Preferred Gateway harmonized KLoSA variable(s)": "Repeated depression, cognition, physical, and multimorbidity features after full harmonized reconstruction",
            "Preferred raw KLoSA candidate(s)": "Exploratory only; not recommended before M1 and PTS are complete",
            "wave availability for primary wave3→4 validation": "Potentially feasible only after the harmonized route is proven",
            "scale/range": "Feature-based clustering",
            "direction": "Not primary validation target",
            "comparability rating": "low",
            "best current validation use": "K5 only if earlier stages succeed",
            "most defensible validation level": "Exploratory transportability check only",
            "source evidence": "Locked project positioning; Gateway do-file June 2025",
            "notes": "Do not prioritise this over M1 and PTS_6_core.",
        },
    ]
    return pd.DataFrame(rows)


def build_markdown(
    wave_df: pd.DataFrame,
    harmonized_paths: list[Path],
    stata_binary: Path | None,
    imputation_info: dict,
    exit_info: dict,
) -> str:
    w = wave_df[wave_df["file_group"] == "w"][["wave", "rows"]].sort_values("wave")
    s = wave_df[wave_df["file_group"] == "str"][["wave", "rows"]].sort_values("wave")
    lt = wave_df[wave_df["file_group"] == "Lt"][["wave", "rows"]].sort_values("wave")
    w05_new = wave_df.loc[wave_df["file_name"] == "w05_new_e.dta", "rows"].iloc[0]

    lines = []
    lines.append("# KLoSA Feasibility Audit for External Validation")
    lines.append("")
    lines.append("## Audit purpose")
    lines.append(
        "This audit was conducted to determine whether KLoSA can support an honest external validation or transportability check of the locked CHARLS prognostic framework. The priority target is the wave-3 current-burden model (M1), followed by the pragmatic high-risk rule PTS_6_core or the closest defensible adapted analogue. Full phenotype transport is not treated as the default target."
    )
    lines.append("")
    lines.append("## Local data availability")
    if harmonized_paths:
        lines.append(
            f"- Local harmonized KLoSA directories detected: {', '.join(str(p) for p in harmonized_paths)}"
        )
        lines.append("- A harmonized build output has already been created locally and can now be used directly for cohort definition and structural external replication.")
    else:
        lines.append("- No local Gateway Harmonized KLoSA directory was detected.")
    lines.append(f"- Raw KLoSA STATA package detected: `{RAW_DIR}`")
    if HARMONIZED_DO_FILE.exists():
        lines.append(f"- Official Gateway Harmonized KLoSA creation code detected locally: `{HARMONIZED_DO_FILE}`")
    else:
        lines.append("- Official Gateway Harmonized KLoSA creation code was not detected locally.")
    lines.append("- The official Gateway Harmonized KLoSA creation-code page states that harmonized version F is built from the combined 1-9 wave STATA file plus the 1-7 wave imputation file and exit files for waves 2-9.")
    if imputation_info["has_outer_zip"]:
        lines.append(f"- Local imputation package detected: `{imputation_info['outer_zip']}`")
    else:
        lines.append("- No local imputation package was detected.")
    if imputation_info["has_nested_stata_zip"]:
        lines.append("- The imputation package contains the nested STATA imputation archive required for official harmonized construction.")
    if exit_info["has_w2_to_w6_extracted"]:
        lines.append("- Exit files for waves 2-6 are already extracted locally as Stata datasets.")
    else:
        lines.append("- Exit files for waves 2-6 are not yet complete as extracted Stata datasets.")
    lines.append(
        f"- Exit packages for waves 7-9 detected locally: wave 7 = {exit_info['has_w7_zip']}, wave 8 = {exit_info['has_w8_zip']}, wave 9 = {exit_info['has_w9_zip']}."
    )
    if stata_binary:
        lines.append(f"- Local Stata batch executable detected: `{stata_binary}`")
    else:
        lines.append("- No local Stata batch executable was detected.")
    lines.append("- Taken together, the local machine now has the core raw inputs, official creation code, software, and a completed local harmonized KLoSA build.")
    lines.append("- The present audit therefore distinguishes between what has already been implemented through the harmonized route and what still remains too weak for a strict transport claim.")
    lines.append("")
    lines.append("## Available waves and file structure")
    lines.append("- Raw wave files are available from wave 1 to wave 9.")
    lines.append("- Three recurring file families are present: `w` (main wave file), `str` (derived summary file), and `Lt` (more filtered/derived longitudinal file).")
    lines.append(f"- An additional wave-5 file `w05_new_e.dta` is present ({w05_new} rows) and should not be treated as a full replacement for the standard wave-5 file.")
    lines.append("")
    lines.append("### Row counts by file family")
    lines.append("")
    lines.append("| Wave | w rows | str rows | Lt rows |")
    lines.append("|---|---:|---:|---:|")
    for wave in range(1, 10):
        w_rows = int(w.loc[w["wave"] == wave, "rows"].iloc[0])
        s_rows = int(s.loc[s["wave"] == wave, "rows"].iloc[0])
        lt_rows = int(lt.loc[lt["wave"] == wave, "rows"].iloc[0])
        lines.append(f"| {wave} | {w_rows} | {s_rows} | {lt_rows} |")
    lines.append("")
    lines.append("## Variable/domain feasibility")
    lines.append("")
    lines.append("### ADL and IADL outcomes")
    lines.append("- Direct summary variables were identified in the `str` files, including `w03adl`, `w04adl`, `w05adl`, `w09adl`, and corresponding `iadl` summaries.")
    lines.append("- ADL and IADL outcome construction therefore appears feasible for a landmark incident-disability design.")
    lines.append("")
    lines.append("### Depressive symptoms")
    lines.append("- Direct depression-status summaries (`dep1`) were identified in the local English KLoSA package.")
    lines.append("- More importantly, the official Gateway harmonized build explicitly constructs wave-specific CES-D summaries (`r1-r4cesd10a`, `r5-r9cesd10b`), which substantially improves cross-cohort comparability for the depression domain.")
    lines.append("- The depressive domain should therefore no longer be treated as a weakly mapped construct; it is available, but strict frozen-coefficient transport still depends on how closely the final harmonized burden scale matches the locked CHARLS definition.")
    lines.append("")
    lines.append("### Cognition")
    lines.append("- Cognition should now be regarded as conditionally available rather than absent.")
    lines.append("- The KEIS mapping tables list `mmse`/`mmseg` summary candidates, and the official Gateway creation code clearly reconstructs cognition from wave-specific imputation items (`C401-C419`) into harmonized component measures such as orientation, recall, serial 7s, drawing, sentence repetition, and reading/writing tasks.")
    lines.append("- The remaining issue is therefore not whether cognition exists, but whether the final harmonized KLoSA cognition representation can be mapped cleanly enough to support recalibrated transport or should instead be used for structural external replication.")
    lines.append("")
    lines.append("### Physical limitation / function")
    lines.append("- The physical-function domain is still the weakest exact-match domain.")
    lines.append("- The official Gateway creation code provides harmonized grip-strength measures (`r?gripsum`) and test-completion flags, which are stronger than the earlier ad hoc raw audit but still remain proxies rather than exact analogues of the CHARLS physical-limitation count.")
    lines.append("- Physical mapping is therefore defensible for structural replication and pragmatic-rule adaptation, but not yet for unqualified exact transport.")
    lines.append("")
    lines.append("### Multimorbidity")
    lines.append("- Direct chronic-disease counts (`chronic_sum`) were identified in early `str` waves and can likely be derived from raw disease items where not directly provided.")
    lines.append("- Multimorbidity is one of the strongest and most transportable domain mappings.")
    lines.append("")
    lines.append("### Base covariates")
    lines.append("- Age is directly available as generated age variables.")
    lines.append("- Sex is directly available.")
    lines.append("- Education is directly available and should support low-education recoding.")
    lines.append("- KLoSA contains region categories that can support an urban-rural proxy, but not an exact hukou analogue.")
    lines.append("")
    lines.append("### Community-dwelling or institutionalisation status")
    lines.append("- A clean respondent-level institutionalisation flag was not identified in the current local package.")
    lines.append("- Residential-type variables exist (`enu_type`), but they do not yet appear to offer a strong community-dwelling versus institutionalised split.")
    lines.append("- A strict community-dwelling-only replication may therefore be difficult to implement cleanly.")
    lines.append("")
    lines.append("### Mortality support")
    lines.append("- Respondent-linked exit files are now available locally for waves 2-9, either as extracted Stata datasets or as local zip packages containing Stata datasets, and harmonized interview-status fields (`r?iwstat`) are also available.")
    lines.append("- Post-landmark mortality ascertainment therefore now appears technically feasible, although it has not yet been integrated into the main validation cohort pipeline.")
    lines.append("")
    lines.append("## Feasibility judgment by validation level")
    lines.append("")
    lines.append("### Level A: exact transport validation")
    lines.append("- **Judgment: still not fully feasible as a strict claim.**")
    lines.append("- Rationale: although harmonized KLoSA now exists locally, the physical-function domain is still proxy based, the rural-hukou term remains a Korea-specific urban-rural proxy, and community-dwelling restriction is still not cleanly mirrored.")
    lines.append("")
    lines.append("### Level B: recalibrated transport validation")
    lines.append("- **Judgment: conditionally feasible in principle, but still not the most defensible executed result.**")
    lines.append("- Rationale: depression and cognition are now supported through the completed harmonized build. However, recalibrated transport still depends on accepting the grip-based physical proxy and the Korea-specific rural indicator as close enough to the CHARLS burden structure, which remains debatable.")
    lines.append("")
    lines.append("### Level C: structural external replication")
    lines.append("- **Judgment: feasible, implemented, and currently the most defensible target.**")
    lines.append("- Rationale: KLoSA now supports age, sex, education, an urban-rural proxy, ADL outcome, multimorbidity, harmonized depression burden, and a harmonized cognition reconstruction route, while physical function remains a transparent grip-based proxy domain. That is sufficient for a structure-preserving external replication of current-burden risk stratification, even if strict coefficient transport is not yet justified.")
    lines.append("")
    lines.append("## Priority targets implied by the audit")
    lines.append("- **First priority:** M1 current-burden model, now best framed as an implemented structural external replication with recalibrated transport reserved as a future upgrade only if predictor equivalence can be defended more strongly.")
    lines.append("- **Second priority:** PTS_6_core or the closest pragmatic adapted rule, because pragmatic risk identification is more likely to travel across cohorts than a GMM-derived phenotype and does not require exact phenotype transport.")
    lines.append("- **Not the default target:** full 4-class phenotype transport. This should remain optional and exploratory only if M1 and pragmatic-rule validation are both feasible and well supported.")
    lines.append("")
    lines.append("## Recommendation for the next stage")
    lines.append("- Use the completed harmonized build as the default source for the wave3->4 KLoSA landmark validation cohort.")
    lines.append("- Keep the primary naming conservative: the current safest label remains **structural external replication**.")
    lines.append("- Treat **recalibrated transport validation** as a future upgrade only if the final predictor equivalence argument can be made convincingly enough for peer review.")
    lines.append("")
    return "\n".join(lines)


def format_crosswalk_workbook(path: Path) -> None:
    wb = load_workbook(path)
    ws = wb["crosswalk"]
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    width_map = {
        "A": 18,
        "B": 34,
        "C": 24,
        "D": 42,
        "E": 42,
        "F": 30,
        "G": 26,
        "H": 24,
        "I": 22,
        "J": 30,
        "K": 28,
        "L": 28,
        "M": 34,
    }
    for col, width in width_map.items():
        ws.column_dimensions[col].width = width

    raw_ws = wb["raw_wave_counts"]
    raw_ws.freeze_panes = "A2"
    raw_ws.auto_filter.ref = raw_ws.dimensions
    wb.save(path)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    wave_df = build_wave_summary()
    harmonized_paths = detect_harmonized_klosa()
    stata_binary = detect_stata_binary()
    imputation_info = detect_imputation_packages()
    exit_info = detect_exit_sources()
    crosswalk = build_crosswalk()

    md_text = build_markdown(wave_df, harmonized_paths, stata_binary, imputation_info, exit_info)
    (RESULTS_DIR / "klosa_feasibility_audit.md").write_text(md_text, encoding="utf-8")

    with pd.ExcelWriter(RESULTS_DIR / "charls_to_klosa_crosswalk.xlsx", engine="openpyxl") as writer:
        crosswalk.to_excel(writer, index=False, sheet_name="crosswalk")
        wave_df.to_excel(writer, index=False, sheet_name="raw_wave_counts")
    crosswalk.to_csv(RESULTS_DIR / "charls_to_klosa_crosswalk.csv", index=False)
    wave_df.to_csv(RESULTS_DIR / "klosa_raw_wave_counts.csv", index=False)
    format_crosswalk_workbook(RESULTS_DIR / "charls_to_klosa_crosswalk.xlsx")

    print("Wrote:")
    print(RESULTS_DIR / "klosa_feasibility_audit.md")
    print(RESULTS_DIR / "charls_to_klosa_crosswalk.xlsx")
    print(RESULTS_DIR / "charls_to_klosa_crosswalk.csv")
    print(RESULTS_DIR / "klosa_raw_wave_counts.csv")


if __name__ == "__main__":
    main()

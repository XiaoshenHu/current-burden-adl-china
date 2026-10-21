from __future__ import annotations

from pathlib import Path

# Resolve repository imports independently of the launch directory.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from data_processing_scripts.klosa_processing.harmonized_wave34_m1_utils import (
    RESULTS_DIR,
    build_wave34_landmark_riskset,
    build_wave34_structural_m1_complete_case,
    get_wave34_sample_flow_counts,
)


def build_sample_flow_csv(counts: dict[str, float]) -> pd.DataFrame:
    rows = [
        {
            "step_order": 1,
            "step": "Wave 3 harmonized interview sample",
            "count": int(counts["wave3_interview_alive"]),
            "notes": "Interviewed and alive at wave 3 in GH_KLoSA_f.",
        },
        {
            "step_order": 2,
            "step": "Age >=45 at wave 3",
            "count": int(counts["age45plus"]),
            "notes": "Retained wave-3 interviewees who met the KLoSA age threshold used in the CHARLS target study.",
        },
        {
            "step_order": 3,
            "step": "ADL-disability-free at landmark",
            "count": int(counts["adl_free_landmark"]),
            "notes": "Defined as r3adltotba_k = 0.No before the next-wave outcome window.",
        },
        {
            "step_order": 4,
            "step": "Observed wave-4 ADL outcome",
            "count": int(counts["observed_wave4_adl_among_free"]),
            "notes": "Non-missing r4adltotba_k among the wave-3 ADL-free landmark risk set.",
        },
        {
            "step_order": 5,
            "step": "Complete structural-M1 predictors",
            "count": int(counts["complete_structural_m1"]),
            "notes": "Landmark age, sex, low education, rural proxy, CES-D burden, cognition-derived deficit, grip-based physical proxy, and multimorbidity all observed.",
        },
    ]
    return pd.DataFrame(rows)


def build_markdown(counts: dict[str, float]) -> str:
    event_n = int(counts["complete_event_n"])
    event_rate = float(counts["complete_event_rate"])
    lines = [
        "# KLoSA Validation Cohort Definition",
        "",
        "## Goal of cohort definition",
        "The immediate goal is to define a KLoSA landmark risk set that is maximally comparable to the locked CHARLS incident-ADL design, while remaining honest about cross-cohort differences in measurement and cohort structure.",
        "",
        "## Primary validation interval",
        "- **Chosen interval:** wave 3 to wave 4",
        "- **Why this interval was prioritised:** it matches the CHARLS landmark logic of pre-landmark information followed by next-wave incident ADL disability, and the harmonized build provides the cleanest current-burden inputs for this interval.",
        "",
        "## Landmark risk-set definition",
        "- Interviewed and alive at wave 3 (`inw3 = 1.resp,alive`).",
        "- Age 45 years or older at wave 3.",
        "- Free of ADL disability at wave 3, defined as `r3adltotba_k = 0.No`.",
        "- Incident ADL disability assessed at wave 4 using `r4adltotba_k` among those with non-missing next-wave ADL information.",
        "- A strict community-dwelling-only restriction was **not** imposed because no clean institutionalisation flag was identified in the harmonized output.",
        "",
        "## Current-burden predictors used for the structural replication target",
        "- Landmark age at wave 3.",
        "- Female sex.",
        "- Low education from the harmonized education recode (`raeducl`).",
        "- Rural-urban proxy from `r3rural`.",
        "- Depressive symptoms from harmonized CES-D-10 total (`r3cesd10a`).",
        "- Cognitive deficit reconstructed as `25 - cognitive total`, where the total summed orientation, date naming, immediate recall, serial 7s, drawing, repetition, reading, and sentence-writing components.",
        "- Physical limitation represented by a within-sex grip-based weakness percentile (`1 - percentile rank of r3gripsum`), used explicitly as a proxy rather than as an exact analogue of the CHARLS five-task physical-limitation summary.",
        "- Multimorbidity represented by a seven-condition count from harmonized disease indicators (`hibpe`, `diabe`, `hearte`, `stroke`, `arthre`, `lunge`, `cancre`).",
        "",
        "## Sample flow for the chosen interval",
        f"- Wave 3 harmonized interview sample: {int(counts['wave3_interview_alive'])}",
        f"- Age >=45 at wave 3: {int(counts['age45plus'])}",
        f"- ADL-disability-free at landmark: {int(counts['adl_free_landmark'])}",
        f"- Observed wave-4 ADL outcome: {int(counts['observed_wave4_adl_among_free'])}",
        f"- Complete structural-M1 predictors: {int(counts['complete_structural_m1'])}",
        "",
        "## Event yield in the complete structural-replication cohort",
        f"- Incident ADL events: {event_n}",
        f"- Incident ADL event risk: {event_rate:.4f}",
        "",
        "## What this cohort can and cannot support",
        "- This cohort now clearly supports a harmonized KLoSA **structural external replication** of the current-burden predictor structure.",
        "- It still does **not** support an unqualified exact transport validation, because the physical-function domain is represented by an explicit grip-based proxy and the rural term remains a Korea-specific urban-rural analogue rather than a hukou construct.",
        "- It also does not yet support a strict community-dwelling-only replication because institutionalisation status remains unclear in the available harmonized fields.",
        "",
        "## Implication for model validation",
        "This cohort supports a harmonized KLoSA structural external replication of the M1 current-burden framework, with exact or recalibrated transport reserved for future work only if predictor definitions can be shown to be sufficiently equivalent.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _ = build_wave34_landmark_riskset()
    _ = build_wave34_structural_m1_complete_case()
    counts = get_wave34_sample_flow_counts()

    sample_flow = build_sample_flow_csv(counts)
    sample_flow.to_csv(RESULTS_DIR / "klosa_sample_flow.csv", index=False)

    md = build_markdown(counts)
    (RESULTS_DIR / "klosa_validation_cohort_definition.md").write_text(md, encoding="utf-8")

    print("Wrote:")
    print(RESULTS_DIR / "klosa_validation_cohort_definition.md")
    print(RESULTS_DIR / "klosa_sample_flow.csv")


if __name__ == "__main__":
    main()

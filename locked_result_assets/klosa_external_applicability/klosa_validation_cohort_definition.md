# KLoSA Validation Cohort Definition

## Goal of cohort definition
The immediate goal is to define a KLoSA landmark risk set that is maximally comparable to the locked CHARLS incident-ADL design, while remaining honest about cross-cohort differences in measurement and cohort structure.

## Primary validation interval
- **Chosen interval:** wave 3 to wave 4
- **Why this interval was prioritised:** it matches the CHARLS landmark logic of pre-landmark information followed by next-wave incident ADL disability, and the harmonized build provides the cleanest current-burden inputs for this interval.

## Landmark risk-set definition
- Interviewed and alive at wave 3 (`inw3 = 1.resp,alive`).
- Age 45 years or older at wave 3.
- Free of ADL disability at wave 3, defined as `r3adltotba_k = 0.No`.
- Incident ADL disability assessed at wave 4 using `r4adltotba_k` among those with non-missing next-wave ADL information.
- A strict community-dwelling-only restriction was **not** imposed because no clean institutionalisation flag was identified in the harmonized output.

## Current-burden predictors used for the structural replication target
- Landmark age at wave 3.
- Female sex.
- Low education from the harmonized education recode (`raeducl`).
- Rural-urban proxy from `r3rural`.
- Depressive symptoms from harmonized CES-D-10 total (`r3cesd10a`).
- Cognitive deficit reconstructed as `25 - cognitive total`, where the total summed orientation, date naming, immediate recall, serial 7s, drawing, repetition, reading, and sentence-writing components.
- Physical limitation represented by a within-sex grip-based weakness percentile (`1 - percentile rank of r3gripsum`), used explicitly as a proxy rather than as an exact analogue of the CHARLS five-task physical-limitation summary.
- Multimorbidity represented by a seven-condition count from harmonized disease indicators (`hibpe`, `diabe`, `hearte`, `stroke`, `arthre`, `lunge`, `cancre`).

## Sample flow for the chosen interval
- Wave 3 harmonized interview sample: 7920
- Age >=45 at wave 3: 7920
- ADL-disability-free at landmark: 7555
- Observed wave-4 ADL outcome: 6987
- Complete structural-M1 predictors: 6208

## Event yield in the complete structural-replication cohort
- Incident ADL events: 145
- Incident ADL event risk: 0.0234

## What this cohort can and cannot support
- This cohort now clearly supports a harmonized KLoSA **structural external replication** of the current-burden predictor structure.
- It still does **not** support an unqualified exact transport validation, because the physical-function domain is represented by an explicit grip-based proxy and the rural term remains a Korea-specific urban-rural analogue rather than a hukou construct.
- It also does not yet support a strict community-dwelling-only replication because institutionalisation status remains unclear in the available harmonized fields.

## Implication for model validation
This cohort supports a harmonized KLoSA structural external replication of the M1 current-burden framework, with exact or recalibrated transport reserved for future work only if predictor definitions can be shown to be sufficiently equivalent.

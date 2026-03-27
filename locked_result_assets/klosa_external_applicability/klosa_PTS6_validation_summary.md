# KLoSA PTS_6_core validation summary

## Validation label
- **Implemented label:** adapted external validation
- **Exact transport:** not feasible, because the KLoSA physical component is a grip-strength proxy rather than the original CHARLS physical-limitation count, and the depressive component is binary rather than continuous.
- **Adapted transport:** feasible, because all pragmatic-rule components can be mapped to KLoSA with explicit scale adaptation and percentile-based threshold mapping.

## Adapted rule definition used in KLoSA
- Landmark interval: waves 1-3 for feature derivation, wave 3 as the landmark, and wave 4 incident ADL disability as the next-wave outcome.
- Current depression flag: wave-3 depression status (`dep1 = 1`).
- Current multimorbidity flag: wave-3 chronic disease count at or above the overall 75th percentile.
- Current physical flag: wave-3 grip strength at or below the within-sex 25th percentile.
- Depression-change flag: participant-specific pre-wave-3 depression slope at or above the overall 75th percentile.
- Physical-change flag: participant-specific pre-wave-3 grip slope at or below the within-sex 10th percentile.
- Score construction preserved the locked CHARLS weight structure: `2*physical_current + depression_current + multimorbidity_current + depression_change + physical_change`.
- High risk was defined as score >=4 and low risk as score <=1, consistent with the locked CHARLS rule structure.

## Performance and clinical usability
- Complete adapted-rule sample: 5801 participants with 108 incident ADL events (overall event risk 0.019).
- Score-based AUC: 0.689.
- High-risk group: 762 participants (13.136%) with observed incident ADL risk 0.042.
- Low-risk group: 3238 participants (55.818%) with observed incident ADL risk 0.009.
- High-versus-low absolute risk difference: 0.033.

## Comparison with locked CHARLS rule performance
- Locked CHARLS reference values for PTS_6_core were: high-risk share about 14.6%, high-risk observed risk about 0.300, low-risk observed risk about 0.048, and OOF AUC about 0.710.
- The KLoSA adapted rule preserved a similar high-risk group size but showed a lower absolute event rate, which is consistent with the lower short-horizon incident ADL risk in the KLoSA validation interval.
- Direct head-to-head comparison with KLoSA M1 is not yet available, because M1 transport/replication remains blocked by unresolved cognition mapping.

## Honest interpretation
- KLoSA can support a meaningful **adapted external validation** of the pragmatic rule structure.
- The rule still identifies a smaller subgroup with a materially elevated short-horizon incident ADL risk, even under cross-cohort scale adaptation.
- This is best interpreted as evidence for pragmatic transportability of simplified high-risk identification, not as strict transport validation of the original CHARLS score thresholds.

## Threshold details used for the adapted validation
- physical proxy thresholds (sex code 1): wave-3 grip strength <= within-sex 25th percentile = 26.500; pre-wave-3 grip slope <= within-sex 10th percentile = -2.625.
- physical proxy thresholds (sex code 5): wave-3 grip strength <= within-sex 25th percentile = 16.031; pre-wave-3 grip slope <= within-sex 10th percentile = -1.750.
- multimorbidity current: wave-3 chronic count >= overall 75th percentile = 2.000.
- depression change: wave-3 depression status = 1 = 1.000; pre-wave-3 depression slope >= overall 75th percentile = 0.000.

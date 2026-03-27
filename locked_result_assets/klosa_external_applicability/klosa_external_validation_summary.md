# KLoSA External Validation and Transportability Summary

## Project scope and priority

This KLoSA project was designed as a transportability check for the locked CHARLS prognostic framework rather than a re-run of the original discovery analysis. The priority targets were the current-burden model (M1) and the simplified pragmatic rule PTS_6_core, not formal transport of the 4-class phenotype solution.

## What has now been completed

### Stage K1: feasibility and mapping audit

- Raw KLoSA waves 1-9 were available locally.
- The official Gateway Harmonized KLoSA creation code (`bukpmcwp.do`) was obtained locally.
- Local Stata batch execution was verified and used successfully.
- Gateway Harmonized KLoSA version F was built locally and audited at the file level.
- The crosswalk was hardened for the main M1 and PTS_6_core components, with depression, cognition, ADL, education, rural proxy, and multimorbidity now anchored to the harmonized build.
- The main remaining transport limitation is not missing data availability, but imperfect predictor equivalence, especially for physical function and the rural-hukou analogue.

### Stage K2: external validation cohort definition

- The primary external validation interval was defined as **wave 3 to wave 4**.
- The harmonized KLoSA landmark risk-set counts were:
  - wave 3 harmonized interview sample: 7920
  - age >=45: 7920
  - ADL-disability-free at landmark: 7555
  - observed wave-4 ADL outcome: 6987
  - complete structural-M1 predictors: 6208
- A strict community-dwelling-only restriction was not imposed because no clean institutionalisation flag was identified in the harmonized output.

### Stage K3: M1 current-burden validation

- **M1 has now been implemented in KLoSA as a structural external replication.**
- It was not labelled as exact external validation because the physical-function domain remains grip-based and the rural term remains a Korea-specific urban-rural analogue rather than a hukou construct.
- It was not labelled as recalibrated transport because, despite the harmonized build, predictor equivalence is still not close enough to defend frozen-coefficient or intercept/slope-recalibrated transport as the primary claim.
- The implemented design was a fixed-seed 10-fold out-of-fold structural replication in the wave3->wave4 landmark cohort.

### Stage K4: PTS_6_core validation

- **PTS_6_core remains implemented as an adapted external validation.**
- Exact rule transport was not defensible because the physical component had to be represented by grip-based proxies and the depression/current-burden components required cohort-specific mapping.
- Even so, the pragmatic rule retained clinically meaningful high-risk separation in KLoSA.

### Stage K5: phenotype transport

- Not pursued.
- This remains deliberately non-prioritised because the current paper's locked conclusion is driven by current multidomain burden and pragmatic transportability, not by full transport of the original clustering solution.

## Key KLoSA results obtained so far

### M1 structural external replication

- Validation label: **structural external replication**
- Complete validation cohort: 6208
- Incident ADL events: 145
- Overall event risk: 0.0234
- Out-of-fold AUC: 0.830
- Brier score: 0.022
- Calibration intercept: 0.002
- Calibration slope: 0.945
- Top risk quintile:
  - share: 0.200
  - observed event risk: 0.082
- Top risk decile:
  - share: 0.100
  - observed event risk: 0.111
- Calibration pattern:
  - overall acceptable
  - mild overprediction in the highest-risk decile
  - lower deciles noisier because of the low event rate

### PTS_6_core adapted external validation

- Validation label: **adapted external validation**
- Complete adapted-rule sample: 5801
- Incident ADL events: 108
- Overall event risk: 0.0186
- Score-based AUC: 0.689
- High-risk group:
  - 762 participants
  - 13.1% of the cohort
  - observed incident ADL risk: 0.0420
- Low-risk group:
  - 3238 participants
  - 55.8% of the cohort
  - observed incident ADL risk: 0.00865
- High-versus-low absolute risk difference: 0.0333

## Direct answers to the required questions

### 1. Can KLoSA provide a meaningful external validation or structural external replication for M1?

**Yes, but the most honest implemented result is structural external replication rather than formal transport validation.**

KLoSA now provides a serious wave3->wave4 landmark validation cohort and a completed structural replication of the M1 current-burden predictor structure. The result is meaningful and clearly supports external applicability of the current-burden signal. However, it should not be overstated as exact or recalibrated transport because predictor equivalence remains imperfect.

### 2. Can KLoSA provide a meaningful real-world validation for PTS_6_core?

**Yes, as an adapted external validation.**

The adapted KLoSA rule preserved a clinically meaningful smaller high-risk subgroup and separated incident ADL risk between high- and low-risk groups. That is useful external evidence for pragmatic risk identification, even though it is not strict one-to-one transport of the original CHARLS rule.

### 3. Which validation level is the most honest and defensible?

- For **M1**, the most honest implemented label is:
  - **structural external replication**
- For **PTS_6_core**, the most honest implemented label is:
  - **adapted external validation**

### 4. If this KLoSA work were added to the current paper, what is the most appropriate name?

The most defensible manuscript-level wording would be:

- **exploratory transportability and external applicability check in KLoSA**

If a more specific label is needed:

- for M1: **structural external replication**
- for PTS_6_core: **adapted external validation**

It should **not** be called formal external validation.

### 5. Which result is most worth adding to the current paper, and which is better left for a follow-up study?

The result most worth adding to the current paper is the **M1 structural external replication**, because it directly supports the paper's locked primary message that current multidomain burden is the dominant prognostic signal. The **PTS_6_core adapted validation** is also useful, but it is more naturally a supportive translational extension than a central headline.

What is better left for a follow-up study is:

- any attempt to rename the current KLoSA M1 result as recalibrated transport validation,
- and any attempt at full phenotype transport or phenotype reconstruction.

Those would require a stronger equivalence argument and a cleaner methodological paper of their own.

## Overall conclusion

KLoSA is now sufficient to support:

- a completed harmonized feasibility and mapping audit,
- a clean wave3->wave4 external landmark cohort,
- a meaningful **structural external replication** of M1,
- and an already implemented **adapted external validation** of PTS_6_core.

The current KLoSA contribution is therefore real and manuscript-usable, but it should be presented conservatively. The safest framing is an **exploratory transportability and external applicability check**, with the M1 result carrying the most direct value for the current paper.

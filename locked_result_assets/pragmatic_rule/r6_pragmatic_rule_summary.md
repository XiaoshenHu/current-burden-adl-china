# Stage R6 Pragmatic Rule Summary

## Stage aim

This stage did not develop additional new rules. Instead, it integrated the Stage R1-R5 results into a final judgment about whether a reportable pragmatic rule had been obtained, how much performance it lost relative to the full `M2` model, and how it should be positioned in the manuscript.

## Direct answers to the core scientific questions

### 1. Can a more operational high-risk identification rule be derived from M2?
Yes, but three objects should be distinguished:
- `PRM_core5`: a 5-variable simplified regression model closest to full `M2/M5`
- `PTS_6_core`: the most practical point-based score
- `HYB_emerging_nonsevere`: a narrower pre-disability operational flag

Among these, the most reportable pragmatic object is `PTS_6_core`, not `HYB_emerging_nonsevere`.

### 2. What is the main signal carried by the rule?
The most stable rule signal came from a combination of current levels and selective slopes rather than slopes alone. In practical terms, the rule is anchored by current burden and refined by limited pre-landmark change information.

### 3. Does the rule more closely identify a pre-disability high-risk group?
Yes, mainly in operational targeting rather than overall AUC.

- `PTS_6_core >= 4`: 14.614% of the sample; observed incident ADL risk 0.300; absolute risk difference versus the low-risk reference group 0.252
- `HYB_emerging_nonsevere`: 6.834% of the sample; observed incident ADL risk 0.285; absolute risk difference versus the low-risk reference group 0.238

Accordingly, `PTS_6_core` behaves like a broader operational high-risk score, whereas `HYB_emerging_nonsevere` behaves more like a narrow pre-disability flag.

### 4. How does the rule relate to the Emerging phenotype?
The most accurate interpretation is approximate operationalisation with only partial overlap.

- Within the `PTS_6_core` high-risk group, the proportion classified as `Emerging` was 0.482
- Within the `HYB` high-risk group, the proportion classified as `Emerging` was 0.510
- Neither rule captured all `Emerging` participants

These rules should therefore not be described as direct replacements or mechanical translations of the phenotype. A more defensible phrasing is that `PTS_6_core` operationalises a broad high-risk layer around M2-like risk, whereas `HYB` operationalises a narrower Emerging-like pre-disability flag.

### 5. Is the incremental manuscript value substantial, modest, or minimal?
The most defensible answer is modest.

- Full `M2` AUC = 0.769
- `PRM_core5` AUC = 0.769
- `PTS_6_core` AUC = 0.710
- `HYB` AUC = 0.536

The operational score is simpler but loses discrimination. It therefore adds clinical interpretability and operational relevance, not a new headline predictive conclusion.

### 6. What is the most defensible manuscript positioning?
The strongest wording is:

- a practical high-risk identification rule derived from the stable M2 signals

A weaker but still acceptable wording is:

- simplified prognostic score

Wording that should be avoided includes:

- screening tool ready for implementation
- new validated risk classifier
- replacement for the full multidomain model

## Final recommendation

### Best-performing simplified object
If the goal is to retain performance after simplification, the best object is `PRM_core5`, with AUC 0.769, essentially matching full `M2`. However, it remains a small regression model rather than the most usable bedside-style rule.

### Best operational score
If the goal is to report the most practical pragmatic rule, the best object is `PTS_6_core`.

- Complete-case OOF AUC 0.710
- Optimism 0.013
- OOF calibration slope 0.982
- Older-adult high-risk event rate remained approximately 0.359 to 0.386

Its value lies in operationalising part of the M2 high-risk group using a simpler scoring system while preserving materially elevated observed risk.

### Best selective pre-disability flag
`HYB_emerging_nonsevere` is better interpreted as an exploratory flag.

- Complete-case OOF AUC 0.536
- IPW OOF AUC 0.504
- Older-adult high-risk event rate approximately 0.358 to 0.373

Its strength is narrower targeting of a non-severe but high-risk subgroup; its weakness is poor overall discrimination.

## Bottom line

1. A more operational high-risk identification object can be derived from `M2`.
2. Its main signal reflects current burden first and pre-landmark change second.
3. `PTS_6_core` is the most reportable pragmatic rule.
4. `HYB_emerging_nonsevere` is better treated as an exploratory operational flag rather than a main manuscript rule.
5. The rule system adds modest value through clinical interpretability, not through a change in the main conclusions.
6. The main manuscript storyline should remain:
   - wave-3 current burden dominates
   - pre-landmark change adds modest refinement
   - the operational rule is an optional translational extension rather than a replacement

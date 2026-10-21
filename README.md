# CHARLS landmark prognostic stratification study: reproducibility release package

## Reproducibility status

Portable paths, repository imports, a pinned requirements file and stage-level
commands are provided. **Full statistical reruns remain blocked** by restricted
data and missing original CHARLS source/input dependencies; no statistical
results were regenerated in this repair. All released locked assets are unchanged.

- [Installation and path configuration](reproducibility/environment_requirements.md)
- [Execution commands and input/output relationships](reproducibility/execution_order.md)
- [Missing source and verification boundaries](reproducibility/missing_source_audit.md)
- [Reproducibility guide](reproducibility/reproducibility_guide.md)

After installation, start with:

```bash
python -m unittest discover -s tests -v
python reproducibility/run.py --init-workspace
python reproducibility/run.py --check all
```

The availability check intentionally exits with code 1 for missing inputs or
sources. This is expected for the public data-free checkout.

## Project title

Current multidomain burden dominates short-horizon ADL disability prediction, while pre-landmark multidomain change provides modest incremental refinement in middle-aged and older adults: a nationwide longitudinal CHARLS prognostic stratification study

## Main research question

How current multidomain burden and pre-landmark multidomain change together relate to subsequent ADL disability risk in middle-aged and older adults, and whether the locked framework shows external applicability in KLoSA.

## Current main conclusions

- Current multidomain burden at the landmark is the dominant prognostic signal.
- Pre-landmark multidomain change provides modest incremental refinement.
- Hard phenotype is clinically interpretable but weaker than the continuous current-burden-plus-change representation.
- KLoSA supports structural external replication of the current-burden framework and adapted external validation of the simplified pragmatic rule.

## What this release package contains

- Analytic code
- Data-processing scripts
- Locked aggregated result assets
- Reproducibility instructions
- Release manifests and QC notes

## What this release package does not contain

- Participant-level CHARLS data
- Participant-level KLoSA data
- Participant-level Gateway Harmonized KLoSA data
- Manuscript working drafts or manuscript-supporting folders
- Figure/table source-data CSV files
- Large obsolete deliverable folders and superseded drafts

## Data access

Users must obtain the underlying survey data through the official CHARLS, KLoSA/KEIS, and Gateway to Global Aging Data access routes.

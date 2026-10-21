# Notes on Unavailable Raw Data

This release package does not include any participant-level survey data. Users must obtain the relevant CHARLS and KLoSA data directly from the official providers. The harmonized KLoSA build output (`GH_KLoSA_f.dta`) is also not redistributed here.

Raw data access alone does not resolve the missing original reinforcement module
or the missing producer of the expanded v2 CHARLS input tree. These are separate
source-completeness blockers. See [missing_source_audit.md](missing_source_audit.md)
and the exact required-file contracts in [pipeline.json](pipeline.json).

Do not substitute aggregated files from `locked_result_assets/` for participant
datasets, phenotype assignments or posterior probabilities. No statistical
recalculation was performed during the portability repair.

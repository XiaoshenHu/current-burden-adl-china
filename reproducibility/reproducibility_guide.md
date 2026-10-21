# Reproducibility Guide

This release supports code inspection and locked-result traceability without
redistributing participant-level survey data. It is **not currently an end-to-end
rerunnable release**, even for a reader who obtains only the raw survey data:
two original CHARLS source/input dependencies are missing.

## Start here

1. [Install and configure](environment_requirements.md) the published package versions.
2. Run `python -m unittest discover -s tests -v` for data-free checks.
3. Read the [source completeness audit](missing_source_audit.md).
4. Follow the [exact commands and data flow](execution_order.md).
5. Use `python reproducibility/run.py --describe STAGE` and `--check STAGE`
   before attempting any analytical stage.

The [pipeline contract](pipeline.json) maps 28 executable stages to required
files, producer relationships and outputs. The [IO inventory](io_inventory.md)
lists explicit read/write sites, including dynamic paths. The second Gateway
entrypoint is an identical compatibility copy; the harmonized utility is a
helper, bringing the original source count to 30 Python files.

## Current readiness

| Component | Status after repair |
| --- | --- |
| All 30 released Python files | Importable with installed requirements; paths configurable |
| Locked aggregated results | All 28 files unchanged and protected by SHA-256 checks |
| Legacy CHARLS time-split branch | Requires licensed raw inputs; not numerically rerun |
| Legacy CHARLS master | Requires missing original reinforcement module and raw inputs |
| Expanded v2 CHARLS, M1 and rule branches | Requires original expanded v2 input package/producer, absent from this release |
| KLoSA structural M1 and adapted PTS6 | Imports repaired; requires licensed inputs; not numerically rerun |
| Gateway harmonization | Paths/Stata discovery repaired; requires official source/data and licensed Stata; build untested |

## Included

- Analytic code
- Data-processing scripts
- A curated subset of locked aggregated result assets supporting the main manuscript claims

## Excluded

- Participant-level CHARLS data
- Participant-level KLoSA data
- Participant-level Gateway Harmonized KLoSA data
- Working manuscript files and manuscript-supporting folders
- Figure and table source-data CSV files
- KLoSA build logs and other non-essential provenance files
- Superseded deliverables and obsolete manuscript drafts

## Practical implication

See the audit for the missing module's eight callables and the separate missing
expanded v2 producer. Neither has been replaced with a guessed implementation.
The legacy master outputs and expanded v2 inputs must not be treated as
interchangeable. Full statistical reproduction needs the original sources,
permitted data/input package and provenance, followed by numerical comparisons
against released assets. A passing import or unit test is not evidence that the
paper's statistical results have been independently reproduced.

Historical release manifests and original software records are retained as
provenance. Generated files use the ignored `work/` workspace by default.

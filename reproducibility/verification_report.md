# Repair verification

Date: 2026-10-21 (Asia/Shanghai). Baseline:
`73ba8f9caaaa9abe78904954fde031847ce551a2`.

| Check | Outcome |
| --- | --- |
| Install ten recorded direct package versions, Linux / Python 3.9.25 | Passed; dependency compatibility check passed |
| Install same direct versions, Linux / Python 3.12.14 | Passed; `pip check` passed |
| `python -m unittest discover -s tests -v`, both environments | 15 tests passed in each environment |
| Import all 30 original Python files from an unrelated directory | Passed; no participant files read or analysis workspace created |
| Configurable input/output roots and paths containing spaces | Passed |
| Missing inputs, original source and invalid explicit Stata path | Correctly blocked; no analytical subprocess started |
| Stage command / resolved input-output description | Passed |
| KLoSA audit prerequisite coverage | All 28 original wave-file reads are required before launch |
| Function-body and analysis-constant comparison to baseline | Passed with only documented execution-support changes allowed |
| All 28 released locked assets | SHA-256 values unchanged |
| Source compilation and `git diff --check` | Passed |

The tests include no survey-data model fitting, invented coefficients or
fabricated results. They validate execution support and preserve original
statistical code. Synthetic fixtures cover only paths, file availability and
executable discovery.

The exact original Python 3.9.6/transitive environment was not recovered. A few
upstream matplotlib/pyparsing deprecation warnings occur without test failures;
direct versions remain those in the original release. macOS and Windows have
not been executed in this repair.

CI pins Ubuntu 22.04 for Python 3.9 and Ubuntu 24.04 for Python 3.12. The initial
`ubuntu-latest` run reached Ubuntu 26.04, for which `setup-python` could not find
Python 3.9; no analytical tests ran in that job. Fixed runner versions avoid
that interpreter-installation failure and future implicit OS changes.

**Not verified:** full CHARLS/KLoSA sample derivation, all estimated effects,
performance/calibration metrics, uncertainty, figure/table regeneration, or
the licensed Stata harmonization. These require the original source/input
packages and restricted data described in [missing_source_audit.md](missing_source_audit.md).

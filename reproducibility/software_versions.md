# Software Versions

The following is the original release's recorded environment, not a complete
lockfile. The Stata path is historical provenance; runtime discovery now uses
`STATA_BIN`, `PATH`, or macOS application detection.

- Python: 3.9.6
- pandas: 2.3.3
- numpy: 2.0.2
- scipy: 1.13.1
- scikit-learn: 1.6.1
- statsmodels: 0.14.6
- lifelines: 0.30.0
- matplotlib: 3.9.4
- seaborn: 0.13.2
- openpyxl: 3.1.5
- python-docx: 1.2.0
- Stata executable used for Gateway Harmonized KLoSA build: /Applications/Stata/StataMP.app/Contents/MacOS/stata-mp

## Repair verification environment (2026-10-21)

- Linux x86_64; Python 3.9.25 and Python 3.12.14.
- The ten direct Python versions above installed from `requirements.txt`.
- Installation/dependency and data-free checks only; no Stata or survey-data run.
- The original transitive package set, Stata version, BLAS build and OS details
  were not recovered. See `environment_requirements.md` for this limitation.

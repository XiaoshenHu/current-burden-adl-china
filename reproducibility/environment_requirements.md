# Environment installation and configuration

The original release records Python **3.9.6** and the ten direct package versions
in [requirements.txt](../requirements.txt). These are the published versions;
the original transitive dependency lockfile and Stata version are unavailable.
The repair is checked on Linux with Python 3.9.25 and 3.12.14. These checks cover
installation, imports and execution support, not statistical equivalence.

## Install (macOS or Linux)

From the checkout root, using an installed Python 3.9 interpreter:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
MPLBACKEND=Agg python -m unittest discover -s tests -v
```

Python 3.12 can also run these data-free checks (`python3.12 -m venv .venv`).
To record a new environment locally, run
`python -m pip freeze > work/environment_freeze.txt` after initializing the
workspace. A new freeze is not evidence of the original March environment.
BLAS/platform differences and missing original dependencies preclude a claim of
bitwise identical statistical outputs.

Stata is required **only** to generate Gateway Harmonized KLoSA from the official
creation code. Supply a licensed installation and the original `bukpmcwp.do`.
If a licensed `GH_KLoSA_f.dta` is already available, K2/K3 can read it directly.
The Python stages use pandas' Stata reader; they do not require Stata.

## Configure paths

All executable Python files use shared path configuration. The default workspace
is `<checkout>/work`, which is ignored by git. Participant files and regenerated
results go there; released assets under `locked_result_assets/` remain references.

| Variable | Default | Purpose |
| --- | --- | --- |
| `ADL_WORK_ROOT` | `<checkout>/work` | Derived data, results, logs and local notes |
| `ADL_CHARLS_RAW_DIR` | `<work>/derived/raw_extracts` | CHARLS files and `orig_life_history/` |
| `ADL_EXPANDED_RERUN_ROOT` | `<work>/rerun_risk_stratification_expanded_v2_20260323` | Original expanded v2 inputs; producer absent |
| `ADL_KLOSA_DATA_ROOT` | `<work>/data/klosa` | Raw KLoSA archives and Gateway source |
| `ADL_KLOSA_RAW_DIR` | `<klosa_data>/KLoSA 1-9th wave (STATA)` | `wNN_e.dta` and/or `strNN_e.dta` files |
| `ADL_GATEWAY_DO` | `<klosa_data>/bukpmcwp.do` | Official Gateway creation program |
| `ADL_GATEWAY_BUILD_ROOT` | `<work>/gateway_build` | Stata staging, logs and harmonized output |
| `ADL_KLOSA_HARMONIZED_PATH` | `<work>/results/klosa_external_validation/gateway_harmonized_build/GH_KLoSA_f.dta` | Existing harmonized data read by K2/K3 |
| `STATA_BIN` | `PATH`, then known macOS applications | Stata executable path or command name |

Values expand `~`; relative values resolve against the launch directory.
Use absolute paths for consistent runs from different directories. Scripts read
variables at import/startup; restart Python after changing configuration.

```bash
cp reproducibility/config.env.example .env
# Edit .env and uncomment the variables you need.
source .env
python reproducibility/run.py --init-workspace
python reproducibility/run.py --list
python reproducibility/run.py --check stage2
```

The template uses `export`; `.env` is not loaded automatically. Workspace
initialization copies only the released index and limitations notes, only when
missing. It never creates analysis data, coefficients, or statistical results.

## Stata platform scope

The build still uses the original macOS/Linux command `stata-mp -b do <program>`
and symbolic links. Discovery supports `STATA_BIN` and `PATH` in addition to
macOS applications. Windows batch flags and symlink permissions are unvalidated.
No Stata build was run during this repair because the licensed raw data,
creation source and executable were unavailable.

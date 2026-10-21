"""Portable paths shared by the released scripts (no data are loaded here)."""
from __future__ import annotations

import importlib
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def configured_path(name: str, default: Path) -> Path:
    """Resolve an override, including ~; relative overrides use the launch cwd."""
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


def project_root() -> Path:
    """Keep regenerated and participant-level files outside tracked assets."""
    return configured_path("ADL_WORK_ROOT", REPO_ROOT / "work")


def charls_raw_dir() -> Path:
    return configured_path("ADL_CHARLS_RAW_DIR", project_root() / "derived" / "raw_extracts")


def expanded_rerun_root() -> Path:
    return configured_path(
        "ADL_EXPANDED_RERUN_ROOT",
        project_root() / "rerun_risk_stratification_expanded_v2_20260323",
    )


def klosa_data_root() -> Path:
    return configured_path("ADL_KLOSA_DATA_ROOT", project_root() / "data" / "klosa")


def klosa_raw_dir() -> Path:
    return configured_path("ADL_KLOSA_RAW_DIR", klosa_data_root() / "KLoSA 1-9th wave (STATA)")


def gateway_do_file() -> Path:
    return configured_path("ADL_GATEWAY_DO", klosa_data_root() / "bukpmcwp.do")


def harmonized_klosa_path() -> Path:
    return configured_path(
        "ADL_KLOSA_HARMONIZED_PATH",
        project_root() / "results" / "klosa_external_validation" / "gateway_harmonized_build" / "GH_KLoSA_f.dta",
    )


def find_stata() -> Path | None:
    """An explicit override must be valid; otherwise search PATH and macOS apps."""
    override = os.environ.get("STATA_BIN")
    if override:
        found = shutil.which(os.path.expanduser(override))
        if not found:
            raise FileNotFoundError(f"STATA_BIN is not an executable: {override}")
        return Path(found).resolve()
    for name in ("stata-mp", "stata-se", "stata", "StataMP", "StataSE"):
        found = shutil.which(name)
        if found:
            return Path(found).resolve()
    for candidate in (
        "/Applications/Stata/StataMP.app/Contents/MacOS/stata-mp",
        "/Applications/Stata/StataMP.app/Contents/MacOS/StataMP",
        "/Applications/Stata/StataSE.app/Contents/MacOS/stata-se",
        "/Applications/Stata/Stata.app/Contents/MacOS/Stata",
    ):
        if os.access(candidate, os.X_OK):
            return Path(candidate)
    return None


REINFORCEMENT_MODULE = "analytic_code.charls_main.run_submission_grade_reinforcement"
REINFORCEMENT_FUNCTIONS = (
    "cox_table", "draw_flowchart", "fit_cox", "make_baseline_table",
    "make_ipw_weights", "ph_diagnostics", "plot_schoenfeld", "spline_curve",
)


def require_reinforcement():
    """Require the original implementation; never replace statistical helpers."""
    try:
        module = importlib.import_module(REINFORCEMENT_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name != REINFORCEMENT_MODULE:
            raise  # A dependency *inside* a recovered module is a separate error.
        raise ModuleNotFoundError(
            "Original source is missing: analytic_code/charls_main/"
            "run_submission_grade_reinforcement.py. Restore the original module "
            "and its dependencies before running the legacy master. No statistical "
            "replacement is provided. See reproducibility/missing_source_audit.md.",
            name=REINFORCEMENT_MODULE,
        ) from exc
    missing = [name for name in REINFORCEMENT_FUNCTIONS if not callable(getattr(module, name, None))]
    if missing:
        raise ImportError(f"Recovered reinforcement module lacks callable functions: {', '.join(missing)}")
    return module

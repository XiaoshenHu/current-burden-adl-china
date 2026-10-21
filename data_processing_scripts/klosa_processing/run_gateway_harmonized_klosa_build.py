from __future__ import annotations

import io
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

# Resolve repository imports independently of the launch directory.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reproducibility.project_paths import configured_path, find_stata, gateway_do_file, klosa_data_root, klosa_raw_dir, project_root


PROJECT_ROOT = project_root()
DATA_ROOT = klosa_data_root()
RAW_DIR = klosa_raw_dir()
ORIGINAL_DO = gateway_do_file()
BUILD_ROOT = configured_path("ADL_GATEWAY_BUILD_ROOT", PROJECT_ROOT / "gateway_build")
INPUT_DIR = BUILD_ROOT / "input"
OUTPUT_DIR = BUILD_ROOT / "output"
LOG_DIR = BUILD_ROOT / "logs"
PARAM_DO = BUILD_ROOT / "bukpmcwp_build.do"
RUN_LOG = BUILD_ROOT / "build_stdout_stderr.log"
OUTPUT_DATA = OUTPUT_DIR / "GH_KLoSA_f.dta"
PROJECT_RESULT_DIR = PROJECT_ROOT / "results" / "klosa_external_validation" / "gateway_harmonized_build"


def detect_stata_binary() -> Path:
    binary = find_stata()
    if binary is None:
        raise FileNotFoundError("No Stata executable found. Set STATA_BIN or add Stata to PATH.")
    return binary


def ensure_clean_dirs() -> None:
    for path in [BUILD_ROOT, INPUT_DIR, OUTPUT_DIR, LOG_DIR, PROJECT_RESULT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def stage_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        dst.unlink()
    os.symlink(src, dst)


def stage_main_wave_files() -> None:
    required = [f"w{i:02d}_e.dta" for i in range(1, 10)] + ["w05_new_e.dta"]
    for name in required:
        src = RAW_DIR / name
        if not src.exists():
            raise FileNotFoundError(f"Missing required raw file: {src}")
        stage_symlink(src, INPUT_DIR / name)


def stage_extracted_exit_files() -> None:
    for name in ["w02_exit_e.dta", "w03_exit_e.dta", "w04_exit_e.dta", "w05_exit_e.dta", "w06_Exit_e.dta"]:
        src = DATA_ROOT / name
        if not src.exists():
            raise FileNotFoundError(f"Missing extracted exit file: {src}")
        stage_symlink(src, INPUT_DIR / name)


def extract_single_member(zip_path: Path, member_name: str, out_dir: Path) -> None:
    out_path = out_dir / member_name
    if out_path.exists():
        return
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member_name) as src, open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst)


def stage_zipped_exit_files() -> None:
    zip_map = {
        DATA_ROOT / "2018 KLoSA 7 wave EXIT.zip": "w07_exit_e.dta",
        DATA_ROOT / "KLoSA 8th wave_EXIT.zip": "w08_exit_e.dta",
        DATA_ROOT / "KLoSA 9 wave Exit.zip": "Exit09_e.dta",
    }
    for zip_path, member_name in zip_map.items():
        if not zip_path.exists():
            raise FileNotFoundError(f"Missing exit zip: {zip_path}")
        extract_single_member(zip_path, member_name, INPUT_DIR)


def stage_imputation_files() -> None:
    outer_zip = DATA_ROOT / "KLoSA_Imputation_E_v2_20210324.zip"
    if not outer_zip.exists():
        alt = DATA_ROOT / "KLoSA_Imputation_E_v2_20210324 (1).zip"
        outer_zip = alt if alt.exists() else outer_zip
    if not outer_zip.exists():
        raise FileNotFoundError("No KLoSA imputation zip was found.")

    wanted = [f"e_imp_w{i:02d}.dta" for i in range(1, 8)] + ["e_imp_w05_new.dta"]
    with zipfile.ZipFile(outer_zip) as outer:
        inner_name = [n for n in outer.namelist() if "STATA" in n.upper()][0]
        inner_bytes = outer.read(inner_name)
        with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
            names = set(inner.namelist())
            for member in wanted:
                if member not in names:
                    raise FileNotFoundError(f"{member} not found in nested STATA imputation zip.")
                out_path = INPUT_DIR / member
                if out_path.exists():
                    continue
                with inner.open(member) as src, open(out_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)


def write_parameterized_do_file() -> None:
    text = ORIGINAL_DO.read_text(encoding="utf-8", errors="ignore")
    text = text.replace("||KLoSA_data_folder||", INPUT_DIR.as_posix())
    text = text.replace("||save to folder||", OUTPUT_DIR.as_posix())
    PARAM_DO.write_text(text, encoding="utf-8")


def run_stata_build() -> None:
    stata_bin = detect_stata_binary()
    cmd = [str(stata_bin), "-b", "do", str(PARAM_DO)]
    result = subprocess.run(
        cmd,
        cwd=BUILD_ROOT,
        capture_output=True,
        text=True,
    )
    RUN_LOG.write_text(
        f"COMMAND: {' '.join(cmd)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n",
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Stata build failed with exit code {result.returncode}. See {RUN_LOG}"
        )


def copy_project_outputs() -> None:
    if not OUTPUT_DATA.exists():
        raise FileNotFoundError(f"Expected harmonized output not found: {OUTPUT_DATA}")

    for path in [OUTPUT_DATA, PARAM_DO, BUILD_ROOT / "bukpmcwp_build.log", RUN_LOG]:
        if path.exists():
            shutil.copy2(path, PROJECT_RESULT_DIR / path.name)


def main() -> None:
    ensure_clean_dirs()
    stage_main_wave_files()
    stage_extracted_exit_files()
    stage_zipped_exit_files()
    stage_imputation_files()
    write_parameterized_do_file()
    run_stata_build()
    copy_project_outputs()
    print("Build root:", BUILD_ROOT)
    print("Input dir:", INPUT_DIR)
    print("Output dataset:", OUTPUT_DATA)
    print("Project results mirror:", PROJECT_RESULT_DIR)


if __name__ == "__main__":
    main()

"""Inspect stage contracts and run one stage without changing its analysis."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.project_paths import (
    REPO_ROOT, charls_raw_dir, configured_path,
    expanded_rerun_root, find_stata, gateway_do_file, harmonized_klosa_path,
    klosa_data_root, klosa_raw_dir, project_root,
)


def load_stages() -> dict:
    return json.loads((REPO_ROOT / "reproducibility" / "pipeline.json").read_text())["stages"]


def resolve_path(token: str) -> Path:
    prefix, _, relative = token.partition(":")
    roots = {
        "work": project_root(), "repo": REPO_ROOT, "charls_raw": charls_raw_dir(),
        "expanded": expanded_rerun_root(), "klosa_data": klosa_data_root(),
        "klosa_raw": klosa_raw_dir(), "harmonized": harmonized_klosa_path(),
        "gateway_do": gateway_do_file(),
        "gateway_build": configured_path("ADL_GATEWAY_BUILD_ROOT", project_root() / "gateway_build"),
    }
    if prefix not in roots:
        raise ValueError(f"Unknown path prefix: {prefix}")
    return roots[prefix] / relative if relative else roots[prefix]


def initialize_workspace() -> None:
    root = project_root()
    root.mkdir(parents=True, exist_ok=True)
    for name in ("RESULTS_INDEX.md", "LIMITATIONS_OF_REANALYSIS.md"):
        dest = root / name
        if not dest.exists():
            shutil.copy2(REPO_ROOT / "locked_result_assets" / "context" / name, dest)
    print(f"Workspace initialized: {root}")


def check_stage(stage: dict) -> list[str]:
    problems = []
    for token in stage.get("inputs", []):
        if not resolve_path(token).is_file():
            problems.append(f"Missing file: {resolve_path(token)}")
    for alternatives in stage.get("input_alternatives", []):
        if not any(resolve_path(token).is_file() for token in alternatives):
            problems.append("Need one of: " + ", ".join(str(resolve_path(token)) for token in alternatives))
    for module in stage.get("source_modules", []):
        if importlib.util.find_spec(module) is None:
            problems.append(f"Missing original source: {module}")
    if stage.get("requires_stata"):
        try:
            if find_stata() is None:
                problems.append("Stata unavailable: set STATA_BIN or add Stata to PATH")
        except FileNotFoundError as exc:
            problems.append(str(exc))
    if stage.get("requires_gateway_placeholders") and gateway_do_file().is_file():
        source = gateway_do_file().read_text(encoding="utf-8", errors="ignore")
        for marker in ("||KLoSA_data_folder||", "||save to folder||"):
            if marker not in source:
                problems.append(f"Gateway source lacks expected placeholder {marker}; verify the source release")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true")
    group.add_argument("--init-workspace", action="store_true")
    group.add_argument("--describe", metavar="STAGE", help="print exact command and resolved input/output paths")
    group.add_argument("--check", metavar="STAGE", help="stage id or all; file/source availability only")
    group.add_argument("--run", metavar="STAGE", help="run exactly one stage after its preflight passes")
    args = parser.parse_args(argv)
    stages = load_stages()
    if args.init_workspace:
        initialize_workspace()
        return 0
    if args.list:
        for name, stage in stages.items():
            print(f"{name:25} {stage['description']}")
        return 0
    key = args.check or args.run or args.describe
    if key != "all" and key not in stages:
        parser.error(f"Unknown stage {key!r}; use --list")
    if args.run and key == "all":
        parser.error("Run one explicit stage at a time; --check all is available")
    if args.describe:
        for name in stages if key == "all" else [key]:
            stage = dict(stages[name])
            stage["id"] = name
            stage["command"] = [sys.executable, str(REPO_ROOT / stage["script"])]
            for field in ("inputs", "output_files", "additional_output_directories"):
                stage[field] = [str(resolve_path(token)) for token in stage.get(field, [])]
            stage["output_directory"] = str(resolve_path(stage["output_directory"]))
            stage["input_alternatives"] = [[str(resolve_path(token)) for token in group]
                                           for group in stage.get("input_alternatives", [])]
            print(json.dumps(stage, indent=2))
        return 0
    blocked = False
    for name in stages if key == "all" else [key]:
        stage = stages[name]
        problems = check_stage(stage)
        print(f"{name}: {'BLOCKED' if problems else 'INPUTS PRESENT (content not validated)'}", flush=True)
        for problem in problems:
            print(f"  {problem}", flush=True)
        if stage.get("note"):
            print(f"  Note: {stage['note']}", flush=True)
        if problems:
            blocked = True
            continue
        if args.run:
            env = dict(os.environ)
            env.setdefault("MPLBACKEND", "Agg")
            command = [sys.executable, str(REPO_ROOT / stage["script"])]
            print("Command: " + " ".join(command), flush=True)
            return subprocess.run(command, cwd=REPO_ROOT, env=env).returncode
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())

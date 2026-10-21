"""Data-free checks of portability, source integrity and immutable assets."""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reproducibility import project_paths as paths
from reproducibility.collect_io import inventory_text
from reproducibility.run import check_stage, initialize_workspace, load_stages, main, resolve_path

REPO = paths.REPO_ROOT
BASE = "73ba8f9caaaa9abe78904954fde031847ce551a2"
SCRIPTS = sorted([*(REPO / "analytic_code").rglob("*.py"), *(REPO / "data_processing_scripts").rglob("*.py")])


class PortabilityTests(unittest.TestCase):
    def setUp(self):
        # A user's real-data configuration must never affect data-free fixtures.
        reset = {key: "" for key in os.environ if key.startswith("ADL_") or key == "STATA_BIN"}
        self.environment = patch.dict(os.environ, reset)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_all_release_modules_import_without_data(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp) / "no data here"
            code = "import runpy\nfor p in " + repr([str(p) for p in SCRIPTS]) + ":\n runpy.run_path(p, run_name='portability_import')\n"
            env = dict(os.environ, ADL_WORK_ROOT=str(work), MPLBACKEND="Agg")
            result = subprocess.run([sys.executable, "-c", code], cwd=temp, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(work.exists(), "Imports must not create a data workspace")

    def test_configured_paths_accept_spaces_and_independent_roots(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
            "ADL_WORK_ROOT": str(Path(temp) / "work root"),
            "ADL_CHARLS_RAW_DIR": str(Path(temp) / "CHARLS input"),
            "ADL_KLOSA_RAW_DIR": str(Path(temp) / "KLoSA input"),
            "ADL_EXPANDED_RERUN_ROOT": str(Path(temp) / "locked v2"),
        }):
            self.assertEqual(paths.project_root(), Path(temp) / "work root")
            self.assertEqual(resolve_path("charls_raw:H_CHARLS_D_Data.dta"), Path(temp) / "CHARLS input" / "H_CHARLS_D_Data.dta")
            self.assertEqual(resolve_path("klosa_raw:str01_e.dta"), Path(temp) / "KLoSA input" / "str01_e.dta")
            self.assertEqual(resolve_path("expanded:02_data_processing/analysis_dataset_landmark.csv"), Path(temp) / "locked v2" / "02_data_processing" / "analysis_dataset_landmark.csv")

    def test_workspace_init_preserves_existing_notes(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"ADL_WORK_ROOT": temp}):
            target = Path(temp) / "RESULTS_INDEX.md"
            target.write_text("local notes\n")
            initialize_workspace()
            self.assertEqual(target.read_text(), "local notes\n")
            self.assertTrue((Path(temp) / "LIMITATIONS_OF_REANALYSIS.md").is_file())

    @unittest.skipIf(os.name == "nt", "Executable fixture uses POSIX permissions")
    def test_stata_override_is_used_and_invalid_override_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            executable = Path(temp) / "Stata MP"
            executable.write_text("#!/bin/sh\nexit 0\n")
            executable.chmod(0o755)
            with patch.dict(os.environ, {"STATA_BIN": str(executable)}):
                self.assertEqual(paths.find_stata(), executable.resolve())
            with patch.dict(os.environ, {"STATA_BIN": str(Path(temp) / "missing")}):
                with self.assertRaisesRegex(FileNotFoundError, "STATA_BIN"):
                    paths.find_stata()

    def test_missing_original_source_blocks_master_before_output(self):
        master = importlib.import_module("analytic_code.charls_main.run_landmark_primary_rerun_master")
        error = ModuleNotFoundError("missing", name=paths.REINFORCEMENT_MODULE)
        with patch.object(paths.importlib, "import_module", side_effect=error):
            with self.assertRaisesRegex(ModuleNotFoundError, "Restore the original module"):
                master.main()

    def test_missing_recovered_modules_dependency_is_not_mislabelled(self):
        error = ModuleNotFoundError("missing transitive dependency", name="private_dependency")
        with patch.object(paths.importlib, "import_module", side_effect=error):
            with self.assertRaises(ModuleNotFoundError) as caught:
                paths.require_reinforcement()
        self.assertIs(caught.exception, error)

    def test_preflight_refuses_missing_inputs_and_does_not_run(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"ADL_WORK_ROOT": temp}):
            problems = check_stage(load_stages()["stage2"])
            self.assertEqual(len(problems), 2)
            with patch("reproducibility.run.subprocess.run") as run:
                self.assertEqual(main(["--run", "stage2"]), 1)
                run.assert_not_called()

    def test_describe_resolves_paths_without_creating_outputs(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"ADL_WORK_ROOT": str(Path(temp) / "new work")}):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["--describe", "stage2"]), 0)
            contract = json.loads(output.getvalue())
            self.assertEqual(contract["inputs"][0], str(paths.expanded_rerun_root() / "02_data_processing" / "analysis_dataset_landmark.csv"))
            self.assertEqual(contract["command"][0], sys.executable)
            self.assertFalse(paths.project_root().exists())

    def test_inventory_requires_all_raw_wave_groups_before_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "empty workspace"
            env = dict(os.environ, ADL_WORK_ROOT=str(root), ADL_KLOSA_DATA_ROOT=str(root / "absent data"))
            env.pop("STATA_BIN", None)
            result = subprocess.run([sys.executable, str(REPO / "reproducibility" / "run.py"), "--run", "klosa-audit"],
                                    cwd=temp, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("w01_e.dta", result.stdout)
            self.assertIn("Lt09_e.dta", result.stdout)
            self.assertEqual(result.stdout.count("Missing file:"), 28)
            self.assertFalse(root.exists(), "Preflight must stop before original data readers run")

    def test_build_contract_checks_alternatives_stata_and_do_placeholders(self):
        stage = load_stages()["klosa-build"]
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
            "ADL_WORK_ROOT": temp, "ADL_KLOSA_DATA_ROOT": temp,
            "ADL_KLOSA_RAW_DIR": str(Path(temp) / "raw"),
            "ADL_GATEWAY_DO": str(Path(temp) / "source.do"),
        }):
            for token in stage["inputs"]:
                target = resolve_path(token)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("")
            resolve_path(stage["input_alternatives"][0][1]).write_text("")
            with patch("reproducibility.run.find_stata", return_value=Path(temp) / "stata"):
                self.assertEqual(len(check_stage(stage)), 2)
                paths.gateway_do_file().write_text("||KLoSA_data_folder||\n||save to folder||")
                self.assertEqual(check_stage(stage), [])

    def test_stage_contracts_cover_every_entrypoint(self):
        stages = load_stages()
        mapped = {x["script"] for x in stages.values()}
        mapped.add("analytic_code/klosa_external_applicability/run_gateway_harmonized_klosa_build.py")
        for script in SCRIPTS:
            if "def main(" in script.read_text():
                self.assertIn(script.relative_to(REPO).as_posix(), mapped)
        for stage in stages.values():
            self.assertTrue((REPO / stage["script"]).is_file())
            self.assertTrue(set(stage["depends_on"]) <= stages.keys())
            for token in stage["inputs"] + [stage["output_directory"]] + stage["output_files"]:
                resolve_path(token)

    def test_locked_assets_match_original_release(self):
        manifest = json.loads((REPO / "reproducibility" / "locked_assets_sha256.json").read_text())
        actual = {p.relative_to(REPO).as_posix() for p in (REPO / "locked_result_assets").rglob("*") if p.is_file()}
        self.assertEqual(actual, set(manifest["files"]))
        for name, digest in manifest["files"].items():
            self.assertEqual(hashlib.sha256((REPO / name).read_bytes()).hexdigest(), digest, name)

    def test_io_inventory_is_current(self):
        self.assertEqual((REPO / "reproducibility" / "io_inventory.md").read_text(), inventory_text())

    def test_no_developer_home_paths_remain_in_executable_code(self):
        for p in SCRIPTS:
            for node in ast.walk(ast.parse(p.read_text())):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    self.assertNotIn("/Users/", node.value, str(p))

    def test_statistical_functions_and_constants_unchanged_from_base(self):
        # Full git history is available in CI. A source-only archive can still run
        # every other check; it cannot establish the original AST provenance.
        check = subprocess.run(["git", "cat-file", "-e", BASE], cwd=REPO, capture_output=True)
        if check.returncode:
            self.skipTest("Original commit unavailable; fetch history for AST provenance check")
        path_names = {"ROOT", "PROJECT_ROOT", "DATA_ROOT", "RAW_DIR", "RAW", "RAW_D", "RERUN_ROOT", "PHENO_DIR", "ORIGINAL_PERF", "HARMONIZED_PATH", "ORIGINAL_DO", "HARMONIZED_DO_FILE", "BUILD_ROOT"}
        for p in SCRIPTS:
            rel = p.relative_to(REPO).as_posix()
            old = ast.parse(subprocess.check_output(["git", "show", f"{BASE}:{rel}"], cwd=REPO))
            new = ast.parse(p.read_text())
            old_funcs = {n.name: n for n in old.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            new_funcs = {n.name: n for n in new.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            self.assertEqual(old_funcs.keys(), new_funcs.keys(), rel)
            for name, before in old_funcs.items():
                after = new_funcs[name]
                if name == "detect_stata_binary":
                    continue  # Platform discovery, not a statistical function.
                if p.name == "run_landmark_primary_rerun_master.py":
                    if name == "main": after.body = after.body[3:]
                    if name == "fit_full_followup_cox_supplements": after.body = after.body[7:]
                if p.name in ("prepare_life_course_aging_dataset.py", "build_charls_variable_inventory.py") and name == "main":
                    after.body = after.body[1:]
                self.assertEqual(ast.dump(before), ast.dump(after), f"Statistical body changed: {rel}:{name}")
            def assignments(tree):
                return {n.targets[0].id: ast.dump(n.value) for n in tree.body
                        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id not in path_names}
            self.assertEqual(assignments(old), assignments(new), f"Analysis constants changed: {rel}")


if __name__ == "__main__":
    unittest.main()

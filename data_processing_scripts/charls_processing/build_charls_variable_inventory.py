#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

# Resolve repository imports independently of the launch directory.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reproducibility.project_paths import charls_raw_dir, project_root

import pandas as pd


ROOT = project_root()
RAW = charls_raw_dir()
OUT = ROOT / "derived"

FILES = {
    "D": RAW / "H_CHARLS_D_Data.dta",
    "LH": RAW / "H_CHARLS_LH_a.dta",
    "EOL": RAW / "H_CHARLS_EOL_a.dta",
}

PATTERNS = {
    "ids_and_wave": ["id", "hhid", "community", "inw", "iwstat", "wave"],
    "childhood_adversity": [
        "child", "fam", "faminc", "hunger", "food", "school", "edu",
        "mother", "father", "abuse", "violence", "health", "hospital",
        "bed", "rural", "toilet", "water", "c_wrkstate", "c_", "lh"
    ],
    "cognition": ["cogn", "memory", "orient", "serial", "word", "draw", "mental", "tics"],
    "depression": ["depr", "cesd", "sad", "lonel"],
    "function_disability": ["adl", "iadl", "walk", "chair", "climb", "stoop", "lift", "grip"],
    "chronic_disease": ["diabe", "hyper", "heart", "stroke", "lunge", "arthr", "cancer", "kidney"],
    "behavior_ses": ["work", "retire", "smok", "drink", "insur", "pension", "income", "asset", "wealth"],
    "anthro_biomarker": ["bmi", "waist", "grip", "crp", "hdl", "ldl", "chol", "trig", "hba1c", "glucose"],
    "sleep_pain_sensory": ["sleep", "pain", "hear", "vision", "fall"],
    "death": ["death", "dead", "proxy", "cod", "radage", "rax", "eol", "alive"],
}


def read_columns(path: Path) -> list[str]:
    reader = pd.read_stata(path, iterator=True)
    sample = reader.read(5)
    return list(sample.columns)


def search_columns(columns: list[str], patterns: list[str]) -> list[str]:
    hits: list[str] = []
    seen = set()
    for col in columns:
        cl = col.lower()
        if any(p in cl for p in patterns):
            if col not in seen:
                hits.append(col)
                seen.add(col)
    return hits


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inventory: dict[str, dict[str, list[str]]] = {}
    for tag, path in FILES.items():
        cols = read_columns(path)
        inventory[tag] = {
            "all_columns": cols,
            "groups": {group: search_columns(cols, pats) for group, pats in PATTERNS.items()},
        }

    (OUT / "charls_variable_inventory.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False)
    )

    lines = ["# CHARLS Variable Inventory", ""]
    for tag, payload in inventory.items():
        lines.append(f"## {tag}")
        lines.append(f"- Total columns: {len(payload['all_columns'])}")
        for group, cols in payload["groups"].items():
            lines.append(f"### {group}")
            lines.append(f"- Matches: {len(cols)}")
            if cols:
                preview = ", ".join(cols[:40])
                lines.append(f"- Preview: {preview}")
            else:
                lines.append("- Preview: none")
        lines.append("")

    (OUT / "charls_variable_inventory.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()

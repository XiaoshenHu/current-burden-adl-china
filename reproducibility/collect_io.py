"""Render explicit file IO expressions without running any analysis."""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
READS = {"read_csv", "read_stata", "read_excel", "read_text", "load_stata", "load_workbook"}
WRITES = {"to_csv", "to_stata", "to_excel", "write_text", "savefig", "ExcelWriter", "PdfPages", "copy2"}


def inventory_text() -> str:
    lines = [
        "# Source IO inventory", "",
        "Generated with `python reproducibility/collect_io.py`. This is a static inventory of explicit IO calls,",
        "including helper functions. It does not execute code, prove branch coverage, or imply that inputs exist.",
        "`pipeline.json` specifies required stage inputs, the producer relationships, and resolved static output names.",
        "Local variables such as `outpath`, `writer`, and `path` are retained verbatim; the source line links them to their caller.",
        "The legacy master also declares its complete table map in `main()` (`save_map`) and its file registry.", "",
    ]
    for root in ("data_processing_scripts", "analytic_code"):
        for path in sorted((REPO / root).rglob("*.py")):
            source = path.read_text()
            tree = ast.parse(source)
            calls = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
                if name in READS | WRITES:
                    call = ast.unparse(node).replace("|", "&#124;").replace("\n", " ").replace("`", "&#96;")
                    calls.append((node.lineno, "read" if name in READS else "write", call))
            if not calls:
                continue
            relative = path.relative_to(REPO).as_posix()
            lines += [f"## `{relative}`", "", "| Line | Operation | Expression |", "| --- | --- | --- |"]
            lines.extend(f"| {line} | {kind} | `{expr}` |" for line, kind, expr in sorted(calls))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    target = REPO / "reproducibility" / "io_inventory.md"
    target.write_text(inventory_text(), encoding="utf-8")
    print(target)

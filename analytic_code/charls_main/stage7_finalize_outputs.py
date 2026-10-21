#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

# Resolve repository imports independently of the launch directory.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from analytic_code.model_validation_and_representation.stage7_cluster_robustness import (
    RESULTS_DIR,
    SEED,
    build_plots,
    update_results_index,
    write_report,
)


def main() -> None:
    spec_df = pd.read_csv(RESULTS_DIR / "gmm_specification_comparison.csv")
    stability_df = pd.read_csv(RESULTS_DIR / "cluster_stability_metrics.csv")
    posterior_df = pd.read_csv(RESULTS_DIR / "posterior_distribution_summary.csv")

    build_plots(spec_df, stability_df, posterior_df)
    write_report(spec_df, stability_df, posterior_df)
    update_results_index()

    meta = {
        "stage": "stage7_cluster_robustness",
        "seed": SEED,
        "n_spec_rows": int(len(spec_df)),
        "n_subsample_iterations": int(len(stability_df)),
        "spec_grid_note": "all covariance structures at n_init 20 and 50; diag additionally stress-tested at n_init 100",
        "finalization_mode": "reused_existing_csv_outputs",
    }
    (RESULTS_DIR / "stage7_run_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()

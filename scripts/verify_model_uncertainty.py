"""Verify paired bootstrap and color-normalization sensitivity outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("outputs/modeling/uncertainty")


def main() -> None:
    errors: list[str] = []
    bootstrap_path = ROOT / "paired_mae_bootstrap.csv"
    sensitivity_path = ROOT / "color_normalization_sensitivity.csv"
    figure_path = ROOT / "figures" / "paired_mae_bootstrap.png"
    for path in [bootstrap_path, sensitivity_path, figure_path]:
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Missing or empty output: {path}")
    if errors:
        print(json.dumps({"errors": errors}, indent=2))
        raise SystemExit(1)

    bootstrap = pd.read_csv(bootstrap_path)
    sensitivity = pd.read_csv(sensitivity_path)
    if len(bootstrap) != 4:
        errors.append(f"Expected 4 paired comparisons, found {len(bootstrap)}")
    if len(sensitivity) != 8:
        errors.append(f"Expected 8 sensitivity rows, found {len(sensitivity)}")
    if set(bootstrap["clusters"]) != {96}:
        errors.append("Bootstrap cluster count is not 96")
    if set(bootstrap["bootstrap_repeats"]) != {20_000}:
        errors.append("Bootstrap repeat count is not 20,000")
    if not (
        bootstrap["ci95_lower"]
        <= bootstrap["mae_difference_cnn_minus_ml"]
    ).all() or not (
        bootstrap["mae_difference_cnn_minus_ml"] <= bootstrap["ci95_upper"]
    ).all():
        errors.append("A paired point estimate falls outside its bootstrap interval")
    if not (bootstrap["ci95_lower"] > 0).all():
        errors.append("At least one paired bootstrap interval includes zero")
    numeric = sensitivity[
        ["mae_mean", "mae_std", "rmse_mean", "r2_mean", "mae_change_vs_rgb"]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        errors.append("Sensitivity output contains non-finite values")
    if set(sensitivity["folds"]) != {5}:
        errors.append("Sensitivity summary does not contain five folds per row")

    report = {
        "paired_comparison_rows": len(bootstrap),
        "normalization_sensitivity_rows": len(sensitivity),
        "all_ci95_lower_above_zero": bool((bootstrap["ci95_lower"] > 0).all()),
        "minimum_ci95_lower": float(bootstrap["ci95_lower"].min()),
        "errors": errors,
    }
    (ROOT / "verification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

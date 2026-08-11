"""Verify Extra Trees and Random Forest feature-reduction efficiency outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "modeling" / "feature_efficiency"
ALGORITHM_SUMMARY = (
    ROOT
    / "outputs"
    / "modeling"
    / "reduced_features"
    / "algorithm_feature_summary.csv"
)
MODELS = {
    "extra_trees": "ExtraTrees",
    "random_forest": "RandomForest",
}
FEATURE_COUNTS = {
    "G_only": 1,
    "RGB_primary": 3,
    "Hue_only": 2,
    "HSV_primary": 4,
}


def check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def main() -> None:
    errors: list[str] = []
    performance = pd.read_csv(ALGORITHM_SUMMARY)
    report: dict[str, object] = {}
    for prefix, model_name in MODELS.items():
        fold_path = OUTPUT_DIR / f"{prefix}_efficiency_folds.csv"
        summary_path = OUTPUT_DIR / f"{prefix}_efficiency_summary.csv"
        figure_path = (
            OUTPUT_DIR / "figures" / f"{prefix}_feature_reduction_efficiency.png"
        )
        check(fold_path.is_file(), f"Missing {fold_path.name}", errors)
        check(summary_path.is_file(), f"Missing {summary_path.name}", errors)
        check(
            figure_path.is_file() and figure_path.stat().st_size > 0,
            f"Missing or empty {figure_path.name}",
            errors,
        )
        if not fold_path.is_file() or not summary_path.is_file():
            continue
        folds = pd.read_csv(fold_path)
        summary = pd.read_csv(summary_path)
        check(len(folds) == 40, f"{model_name}: expected 40 fold rows", errors)
        check(len(summary) == 8, f"{model_name}: expected 8 summary rows", errors)
        check(
            folds.groupby(["analyte", "feature_set"])["outer_fold"].nunique().eq(5).all(),
            f"{model_name}: incomplete outer folds",
            errors,
        )
        expected_counts = folds["feature_set"].map(FEATURE_COUNTS)
        check(
            np.array_equal(
                folds["numeric_feature_count"].to_numpy(dtype=int),
                expected_counts.to_numpy(dtype=int),
            ),
            f"{model_name}: numeric feature counts differ from definitions",
            errors,
        )
        maximum_difference = float(folds["prediction_max_abs_difference"].max())
        check(
            maximum_difference <= 1e-10,
            f"{model_name}: refit prediction mismatch {maximum_difference}",
            errors,
        )
        check(
            (folds["model_size_bytes"] > 0).all(),
            f"{model_name}: non-positive model size",
            errors,
        )
        check(
            (folds["total_tree_nodes"] > 0).all(),
            f"{model_name}: non-positive tree-node count",
            errors,
        )
        expected_performance = performance.loc[performance["model"] == model_name]
        merged = summary.merge(
            expected_performance[["analyte", "feature_set", "mae_mean", "r2_mean"]],
            on=["analyte", "feature_set"],
            suffixes=("", "_expected"),
            validate="one_to_one",
        )
        check(
            np.allclose(merged["mae_mean"], merged["mae_mean_expected"], atol=1e-12),
            f"{model_name}: MAE summary mismatch",
            errors,
        )
        check(
            np.allclose(merged["r2_mean"], merged["r2_mean_expected"], atol=1e-12),
            f"{model_name}: R2 summary mismatch",
            errors,
        )
        report[model_name] = {
            "fold_rows": len(folds),
            "summary_rows": len(summary),
            "maximum_prediction_difference": maximum_difference,
            "figure_bytes": figure_path.stat().st_size,
        }
    report["errors"] = errors
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

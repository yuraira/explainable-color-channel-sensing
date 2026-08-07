"""Verify RGB, G-only, Hue-only, and HSV model comparison outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("outputs/modeling/reduced_features")


def main() -> None:
    errors: list[str] = []
    required = [
        ROOT / "reduced_feature_fold_metrics.csv",
        ROOT / "reduced_feature_best_parameters.csv",
        ROOT / "reduced_feature_predictions.csv",
        ROOT / "algorithm_feature_summary.csv",
        ROOT / "nested_selected_feature_family_folds.csv",
        ROOT / "nested_selected_feature_family_summary.csv",
        ROOT / "figures" / "single_vs_full_color_features.png",
        ROOT / "figures" / "random_forest_single_vs_full_color_features.png",
        ROOT / "run_config.json",
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Missing or empty output: {path}")
    if errors:
        print(json.dumps({"errors": errors}, indent=2))
        raise SystemExit(1)

    metrics = pd.read_csv(ROOT / "reduced_feature_fold_metrics.csv")
    parameters = pd.read_csv(ROOT / "reduced_feature_best_parameters.csv")
    predictions = pd.read_csv(ROOT / "reduced_feature_predictions.csv")
    algorithms = pd.read_csv(ROOT / "algorithm_feature_summary.csv")
    selected = pd.read_csv(ROOT / "nested_selected_feature_family_folds.csv")
    summary = pd.read_csv(ROOT / "nested_selected_feature_family_summary.csv")
    config = json.loads((ROOT / "run_config.json").read_text(encoding="utf-8"))

    expected_counts = {
        "reduced_metric_rows": 100,
        "parameter_rows": 100,
        "prediction_rows": 18_240,
        "algorithm_summary_rows": 40,
        "selected_fold_rows": 40,
        "feature_summary_rows": 8,
    }
    actual_counts = {
        "reduced_metric_rows": len(metrics),
        "parameter_rows": len(parameters),
        "prediction_rows": len(predictions),
        "algorithm_summary_rows": len(algorithms),
        "selected_fold_rows": len(selected),
        "feature_summary_rows": len(summary),
    }
    for key, expected in expected_counts.items():
        if actual_counts[key] != expected:
            errors.append(f"{key}: expected {expected}, found {actual_counts[key]}")

    if metrics.duplicated(
        ["analyte", "outer_fold", "feature_set", "model"]
    ).any():
        errors.append("Duplicate reduced-feature fold metric rows")
    if selected.duplicated(["analyte", "outer_fold", "feature_set"]).any():
        errors.append("Duplicate nested-selected feature-family rows")
    if set(metrics["feature_set"]) != {"G_only", "Hue_only"}:
        errors.append("Unexpected reduced feature sets")
    if set(summary["feature_set"]) != {
        "G_only",
        "Hue_only",
        "RGB_primary",
        "HSV_primary",
    }:
        errors.append("Feature-family summary is incomplete")
    fold_counts = selected.groupby(["analyte", "feature_set"])["outer_fold"].nunique()
    if set(fold_counts) != {5}:
        errors.append("A feature family does not contain all five outer folds")
    numeric = metrics[["inner_best_mae", "mae", "rmse", "r2"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        errors.append("Reduced-feature metrics contain non-finite values")
    if config.get("G_only") != ["g_median"]:
        errors.append("G-only feature definition is incorrect")
    if config.get("Hue_only") != ["h_sin_weighted", "h_cos_weighted"]:
        errors.append("Hue-only circular feature definition is incorrect")
    random_forest = algorithms.loc[algorithms["model"] == "RandomForest"]
    if len(random_forest) != 8:
        errors.append("Random Forest comparison does not contain all feature families")
    expected_random_forest_mae = {
        ("glucose", "G_only"): 2.4709554624409344,
        ("glucose", "RGB_primary"): 0.6308399249128323,
        ("glucose", "Hue_only"): 0.5327391578641573,
        ("glucose", "HSV_primary"): 0.3432931770427415,
        ("ketone", "G_only"): 0.6226133985743555,
        ("ketone", "RGB_primary"): 0.1375255843536541,
        ("ketone", "Hue_only"): 0.0438323421052631,
        ("ketone", "HSV_primary"): 0.0403828552631578,
    }
    for (analyte, feature_set), expected in expected_random_forest_mae.items():
        row = random_forest.loc[
            (random_forest["analyte"] == analyte)
            & (random_forest["feature_set"] == feature_set)
        ]
        if len(row) != 1 or not np.isclose(float(row.iloc[0]["mae_mean"]), expected):
            errors.append(
                f"Unexpected Random Forest MAE for {analyte} {feature_set}"
            )

    result = {
        **actual_counts,
        "feature_families": sorted(summary["feature_set"].unique().tolist()),
        "random_forest_results": random_forest[
            ["analyte", "feature_set", "mae_mean", "r2_mean"]
        ].to_dict(orient="records"),
        "nested_selected_results": summary[
            ["analyte", "feature_set", "mae_mean", "r2_mean"]
        ].to_dict(orient="records"),
        "errors": errors,
    }
    (ROOT / "verification_report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

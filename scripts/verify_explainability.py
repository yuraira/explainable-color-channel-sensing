"""Verify explainability outputs, fold coverage, and numerical consistency."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("outputs/explainability")
ML = ROOT / "ml"
CNN = ROOT / "cnn"


def main() -> None:
    errors: list[str] = []
    warnings: list[str] = []

    required = [
        ML / "ml_explainability_fold_results.csv",
        ML / "ml_grouped_shap_values.csv",
        ML / "ml_explainability_model_performance.csv",
        ML / "ml_explainability_summary.csv",
        ML / "figures" / "color_channel_importance.png",
        ML / "figures" / "shap_direction_summary.png",
        CNN / "cnn_gradcam_patch_metrics.csv",
        CNN / "cnn_gradcam_fold_summary.csv",
        CNN / "cnn_gradcam_summary.csv",
        CNN / "figures" / "cnn_region_dependence.png",
        CNN / "figures" / "cnn_gradcam_examples.png",
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Missing or empty output: {path}")

    if errors:
        report = {"errors": errors, "warnings": warnings}
        print(json.dumps(report, indent=2))
        raise SystemExit(1)

    ml_folds = pd.read_csv(ML / "ml_explainability_fold_results.csv")
    ml_values = pd.read_csv(ML / "ml_grouped_shap_values.csv")
    ml_performance = pd.read_csv(ML / "ml_explainability_model_performance.csv")
    ml_summary = pd.read_csv(ML / "ml_explainability_summary.csv")
    cnn_metrics = pd.read_csv(CNN / "cnn_gradcam_patch_metrics.csv")
    cnn_folds = pd.read_csv(CNN / "cnn_gradcam_fold_summary.csv")
    cnn_summary = pd.read_csv(CNN / "cnn_gradcam_summary.csv")

    expected_counts = {
        "ml_fold_rows": 60,
        "ml_value_rows": 10_944,
        "ml_performance_rows": 20,
        "ml_summary_rows": 12,
        "cnn_patch_rows": 3_648,
        "cnn_fold_rows": 20,
        "cnn_summary_rows": 4,
    }
    actual_counts = {
        "ml_fold_rows": len(ml_folds),
        "ml_value_rows": len(ml_values),
        "ml_performance_rows": len(ml_performance),
        "ml_summary_rows": len(ml_summary),
        "cnn_patch_rows": len(cnn_metrics),
        "cnn_fold_rows": len(cnn_folds),
        "cnn_summary_rows": len(cnn_summary),
    }
    for key, expected in expected_counts.items():
        if actual_counts[key] != expected:
            errors.append(f"{key}: expected {expected}, found {actual_counts[key]}")

    duplicate_ml = ml_folds.duplicated(
        ["analyte", "feature_space", "outer_fold", "feature_group"]
    ).sum()
    duplicate_cnn = cnn_metrics.duplicated(
        ["analyte", "input_mode", "outer_fold", "patch_id"]
    ).sum()
    if duplicate_ml:
        errors.append(f"Duplicate ML explanation rows: {duplicate_ml}")
    if duplicate_cnn:
        errors.append(f"Duplicate CNN explanation rows: {duplicate_cnn}")

    shap_sums = ml_folds.groupby(
        ["analyte", "feature_space", "outer_fold"]
    )["shap_share"].sum()
    permutation_sums = ml_folds.groupby(
        ["analyte", "feature_space", "outer_fold"]
    )["permutation_share"].sum()
    if not np.allclose(shap_sums.to_numpy(), 1.0, atol=1e-10):
        errors.append("SHAP shares do not sum to one within fold")
    if not np.allclose(permutation_sums.to_numpy(), 1.0, atol=1e-10):
        errors.append("Permutation shares do not sum to one within fold")
    maximum_additivity_error = float(ml_folds["shap_additivity_max_error"].max())
    if maximum_additivity_error > 1e-8:
        errors.append(f"SHAP additivity error too large: {maximum_additivity_error}")

    expected_top = {
        ("glucose", "RGB_primary"): "G",
        ("glucose", "HSV_primary"): "Hue",
        ("ketone", "RGB_primary"): "G",
        ("ketone", "HSV_primary"): "Hue",
    }
    for key, feature in expected_top.items():
        row = ml_summary.loc[
            (ml_summary["analyte"] == key[0])
            & (ml_summary["feature_space"] == key[1])
            & (ml_summary["feature_group"] == feature)
        ]
        if len(row) != 1:
            errors.append(f"Missing expected top feature row: {key} {feature}")
            continue
        if int(row.iloc[0]["top_rank_fold_count"]) != 5:
            errors.append(f"Top feature was not rank 1 in all folds: {key} {feature}")
        if int(row.iloc[0]["direction_aligned_fold_count"]) != 5:
            errors.append(f"Direction was not aligned in all folds: {key} {feature}")

    fractions = cnn_metrics[
        [
            "central_area_fraction",
            "central_attention_fraction",
            "positive_central_attention_fraction",
        ]
    ].to_numpy(dtype=float)
    finite_fraction = fractions[np.isfinite(fractions)]
    if finite_fraction.size == 0 or finite_fraction.min() < 0 or finite_fraction.max() > 1:
        errors.append("CNN spatial fractions fall outside [0, 1]")
    maximum_prediction_difference = float(
        cnn_metrics["prediction_reproduction_difference"].max()
    )
    if maximum_prediction_difference > 2e-5:
        errors.append(
            "CNN checkpoint predictions were not reproduced: "
            f"{maximum_prediction_difference}"
        )
    numeric_columns = [
        "central_attention_enrichment",
        "center_occlusion_change_per_10pct_area",
        "outer_occlusion_change_per_10pct_area",
    ]
    if not np.isfinite(cnn_metrics[numeric_columns].to_numpy(dtype=float)).all():
        errors.append("CNN explainability metrics contain non-finite values")

    report = {
        **actual_counts,
        "maximum_shap_additivity_error": maximum_additivity_error,
        "maximum_cnn_prediction_difference": maximum_prediction_difference,
        "expected_top_features": [
            {
                "analyte": analyte,
                "feature_space": feature_space,
                "feature": feature,
            }
            for (analyte, feature_space), feature in expected_top.items()
        ],
        "errors": errors,
        "warnings": warnings,
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "verification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

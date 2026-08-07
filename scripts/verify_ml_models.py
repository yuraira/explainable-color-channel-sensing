"""Verify nested cross-validation outputs from color-feature ML models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "modeling" / "ml"
SOURCE = ROOT / "outputs" / "color_features" / "features.csv"
EXPECTED_PRIMARY = {"RGB_primary", "HSV_primary", "RGB_HSV_combined"}
EXPECTED_SECONDARY = {
    "Chromaticity_secondary",
    "Background_adjusted_secondary",
    "Background_negative_control",
}
EXPECTED_MODELS = {"Ridge", "ElasticNet", "SVR", "RandomForest", "ExtraTrees"}


def check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def main() -> None:
    errors: list[str] = []
    warnings: list[str] = []
    required = [
        "ml_fold_metrics.csv",
        "ml_performance_summary.csv",
        "ml_predictions.csv",
        "ml_concentration_predictions.csv",
        "ml_best_parameters.csv",
        "run_config.json",
        "figures/ml_model_comparison.png",
    ]
    for relative in required:
        check((OUTPUT_DIR / relative).is_file(), f"Missing output: {relative}", errors)
    if errors:
        raise SystemExit("\n".join(errors))

    source = pd.read_csv(SOURCE)
    metrics = pd.read_csv(OUTPUT_DIR / "ml_fold_metrics.csv")
    summary = pd.read_csv(OUTPUT_DIR / "ml_performance_summary.csv")
    predictions = pd.read_csv(OUTPUT_DIR / "ml_predictions.csv")
    concentration = pd.read_csv(OUTPUT_DIR / "ml_concentration_predictions.csv")
    parameters = pd.read_csv(OUTPUT_DIR / "ml_best_parameters.csv")
    config = json.loads((OUTPUT_DIR / "run_config.json").read_text(encoding="utf-8"))

    check(len(metrics) == 380, f"Expected 380 metric rows, found {len(metrics)}", errors)
    check(len(predictions) == 34_656, f"Expected 34,656 prediction rows, found {len(predictions)}", errors)
    check(len(concentration) == 1_805, f"Expected 1,805 concentration rows, found {len(concentration)}", errors)
    check(len(parameters) == 180, f"Expected 180 parameter rows, found {len(parameters)}", errors)
    check(set(metrics["outer_fold"]) == set(range(1, 6)), "Missing outer fold", errors)
    check(
        set(metrics["evaluation_unit"]) == {"patch", "concentration_median"},
        "Unexpected evaluation units",
        errors,
    )
    check(
        metrics.groupby(["analyte", "feature_set", "model", "evaluation_unit"]).size().eq(5).all(),
        "Every candidate/evaluation unit must contain five outer folds",
        errors,
    )
    check(
        EXPECTED_PRIMARY.issubset(set(metrics["feature_set"])),
        "Missing primary feature set",
        errors,
    )
    check(
        EXPECTED_SECONDARY.issubset(set(metrics["feature_set"])),
        "Missing secondary feature set",
        errors,
    )
    check(
        EXPECTED_MODELS.issubset(set(metrics["model"])),
        "Missing model family",
        errors,
    )
    numeric_prediction = predictions[["actual_concentration", "prediction_raw", "prediction"]].to_numpy()
    check(np.isfinite(numeric_prediction).all(), "Predictions contain non-finite values", errors)
    check((predictions["prediction"] >= 0).all(), "Clipped predictions contain negative values", errors)
    for analyte, maximum in {"glucose": 20.0, "ketone": 10.0}.items():
        selected = predictions.loc[predictions["analyte"] == analyte, "prediction"]
        check((selected <= maximum).all(), f"{analyte} predictions exceed physical range", errors)

    for (analyte, feature_set, model), subset in predictions.groupby(
        ["analyte", "feature_set", "model"]
    ):
        expected_ids = set(source.loc[source["analyte"] == analyte, "patch_id"])
        check(
            set(subset["patch_id"]) == expected_ids,
            f"Prediction coverage mismatch: {analyte} {feature_set} {model}",
            errors,
        )
        check(
            subset["patch_id"].nunique() == len(subset),
            f"Duplicate outer prediction: {analyte} {feature_set} {model}",
            errors,
        )

    patch_metrics = metrics.loc[metrics["evaluation_unit"] == "patch"]
    recomputed = (
        patch_metrics.groupby(["analyte", "feature_set", "model"])["mae"]
        .mean()
        .sort_index()
    )
    reported = (
        summary.loc[summary["evaluation_unit"] == "patch"]
        .set_index(["analyte", "feature_set", "model"])["mae_mean"]
        .sort_index()
    )
    check(
        np.allclose(recomputed.to_numpy(), reported.to_numpy(), atol=1e-12),
        "Summary MAE does not reconcile to fold metrics",
        errors,
    )
    check(config.get("random_seed") == 240920, "Unexpected random seed", errors)
    check(config.get("inner_folds") == 5, "Unexpected inner fold count", errors)

    primary_summary = summary.loc[
        (summary["evaluation_unit"] == "patch")
        & summary["feature_set"].isin(EXPECTED_PRIMARY)
        & summary["model"].isin(EXPECTED_MODELS)
    ]
    best_models = (
        primary_summary.sort_values(["analyte", "mae_mean"])
        .groupby("analyte", as_index=False)
        .first()[["analyte", "feature_set", "model", "mae_mean", "rmse_mean", "r2_mean"]]
        .to_dict(orient="records")
    )
    report = {
        "fold_metric_rows": len(metrics),
        "prediction_rows": len(predictions),
        "concentration_prediction_rows": len(concentration),
        "best_parameter_rows": len(parameters),
        "best_primary_models": best_models,
        "errors": errors,
        "warnings": warnings,
    }
    (OUTPUT_DIR / "verification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

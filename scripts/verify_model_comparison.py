"""Verify combined ML-versus-CNN comparison outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "modeling" / "comparison"
SOURCE = ROOT / "outputs" / "color_features" / "features.csv"
MODELS = {
    "Mean baseline",
    "Background only",
    "Nested-selected ML",
    "CNN central ROI",
    "CNN full patch",
}
HEADLINE_MODELS = {"Nested-selected ML", "CNN central ROI", "CNN full patch"}


def check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def main() -> None:
    errors: list[str] = []
    warnings: list[str] = []
    required = [
        "selected_ml_folds.csv",
        "model_comparison_fold_metrics.csv",
        "model_comparison_summary.csv",
        "headline_model_predictions.csv",
        "headline_concentration_predictions.csv",
        "concentration_error_summary.csv",
        "model_efficiency.csv",
        "run_config.json",
        "figures/model_performance_comparison.png",
        "figures/observed_vs_predicted.png",
        "figures/concentration_error_profiles.png",
    ]
    for relative in required:
        check((OUTPUT_DIR / relative).is_file(), f"Missing output: {relative}", errors)
    if errors:
        raise SystemExit("\n".join(errors))

    source = pd.read_csv(SOURCE)
    selected = pd.read_csv(OUTPUT_DIR / "selected_ml_folds.csv")
    metrics = pd.read_csv(OUTPUT_DIR / "model_comparison_fold_metrics.csv")
    summary = pd.read_csv(OUTPUT_DIR / "model_comparison_summary.csv")
    predictions = pd.read_csv(OUTPUT_DIR / "headline_model_predictions.csv")
    concentration = pd.read_csv(OUTPUT_DIR / "headline_concentration_predictions.csv")
    error_profile = pd.read_csv(OUTPUT_DIR / "concentration_error_summary.csv")
    efficiency = pd.read_csv(OUTPUT_DIR / "model_efficiency.csv")
    config = json.loads((OUTPUT_DIR / "run_config.json").read_text(encoding="utf-8"))

    check(len(selected) == 10, f"Expected 10 selected ML fold rows, found {len(selected)}", errors)
    check(len(metrics) == 100, f"Expected 100 metric rows, found {len(metrics)}", errors)
    check(len(summary) == 20, f"Expected 20 summary rows, found {len(summary)}", errors)
    check(len(predictions) == 5_472, f"Expected 5,472 prediction rows, found {len(predictions)}", errors)
    check(len(concentration) == 285, f"Expected 285 concentration rows, found {len(concentration)}", errors)
    check(len(error_profile) == 57, f"Expected 57 concentration-error rows, found {len(error_profile)}", errors)
    check(len(efficiency) == 30, f"Expected 30 efficiency rows, found {len(efficiency)}", errors)
    check(set(metrics["comparison_model"]) == MODELS, "Comparison model set mismatch", errors)
    check(
        metrics.groupby(["analyte", "comparison_model", "evaluation_unit"]).size().eq(5).all(),
        "Every comparison model/evaluation group must contain five folds",
        errors,
    )
    check(
        set(predictions["comparison_model"]) == HEADLINE_MODELS,
        "Headline model set mismatch",
        errors,
    )
    check(np.isfinite(predictions["prediction"]).all(), "Non-finite headline prediction", errors)
    for (analyte, model), subset in predictions.groupby(["analyte", "comparison_model"]):
        expected_ids = set(source.loc[source["analyte"] == analyte, "patch_id"])
        check(
            set(subset["patch_id"]) == expected_ids,
            f"Prediction coverage mismatch: {analyte} {model}",
            errors,
        )
        check(
            subset["patch_id"].nunique() == len(subset),
            f"Duplicate outer prediction: {analyte} {model}",
            errors,
        )

    patch_metrics = metrics.loc[metrics["evaluation_unit"] == "patch"]
    recomputed = (
        patch_metrics.groupby(["analyte", "comparison_model"])["mae"]
        .mean()
        .sort_index()
    )
    reported = (
        summary.loc[summary["evaluation_unit"] == "patch"]
        .set_index(["analyte", "comparison_model"])["mae_mean"]
        .sort_index()
    )
    check(
        np.allclose(recomputed.to_numpy(), reported.to_numpy(), atol=1e-12),
        "Comparison summary MAE does not reconcile to fold metrics",
        errors,
    )
    best_headline = (
        summary.loc[
            (summary["evaluation_unit"] == "patch")
            & (summary["comparison_model"].isin(HEADLINE_MODELS))
        ]
        .sort_values(["analyte", "mae_mean"])
        .groupby("analyte", as_index=False)
        .first()[["analyte", "comparison_model", "mae_mean", "rmse_mean", "r2_mean"]]
        .to_dict(orient="records")
    )
    check(
        "same source image" in config.get("known_limitation", ""),
        "Known source-image limitation is not recorded",
        errors,
    )
    report = {
        "selected_ml_rows": len(selected),
        "comparison_metric_rows": len(metrics),
        "headline_prediction_rows": len(predictions),
        "headline_concentration_rows": len(concentration),
        "best_headline_models": best_headline,
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

"""Verify lightweight CNN training and prediction outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "modeling" / "dl"
SOURCE = ROOT / "outputs" / "color_features" / "features.csv"
INPUT_MODES = {"roi_masked", "full_patch"}


def check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def main() -> None:
    errors: list[str] = []
    warnings: list[str] = []
    required = [
        "dl_fold_metrics.csv",
        "dl_performance_summary.csv",
        "dl_predictions.csv",
        "dl_concentration_predictions.csv",
        "training_history.csv",
        "target_transform_selection.csv",
        "run_config.json",
        "figures/dl_training_curves.png",
        "figures/dl_input_comparison.png",
    ]
    for relative in required:
        check((OUTPUT_DIR / relative).is_file(), f"Missing output: {relative}", errors)
    checkpoint_files = sorted((OUTPUT_DIR / "checkpoints").glob("*.pt"))
    check(len(checkpoint_files) == 20, f"Expected 20 checkpoints, found {len(checkpoint_files)}", errors)
    if errors:
        raise SystemExit("\n".join(errors))

    source = pd.read_csv(SOURCE)
    metrics = pd.read_csv(OUTPUT_DIR / "dl_fold_metrics.csv")
    summary = pd.read_csv(OUTPUT_DIR / "dl_performance_summary.csv")
    predictions = pd.read_csv(OUTPUT_DIR / "dl_predictions.csv")
    concentration = pd.read_csv(OUTPUT_DIR / "dl_concentration_predictions.csv")
    history = pd.read_csv(OUTPUT_DIR / "training_history.csv")
    selection = pd.read_csv(OUTPUT_DIR / "target_transform_selection.csv")
    config = json.loads((OUTPUT_DIR / "run_config.json").read_text(encoding="utf-8"))

    check(len(metrics) == 40, f"Expected 40 metric rows, found {len(metrics)}", errors)
    check(len(predictions) == 3_648, f"Expected 3,648 prediction rows, found {len(predictions)}", errors)
    check(len(concentration) == 190, f"Expected 190 concentration rows, found {len(concentration)}", errors)
    check(len(selection) == 40, f"Expected 40 target-selection rows, found {len(selection)}", errors)
    check(len(history) > 0, "Training history is empty", errors)
    check(set(metrics["outer_fold"]) == set(range(1, 6)), "Missing outer fold", errors)
    check(set(metrics["input_mode"]) == INPUT_MODES, "Unexpected input mode", errors)
    check(
        set(metrics["evaluation_unit"]) == {"patch", "concentration_median"},
        "Unexpected evaluation unit",
        errors,
    )
    check(
        metrics.groupby(["analyte", "input_mode", "evaluation_unit"]).size().eq(5).all(),
        "Every analyte/input/evaluation group must contain five folds",
        errors,
    )
    numeric = predictions[["actual_concentration", "prediction_raw", "prediction"]].to_numpy()
    check(np.isfinite(numeric).all(), "Predictions contain non-finite values", errors)
    check((predictions["prediction"] >= 0).all(), "Predictions contain negative values", errors)
    for analyte, maximum in {"glucose": 20.0, "ketone": 10.0}.items():
        selected_predictions = predictions.loc[predictions["analyte"] == analyte]
        check(
            (selected_predictions["prediction"] <= maximum).all(),
            f"{analyte} predictions exceed the physical range",
            errors,
        )
        expected_ids = set(source.loc[source["analyte"] == analyte, "patch_id"])
        for input_mode, mode_data in selected_predictions.groupby("input_mode"):
            check(
                set(mode_data["patch_id"]) == expected_ids,
                f"Prediction coverage mismatch: {analyte} {input_mode}",
                errors,
            )
            check(
                mode_data["patch_id"].nunique() == len(mode_data),
                f"Duplicate outer prediction: {analyte} {input_mode}",
                errors,
            )

    selected_metrics = metrics.loc[metrics["evaluation_unit"] == "patch"]
    for row in selected_metrics.itertuples(index=False):
        candidates = selection.loc[
            (selection["analyte"] == row.analyte)
            & (selection["outer_fold"] == row.outer_fold)
            & (selection["input_mode"] == row.input_mode)
        ].sort_values("validation_mae")
        check(len(candidates) == 2, "Missing raw/log1p validation candidate", errors)
        if len(candidates) == 2:
            check(
                str(candidates.iloc[0]["target_transform"]) == str(row.target_transform),
                f"Target transform was not selected by validation MAE: {row.analyte} fold {row.outer_fold} {row.input_mode}",
                errors,
            )

    recomputed = (
        selected_metrics.groupby(["analyte", "input_mode"])["mae"]
        .mean()
        .sort_index()
    )
    reported = (
        summary.loc[summary["evaluation_unit"] == "patch"]
        .set_index(["analyte", "input_mode"])["mae_mean"]
        .sort_index()
    )
    check(
        np.allclose(recomputed.to_numpy(), reported.to_numpy(), atol=1e-12),
        "Summary MAE does not reconcile to fold metrics",
        errors,
    )
    check((selected_metrics["parameter_count"] == 169_049).all(), "Unexpected parameter count", errors)
    check(config.get("random_seed") == 240920, "Unexpected random seed", errors)
    check(config.get("color_augmentation") is False, "Color augmentation must be disabled", errors)

    for checkpoint_path in checkpoint_files:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        required_keys = {
            "state_dict",
            "analyte",
            "input_mode",
            "outer_fold",
            "target_transform",
            "target_mean",
            "target_std",
            "image_size",
            "roi_radius_fraction",
        }
        check(
            required_keys.issubset(checkpoint),
            f"Incomplete checkpoint metadata: {checkpoint_path.name}",
            errors,
        )

    patch_summary = summary.loc[summary["evaluation_unit"] == "patch"]
    report = {
        "fold_metric_rows": len(metrics),
        "prediction_rows": len(predictions),
        "concentration_prediction_rows": len(concentration),
        "training_history_rows": len(history),
        "checkpoints": len(checkpoint_files),
        "patch_performance": patch_summary[
            ["analyte", "input_mode", "mae_mean", "rmse_mean", "r2_mean", "normalized_mae_mean"]
        ].to_dict(orient="records"),
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

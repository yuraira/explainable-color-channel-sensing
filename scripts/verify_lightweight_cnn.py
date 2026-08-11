"""Verify the controlled CNN lightweighting experiment and its checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_dl_models import TargetTransform  # noqa: E402
from train_lightweight_cnn import (  # noqa: E402
    EVALUATED_VARIANTS,
    MODEL_IMAGE_SIZE,
    build_model,
    preload_images,
)


OUTPUT_DIR = ROOT / "outputs" / "modeling" / "cnn_lightweight"
FEATURES = ROOT / "outputs" / "color_features" / "features.csv"
PATCH_ROOT = ROOT / "outputs" / "patch_detection"
EXPECTED_ANALYTE_PATCHES = {"glucose": 1_056, "ketone": 768}


def check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


@torch.no_grad()
def reproduce_predictions(
    checkpoint_path: Path,
    rows: pd.DataFrame,
    tensors: dict[str, torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    architecture = str(checkpoint["architecture"])
    representation = str(checkpoint["representation"])
    model = build_model(architecture, representation)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device=device, memory_format=torch.channels_last).eval()
    transform = TargetTransform(
        mode=str(checkpoint["target_transform"]),
        mean=float(checkpoint["target_mean"]),
        std=float(checkpoint["target_std"]),
    )
    standardized: list[np.ndarray] = []
    patch_ids = rows["patch_id"].astype(str).tolist()
    for start in range(0, len(patch_ids), 128):
        batch = torch.stack(
            [tensors[patch_id] for patch_id in patch_ids[start : start + 128]]
        ).to(device=device, memory_format=torch.channels_last)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            output = model(batch)
        standardized.append(output.detach().float().cpu().numpy())
    return transform.decode(np.concatenate(standardized))


def main() -> None:
    errors: list[str] = []
    warnings: list[str] = []
    required = [
        "cnn_lightweight_fold_metrics.csv",
        "cnn_lightweight_performance_summary.csv",
        "cnn_lightweight_predictions.csv",
        "cnn_lightweight_concentration_predictions.csv",
        "cnn_lightweight_training_history.csv",
        "cnn_lightweight_target_selection.csv",
        "cnn_lightweight_efficiency.csv",
        "run_config.json",
        "figures/cnn_lightweight_tradeoffs.png",
    ]
    for relative in required:
        check((OUTPUT_DIR / relative).is_file(), f"Missing output: {relative}", errors)
    checkpoints = sorted((OUTPUT_DIR / "checkpoints").glob("*.pt"))
    check(len(checkpoints) == 40, f"Expected 40 checkpoints, found {len(checkpoints)}", errors)
    if errors:
        raise SystemExit("\n".join(errors))

    features = pd.read_csv(FEATURES)
    metrics = pd.read_csv(OUTPUT_DIR / "cnn_lightweight_fold_metrics.csv")
    summary = pd.read_csv(OUTPUT_DIR / "cnn_lightweight_performance_summary.csv")
    predictions = pd.read_csv(OUTPUT_DIR / "cnn_lightweight_predictions.csv")
    concentration = pd.read_csv(
        OUTPUT_DIR / "cnn_lightweight_concentration_predictions.csv"
    )
    history = pd.read_csv(OUTPUT_DIR / "cnn_lightweight_training_history.csv")
    selection = pd.read_csv(OUTPUT_DIR / "cnn_lightweight_target_selection.csv")
    efficiency = pd.read_csv(OUTPUT_DIR / "cnn_lightweight_efficiency.csv")
    config = json.loads((OUTPUT_DIR / "run_config.json").read_text(encoding="utf-8"))
    expected_variants = set(EVALUATED_VARIANTS)
    metric_variants = set(zip(metrics["architecture"], metrics["representation"], strict=True))

    check(len(metrics) == 80, f"Expected 80 metric rows, found {len(metrics)}", errors)
    check(len(predictions) == 7_296, f"Expected 7,296 predictions, found {len(predictions)}", errors)
    check(len(concentration) == 380, f"Expected 380 concentration rows, found {len(concentration)}", errors)
    check(len(selection) == 40, f"Expected 40 selection rows, found {len(selection)}", errors)
    check(len(history) > 0, "Training history is empty", errors)
    check(metric_variants == expected_variants, "Unexpected evaluated variants", errors)
    check(set(metrics["outer_fold"]) == set(range(1, 6)), "Missing outer fold", errors)
    check(
        set(metrics["evaluation_unit"]) == {"patch", "concentration_median"},
        "Unexpected evaluation unit",
        errors,
    )
    check(
        metrics.groupby(
            ["analyte", "architecture", "representation", "evaluation_unit"]
        ).size().eq(5).all(),
        "Every analyte/variant/evaluation group must contain five folds",
        errors,
    )
    check(
        np.isfinite(
            predictions[["actual_concentration", "prediction_raw", "prediction"]]
        ).to_numpy().all(),
        "Predictions contain non-finite values",
        errors,
    )
    check((predictions["prediction"] >= 0).all(), "Negative clipped prediction", errors)
    for analyte, maximum in {"glucose": 20.0, "ketone": 10.0}.items():
        analyte_predictions = predictions.loc[predictions["analyte"] == analyte]
        check(
            (analyte_predictions["prediction"] <= maximum).all(),
            f"{analyte} predictions exceed the physical range",
            errors,
        )
        expected_ids = set(features.loc[features["analyte"] == analyte, "patch_id"])
        for (architecture, representation), rows in analyte_predictions.groupby(
            ["architecture", "representation"]
        ):
            check(
                len(rows) == EXPECTED_ANALYTE_PATCHES[analyte],
                f"Unexpected prediction count: {analyte} {architecture}/{representation}",
                errors,
            )
            check(
                set(rows["patch_id"]) == expected_ids and rows["patch_id"].is_unique,
                f"Prediction coverage mismatch: {analyte} {architecture}/{representation}",
                errors,
            )

    patch_metrics = metrics.loc[metrics["evaluation_unit"] == "patch"]
    recomputed = patch_metrics.groupby(
        ["analyte", "architecture", "representation"]
    )["mae"].mean().sort_index()
    reported = summary.loc[summary["evaluation_unit"] == "patch"].set_index(
        ["analyte", "architecture", "representation"]
    )["mae_mean"].sort_index()
    check(
        recomputed.index.equals(reported.index)
        and np.allclose(recomputed.to_numpy(), reported.to_numpy(), atol=1e-12),
        "Summary MAE does not reconcile to fold metrics",
        errors,
    )
    check(
        set(zip(efficiency["architecture"], efficiency["representation"], strict=True))
        == expected_variants,
        "Efficiency table does not cover every variant",
        errors,
    )
    check(config.get("random_seed") == 240920, "Unexpected random seed", errors)
    check(
        config.get("lightweight_experiment_image_size") == MODEL_IMAGE_SIZE,
        "Unexpected lightweight image size",
        errors,
    )
    check(
        config.get("target_transform_candidates") == ["raw"],
        "Lightweight comparison must use a fixed raw target",
        errors,
    )

    metadata = features[["patch_id", "analyte", "crop_file", "circle_radius_px"]]
    tensor_cache: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maximum_checkpoint_difference = 0.0
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        required_keys = {
            "state_dict",
            "analyte",
            "architecture",
            "representation",
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
        analyte = str(checkpoint["analyte"])
        architecture = str(checkpoint["architecture"])
        representation = str(checkpoint["representation"])
        outer_fold = int(checkpoint["outer_fold"])
        key = (analyte, representation)
        if key not in tensor_cache:
            tensor_cache[key] = preload_images(
                metadata.loc[metadata["analyte"] == analyte],
                PATCH_ROOT,
                representation,
            )
        rows = predictions.loc[
            (predictions["analyte"] == analyte)
            & (predictions["architecture"] == architecture)
            & (predictions["representation"] == representation)
            & (predictions["outer_fold"] == outer_fold)
        ]
        reproduced = reproduce_predictions(
            checkpoint_path, rows, tensor_cache[key], device
        )
        difference = float(
            np.max(np.abs(reproduced - rows["prediction_raw"].to_numpy(dtype=float)))
        )
        maximum_checkpoint_difference = max(maximum_checkpoint_difference, difference)
    check(
        maximum_checkpoint_difference <= 1e-5,
        f"Checkpoint predictions differ by {maximum_checkpoint_difference:.3g}",
        errors,
    )

    report = {
        "fold_metric_rows": len(metrics),
        "prediction_rows": len(predictions),
        "concentration_prediction_rows": len(concentration),
        "training_history_rows": len(history),
        "checkpoints": len(checkpoints),
        "maximum_checkpoint_prediction_difference": maximum_checkpoint_difference,
        "device": str(device),
        "patch_performance": summary.loc[summary["evaluation_unit"] == "patch", [
            "analyte",
            "architecture",
            "representation",
            "mae_mean",
            "mae_std",
            "rmse_mean",
            "r2_mean",
        ]].to_dict(orient="records"),
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

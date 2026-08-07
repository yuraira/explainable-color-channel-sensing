"""Analyze CNN spatial reliance with absolute Grad-CAM and region occlusion."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.nn import functional as F

from train_dl_models import IMAGE_SIZE, TinyColorCNN, load_image_tensor, preload_images


RANDOM_SEED = 240920
N_SPLITS = 5
BATCH_SIZE = 64
ROI_RADIUS_FRACTION = 0.70
ANALYTES = ["glucose", "ketone"]
INPUT_MODES = ["roi_masked", "full_patch"]
ANALYTE_LABELS = {"glucose": "Glucose", "ketone": "Ketone"}
MODE_LABELS = {"roi_masked": "Central ROI CNN", "full_patch": "Full-patch CNN"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("outputs/color_features/features.csv"),
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("outputs/data_splits/nested_split_assignments.csv"),
    )
    parser.add_argument(
        "--patch-root",
        type=Path,
        default=Path("outputs/patch_detection"),
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=Path("outputs/modeling/dl/checkpoints"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/modeling/dl/dl_predictions.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/explainability/cnn"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {path}. Use --overwrite to replace it."
            )
        shutil.rmtree(path)
    (path / "figures").mkdir(parents=True, exist_ok=True)


def set_reproducible_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def decode_target(values: np.ndarray, checkpoint: dict[str, object]) -> np.ndarray:
    transformed = (
        values.astype(float) * float(checkpoint["target_std"])
        + float(checkpoint["target_mean"])
    )
    if str(checkpoint["target_transform"]) == "log1p":
        return np.expm1(transformed)
    return transformed


def resized_roi_mask(image_path: Path, circle_radius_px: float) -> torch.Tensor:
    with Image.open(image_path) as image:
        width, height = image.size
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    y_grid, x_grid = np.ogrid[:height, :width]
    radius = ROI_RADIUS_FRACTION * float(circle_radius_px)
    mask = ((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2 <= radius**2).astype(
        np.float32
    )
    tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
    return (
        F.interpolate(tensor, size=(IMAGE_SIZE, IMAGE_SIZE), mode="nearest")
        .squeeze(0)
        .squeeze(0)
        .bool()
    )


def regression_gradcam(
    model: TinyColorCNN,
    images: torch.Tensor,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Return standardized predictions, absolute signed CAM, and positive CAM."""
    model.zero_grad(set_to_none=True)
    activation = model.features[:15](images)
    activation.retain_grad()
    tail = model.features[15:](activation)
    output = model.regressor(tail).squeeze(1)
    output.sum().backward()
    gradient = activation.grad
    if gradient is None:
        raise RuntimeError("Grad-CAM gradient was not retained")
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    signed = (weights * activation).sum(dim=1, keepdim=True)
    absolute = F.interpolate(
        signed.abs(), size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
    ).squeeze(1)
    positive = F.interpolate(
        F.relu(signed), size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
    ).squeeze(1)
    return output.detach().cpu().numpy(), absolute.detach(), positive.detach()


@torch.no_grad()
def predict_standardized(model: TinyColorCNN, images: torch.Tensor) -> np.ndarray:
    return model(images).detach().cpu().numpy()


def plot_region_summary(fold_summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    mode_colors = {"roi_masked": "#4e9f50", "full_patch": "#e3ad36"}
    rng = np.random.default_rng(RANDOM_SEED)
    for row_index, analyte in enumerate(ANALYTES):
        data = fold_summary.loc[fold_summary["analyte"] == analyte]
        attention_axis = axes[row_index, 0]
        for x_index, input_mode in enumerate(INPUT_MODES):
            values = data.loc[
                data["input_mode"] == input_mode, "central_attention_fraction"
            ].to_numpy(dtype=float)
            attention_axis.bar(
                x_index,
                100 * np.mean(values),
                yerr=100 * np.std(values, ddof=1),
                color=mode_colors[input_mode],
                width=0.62,
                capsize=4,
            )
            attention_axis.scatter(
                x_index + rng.normal(0, 0.035, len(values)),
                100 * values,
                color="#222222",
                s=22,
                zorder=3,
            )
        area_reference = 100 * float(data["central_area_fraction"].mean())
        attention_axis.axhline(
            area_reference,
            color="#666666",
            linestyle="--",
            linewidth=1.2,
            label=f"Central ROI area ({area_reference:.1f}%)",
        )
        attention_axis.set_xticks([0, 1], ["Central ROI", "Full patch"])
        attention_axis.set_ylabel("Absolute Grad-CAM inside central ROI (%)")
        attention_axis.set_ylim(0, 100)
        attention_axis.set_title(f"{ANALYTE_LABELS[analyte]} · spatial sensitivity")
        attention_axis.legend(frameon=False, fontsize=9, loc="upper right")
        attention_axis.grid(axis="y", alpha=0.22)

        occlusion_axis = axes[row_index, 1]
        full = data.loc[data["input_mode"] == "full_patch"]
        labels = ["Central ROI occluded", "Outside ROI occluded"]
        columns = [
            "center_occlusion_change_per_10pct_area",
            "outer_occlusion_change_per_10pct_area",
        ]
        colors = ["#477db3", "#b47955"]
        for x_index, (column, color) in enumerate(zip(columns, colors, strict=True)):
            values = full[column].to_numpy(dtype=float)
            occlusion_axis.bar(
                x_index,
                np.mean(values),
                yerr=np.std(values, ddof=1),
                color=color,
                width=0.62,
                capsize=4,
            )
            occlusion_axis.scatter(
                x_index + rng.normal(0, 0.035, len(values)),
                values,
                color="#222222",
                s=22,
                zorder=3,
            )
        occlusion_axis.set_xticks([0, 1], labels, rotation=8)
        occlusion_axis.set_ylabel("Prediction change per 10% occluded area (mg/mL)")
        occlusion_axis.set_title(
            f"{ANALYTE_LABELS[analyte]} · area-normalized full-patch occlusion"
        )
        occlusion_axis.grid(axis="y", alpha=0.22)
    fig.suptitle(
        "CNN reliance on the sensor center and surrounding pixels",
        fontsize=18,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def representative_orders(data: pd.DataFrame) -> list[int]:
    orders = sorted(data["concentration_order"].astype(int).unique())
    return [orders[0], orders[len(orders) // 2], orders[-1]]


def plot_examples(
    metrics: pd.DataFrame,
    maps: dict[tuple[str, str], np.ndarray],
    metadata: pd.DataFrame,
    patch_root: Path,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(11, 13), constrained_layout=True)
    row_index = 0
    metadata_index = metadata.set_index("patch_id")
    for analyte in ANALYTES:
        for input_mode in INPUT_MODES:
            subset = metrics.loc[
                (metrics["analyte"] == analyte) & (metrics["input_mode"] == input_mode)
            ]
            for column_index, concentration_order in enumerate(representative_orders(subset)):
                candidates = subset.loc[
                    subset["concentration_order"] == concentration_order
                ].copy()
                median_error = float(candidates["absolute_error"].median())
                selected = candidates.iloc[
                    np.argmin(np.abs(candidates["absolute_error"] - median_error))
                ]
                patch_id = str(selected["patch_id"])
                row = metadata_index.loc[patch_id]
                tensor = load_image_tensor(
                    patch_root / str(row["crop_file"]),
                    float(row["circle_radius_px"]),
                    input_mode,
                )
                display = np.clip(
                    np.transpose((tensor.numpy() * 0.25 + 0.5), (1, 2, 0)), 0, 1
                )
                heat = maps[(input_mode, patch_id)].astype(float) / 255.0
                heat_rgb = plt.get_cmap("inferno")(heat)[..., :3]
                overlay = np.clip(0.58 * display + 0.42 * heat_rgb, 0, 1)
                axis = axes[row_index, column_index]
                axis.imshow(overlay)
                axis.set_title(
                    f"{selected['actual_concentration']:g} mg/mL\n"
                    f"pred. {selected['prediction_reported']:g}",
                    fontsize=10,
                )
                if column_index == 0:
                    axis.text(
                        0.02,
                        0.98,
                        f"{ANALYTE_LABELS[analyte]}\n{MODE_LABELS[input_mode]}",
                        transform=axis.transAxes,
                        va="top",
                        ha="left",
                        color="white",
                        fontsize=9,
                        fontweight="bold",
                        bbox={
                            "facecolor": "black",
                            "alpha": 0.55,
                            "edgecolor": "none",
                            "pad": 3,
                        },
                    )
                axis.axis("off")
            row_index += 1
    fig.suptitle(
        "Representative absolute Grad-CAM maps",
        fontsize=18,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    prepare_output_dir(args.output_dir, args.overwrite)
    set_reproducible_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = pd.read_csv(args.features)
    splits = pd.read_csv(args.splits)
    expected_predictions = pd.read_csv(args.predictions)
    expected_lookup = expected_predictions.set_index(
        ["analyte", "input_mode", "outer_fold", "patch_id"]
    )[["prediction_raw", "prediction"]]
    metadata_index = features.set_index("patch_id")

    metric_rows: list[dict[str, object]] = []
    maps: dict[tuple[str, str], np.ndarray] = {}
    maximum_prediction_difference = 0.0
    for analyte in ANALYTES:
        analyte_metadata = features.loc[features["analyte"] == analyte].copy()
        for input_mode in INPUT_MODES:
            tensors = preload_images(analyte_metadata, args.patch_root, input_mode)
            for outer_fold in range(1, N_SPLITS + 1):
                checkpoint_path = (
                    args.checkpoints / f"{analyte}_{input_mode}_fold{outer_fold}.pt"
                )
                checkpoint = torch.load(
                    checkpoint_path, map_location=device, weights_only=False
                )
                model = TinyColorCNN().to(device)
                model.load_state_dict(checkpoint["state_dict"])
                model.eval()
                test_ids = splits.loc[
                    (splits["analyte"] == analyte)
                    & (splits["outer_fold"] == outer_fold)
                    & (splits["dl_role"] == "test"),
                    "patch_id",
                ].astype(str).tolist()
                for start in range(0, len(test_ids), BATCH_SIZE):
                    batch_ids = test_ids[start : start + BATCH_SIZE]
                    images = torch.stack([tensors[patch_id] for patch_id in batch_ids]).to(
                        device
                    )
                    masks = torch.stack(
                        [
                            resized_roi_mask(
                                args.patch_root / str(metadata_index.loc[patch_id, "crop_file"]),
                                float(metadata_index.loc[patch_id, "circle_radius_px"]),
                            )
                            for patch_id in batch_ids
                        ]
                    ).to(device)
                    standardized, absolute_cam, positive_cam = regression_gradcam(
                        model, images
                    )
                    prediction_raw = decode_target(standardized, checkpoint)
                    center_occluded = images.detach().clone()
                    center_occluded[masks.unsqueeze(1).expand_as(center_occluded)] = 0.0
                    outer_occluded = images.detach().clone()
                    outer_occluded[(~masks).unsqueeze(1).expand_as(outer_occluded)] = 0.0
                    center_prediction = decode_target(
                        predict_standardized(model, center_occluded), checkpoint
                    )
                    outer_prediction = decode_target(
                        predict_standardized(model, outer_occluded), checkpoint
                    )
                    for index, patch_id in enumerate(batch_ids):
                        expected_row = expected_lookup.loc[
                            (analyte, input_mode, outer_fold, patch_id)
                        ]
                        expected = float(expected_row["prediction_raw"])
                        reported_prediction = float(expected_row["prediction"])
                        difference = abs(float(prediction_raw[index]) - expected)
                        maximum_prediction_difference = max(
                            maximum_prediction_difference, difference
                        )
                        cam = absolute_cam[index].cpu().numpy()
                        positive = positive_cam[index].cpu().numpy()
                        mask = masks[index].detach().cpu().numpy().astype(bool)
                        total = float(cam.sum())
                        positive_total = float(positive.sum())
                        central_fraction = float(cam[mask].sum() / total) if total > 0 else math.nan
                        positive_central_fraction = (
                            float(positive[mask].sum() / positive_total)
                            if positive_total > 0
                            else math.nan
                        )
                        area_fraction = float(mask.mean())
                        normalized_map = cam / max(float(np.percentile(cam, 99)), 1e-12)
                        maps[(input_mode, patch_id)] = np.round(
                            255 * np.clip(normalized_map, 0, 1)
                        ).astype(np.uint8)
                        row = metadata_index.loc[patch_id]
                        actual = float(row["concentration_mg_ml"])
                        metric_rows.append(
                            {
                                "analyte": analyte,
                                "input_mode": input_mode,
                                "outer_fold": outer_fold,
                                "patch_id": patch_id,
                                "concentration_order": int(row["concentration_order"]),
                                "actual_concentration": actual,
                                "prediction_raw": float(prediction_raw[index]),
                                "prediction_reported": reported_prediction,
                                "absolute_error": abs(reported_prediction - actual),
                                "central_area_fraction": area_fraction,
                                "central_attention_fraction": central_fraction,
                                "central_attention_enrichment": central_fraction / area_fraction,
                                "positive_central_attention_fraction": positive_central_fraction,
                                "center_occlusion_change": abs(
                                    float(center_prediction[index]) - float(prediction_raw[index])
                                ),
                                "outer_occlusion_change": abs(
                                    float(outer_prediction[index]) - float(prediction_raw[index])
                                ),
                                "center_occlusion_change_per_10pct_area": (
                                    0.1
                                    * abs(
                                        float(center_prediction[index])
                                        - float(prediction_raw[index])
                                    )
                                    / area_fraction
                                ),
                                "outer_occlusion_change_per_10pct_area": (
                                    0.1
                                    * abs(
                                        float(outer_prediction[index])
                                        - float(prediction_raw[index])
                                    )
                                    / (1.0 - area_fraction)
                                ),
                                "prediction_reproduction_difference": difference,
                            }
                        )
                print(
                    f"Explained {analyte} {input_mode} fold={outer_fold}", flush=True
                )
    if maximum_prediction_difference > 2e-5:
        raise RuntimeError(
            "Checkpoint prediction reproduction failed: "
            f"{maximum_prediction_difference}"
        )
    metrics = pd.DataFrame(metric_rows)
    fold_summary = (
        metrics.groupby(["analyte", "input_mode", "outer_fold"], as_index=False)
        .agg(
            central_area_fraction=("central_area_fraction", "mean"),
            central_attention_fraction=("central_attention_fraction", "mean"),
            central_attention_enrichment=("central_attention_enrichment", "mean"),
            positive_central_attention_fraction=(
                "positive_central_attention_fraction",
                "mean",
            ),
            center_occlusion_change=("center_occlusion_change", "median"),
            outer_occlusion_change=("outer_occlusion_change", "median"),
            center_occlusion_change_per_10pct_area=(
                "center_occlusion_change_per_10pct_area",
                "median",
            ),
            outer_occlusion_change_per_10pct_area=(
                "outer_occlusion_change_per_10pct_area",
                "median",
            ),
            prediction_reproduction_difference=(
                "prediction_reproduction_difference",
                "max",
            ),
        )
    )
    summary = (
        fold_summary.groupby(["analyte", "input_mode"], as_index=False)
        .agg(
            central_area_fraction_mean=("central_area_fraction", "mean"),
            central_attention_fraction_mean=("central_attention_fraction", "mean"),
            central_attention_fraction_std=("central_attention_fraction", "std"),
            central_attention_enrichment_mean=("central_attention_enrichment", "mean"),
            center_occlusion_change_mean=("center_occlusion_change", "mean"),
            center_occlusion_change_std=("center_occlusion_change", "std"),
            outer_occlusion_change_mean=("outer_occlusion_change", "mean"),
            outer_occlusion_change_std=("outer_occlusion_change", "std"),
            center_occlusion_change_per_10pct_area_mean=(
                "center_occlusion_change_per_10pct_area",
                "mean",
            ),
            center_occlusion_change_per_10pct_area_std=(
                "center_occlusion_change_per_10pct_area",
                "std",
            ),
            outer_occlusion_change_per_10pct_area_mean=(
                "outer_occlusion_change_per_10pct_area",
                "mean",
            ),
            outer_occlusion_change_per_10pct_area_std=(
                "outer_occlusion_change_per_10pct_area",
                "std",
            ),
        )
    )
    metrics.to_csv(args.output_dir / "cnn_gradcam_patch_metrics.csv", index=False)
    fold_summary.to_csv(args.output_dir / "cnn_gradcam_fold_summary.csv", index=False)
    summary.to_csv(args.output_dir / "cnn_gradcam_summary.csv", index=False)
    plot_region_summary(
        fold_summary, args.output_dir / "figures" / "cnn_region_dependence.png"
    )
    plot_examples(
        metrics,
        maps,
        features,
        args.patch_root,
        args.output_dir / "figures" / "cnn_gradcam_examples.png",
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "random_seed": RANDOM_SEED,
                "device": str(device),
                "outer_folds": N_SPLITS,
                "target_layer": "last convolution before ReLU",
                "map": "absolute value of signed regression Grad-CAM",
                "occlusion_fill": "neutral gray corresponding to normalized zero",
                "roi_radius_fraction": ROI_RADIUS_FRACTION,
                "maximum_prediction_reproduction_difference": maximum_prediction_difference,
                "interpretation": "spatial sensitivity and occlusion diagnostics, not causal proof",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

"""Train and evaluate color-channel and architecture-reduced CNN regressors."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from train_dl_models import (
    IMAGE_SIZE,
    N_SPLITS,
    RANDOM_SEED,
    ROI_RADIUS_FRACTION,
    TargetTransform,
    TinyColorCNN,
    set_reproducible_seed,
)


ANALYTES = ["glucose", "ketone"]
MODEL_IMAGE_SIZE = 48
EXPERIMENT_TARGET_MODES = ["raw"]
ANALYTE_LABELS = {"glucose": "Glucose", "ketone": "Ketone"}
REPRESENTATIONS = ["rgb", "g", "hue_circular"]
REPRESENTATION_LABELS = {
    "rgb": "RGB",
    "g": "G-only",
    "hue_circular": "Circular Hue",
}
INPUT_CHANNELS = {"rgb": 3, "g": 1, "hue_circular": 2}
ARCHITECTURES = ["tiny", "lite"]
ARCHITECTURE_LABELS = {"tiny": "TinyColorCNN", "lite": "ColorLiteCNN"}
EVALUATED_VARIANTS = [
    ("tiny", "rgb"),
    ("tiny", "g"),
    ("tiny", "hue_circular"),
    ("lite", "rgb"),
]
TRAIN_VARIANTS = set(EVALUATED_VARIANTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=Path("outputs/color_features/features.csv")
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("outputs/data_splits/nested_split_assignments.csv"),
    )
    parser.add_argument(
        "--patch-root", type=Path, default=Path("outputs/patch_detection")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/modeling/cnn_lightweight")
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {path}. Use --overwrite to replace it."
            )
        shutil.rmtree(path)
    (path / "figures").mkdir(parents=True, exist_ok=True)
    (path / "checkpoints").mkdir(parents=True, exist_ok=True)


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                input_channels,
                input_channels,
                kernel_size=3,
                padding=1,
                groups=input_channels,
                bias=False,
            ),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class ColorLiteCNN(nn.Module):
    """Narrow depthwise-separable regressor for resource-constrained inference."""

    def __init__(self, input_channels: int, dropout: float = 0.20) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            DepthwiseSeparableBlock(16, 24),
            nn.MaxPool2d(2),
            DepthwiseSeparableBlock(24, 32),
            nn.MaxPool2d(2),
            DepthwiseSeparableBlock(32, 48),
            nn.MaxPool2d(2),
            DepthwiseSeparableBlock(48, 64),
            nn.AdaptiveAvgPool2d(1),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.features(inputs)).squeeze(1)


def build_model(architecture: str, representation: str) -> nn.Module:
    input_channels = INPUT_CHANNELS[representation]
    if architecture == "tiny":
        return TinyColorCNN(input_channels=input_channels)
    if architecture == "lite":
        return ColorLiteCNN(input_channels=input_channels)
    raise ValueError(f"Unknown architecture: {architecture}")


def circular_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2HSV)
    angle = np.deg2rad(hsv[..., 0].astype(np.float32))
    saturation = hsv[..., 1].astype(np.float32)
    value = hsv[..., 2].astype(np.float32)
    hue_sin = saturation * np.sin(angle)
    hue_cos = saturation * np.cos(angle)
    return hue_sin, hue_cos, saturation, value


def transform_representation(rgb: np.ndarray, representation: str) -> np.ndarray:
    if representation == "rgb":
        return (rgb - 0.5) / 0.25
    if representation == "g":
        return ((rgb[..., 1:2] - 0.5) / 0.25).astype(np.float32)
    hue_sin, hue_cos, saturation, value = circular_hsv(rgb)
    if representation == "hue_circular":
        return np.stack([hue_sin, hue_cos], axis=-1).astype(np.float32)
    raise ValueError(f"Unknown representation: {representation}")


def load_image_tensor(
    image_path: Path, circle_radius_px: float, representation: str
) -> torch.Tensor:
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    height, width = rgb.shape[:2]
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    y_grid, x_grid = np.ogrid[:height, :width]
    radius = ROI_RADIUS_FRACTION * float(circle_radius_px)
    mask = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2 <= radius**2
    rgb = rgb.copy()
    rgb[~mask] = 0.5
    array = transform_representation(rgb, representation)
    array[~mask] = 0.0
    tensor = torch.from_numpy(np.transpose(array, (2, 0, 1))).unsqueeze(0)
    return F.interpolate(
        tensor,
        size=(MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def preload_images(
    metadata: pd.DataFrame, patch_root: Path, representation: str
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for row in metadata.itertuples(index=False):
        image_path = patch_root / str(row.crop_file)
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing patch image: {image_path}")
        tensors[str(row.patch_id)] = load_image_tensor(
            image_path, float(row.circle_radius_px), representation
        )
    return tensors


class PatchDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        tensors: dict[str, torch.Tensor],
        targets: np.ndarray,
        augment: bool,
    ) -> None:
        self.patch_ids = rows["patch_id"].astype(str).tolist()
        self.tensors = tensors
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.tensors[self.patch_ids[index]]
        if self.augment:
            image = torch.rot90(image, int(torch.randint(0, 4, (1,)).item()), (1, 2))
            if bool(torch.randint(0, 2, (1,)).item()):
                image = torch.flip(image, (2,))
            if bool(torch.randint(0, 2, (1,)).item()):
                image = torch.flip(image, (1,))
        return image, self.targets[index]


def create_loader(
    rows: pd.DataFrame,
    tensors: dict[str, torch.Tensor],
    target_transform: TargetTransform,
    augment: bool,
    batch_size: int,
    seed: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    targets = target_transform.encode(
        rows["concentration_mg_ml"].to_numpy(dtype=float)
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        PatchDataset(rows, tensors, targets, augment),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
        drop_last=False,
    )


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    target_transform: TargetTransform,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    for images, _ in loader:
        images = images.to(
            device, non_blocking=True, memory_format=torch.channels_last
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            values = model(images)
        predictions.append(values.detach().float().cpu().numpy())
    return target_transform.decode(np.concatenate(predictions))


def experiment_seed(analyte: str, outer_fold: int, target_mode: str) -> int:
    return (
        RANDOM_SEED
        + (0 if analyte == "glucose" else 10_000)
        + outer_fold * 100
        + EXPERIMENT_TARGET_MODES.index(target_mode)
    )


def train_candidate(
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    tensors: dict[str, torch.Tensor],
    architecture: str,
    representation: str,
    target_mode: str,
    device: torch.device,
    seed: int,
    maximum_epochs: int,
    patience: int,
    batch_size: int,
) -> dict[str, Any]:
    set_reproducible_seed(seed)
    transform = TargetTransform.fit(
        train_rows["concentration_mg_ml"].to_numpy(dtype=float), target_mode
    )
    train_loader = create_loader(
        train_rows, tensors, transform, True, batch_size, seed, True, device
    )
    validation_loader = create_loader(
        validation_rows, tensors, transform, False, batch_size, seed, False, device
    )
    model = build_model(architecture, representation).to(
        device=device, memory_format=torch.channels_last
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=6, min_lr=1e-5
    )
    criterion = nn.SmoothL1Loss(beta=0.5)
    validation_actual = validation_rows["concentration_mg_ml"].to_numpy(dtype=float)
    concentration_max = float(train_rows["concentration_mg_ml"].max())
    best_validation_mae = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    epochs_without_improvement = 0
    minimum_epochs = min(20, maximum_epochs)
    started = time.perf_counter()

    for epoch in range(1, maximum_epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for images, targets in train_loader:
            images = images.to(
                device, non_blocking=True, memory_format=torch.channels_last
            )
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model(images)
                loss = criterion(output, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * len(images)
            total_count += len(images)
        validation_raw = predict_loader(model, validation_loader, transform, device)
        validation_prediction = np.clip(validation_raw, 0.0, concentration_max)
        validation_mae = float(
            mean_absolute_error(validation_actual, validation_prediction)
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": total_loss / max(total_count, 1),
                "validation_mae": validation_mae,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        scheduler.step(validation_mae)
        if validation_mae < best_validation_mae - 1e-6:
            best_validation_mae = validation_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch >= minimum_epochs and epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("CNN training did not produce a checkpoint")
    return {
        "state_dict": best_state,
        "target_transform": transform,
        "best_validation_mae": best_validation_mae,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "history": history,
        "training_seconds": time.perf_counter() - started,
        "target_mode": target_mode,
        "seed": seed,
    }


def calculate_metrics(
    actual: np.ndarray,
    prediction_raw: np.ndarray,
    prediction: np.ndarray,
    concentration_range: float,
) -> dict[str, float]:
    mae = float(mean_absolute_error(actual, prediction))
    rmse = float(math.sqrt(mean_squared_error(actual, prediction)))
    rho = (
        np.nan
        if np.unique(actual).size < 2 or np.unique(prediction).size < 2
        else float(spearmanr(actual, prediction).statistic)
    )
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": float(r2_score(actual, prediction)),
        "spearman_rho": rho,
        "normalized_mae": mae / concentration_range,
        "normalized_rmse": rmse / concentration_range,
        "unclipped_mae": float(mean_absolute_error(actual, prediction_raw)),
        "clipped_fraction": float(np.mean(prediction_raw != prediction)),
    }


def count_macs(
    model: nn.Module, input_channels: int, image_size: int = MODEL_IMAGE_SIZE
) -> int:
    total = 0

    def hook(
        module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        nonlocal total
        if isinstance(module, nn.Conv2d):
            kernel_operations = (
                module.kernel_size[0]
                * module.kernel_size[1]
                * module.in_channels
                // module.groups
            )
            total += int(output.numel() * kernel_operations)
        elif isinstance(module, nn.Linear):
            total += int(output.numel() * module.in_features)

    handles = [
        module.register_forward_hook(hook)
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    model.eval()(torch.zeros(1, input_channels, image_size, image_size))
    for handle in handles:
        handle.remove()
    return total


@torch.no_grad()
def benchmark_inference(
    model: nn.Module, input_channels: int, device: torch.device
) -> tuple[float, float]:
    model = model.to(device).eval()
    sample = torch.zeros(
        1, input_channels, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE, device=device
    )
    for _ in range(50):
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    timings: list[float] = []
    for _ in range(200):
        started = time.perf_counter()
        model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append(1000 * (time.perf_counter() - started))
    return float(np.median(timings)), float(np.percentile(timings, 95))


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "mae",
        "rmse",
        "r2",
        "spearman_rho",
        "normalized_mae",
        "normalized_rmse",
        "best_epoch",
        "epochs_run",
        "training_seconds",
        "validation_mae",
    ]
    summary = metrics.groupby(
        ["analyte", "architecture", "representation", "evaluation_unit"],
        as_index=False,
    )[numeric].agg(["mean", "std"])
    summary.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    return summary


def plot_results(
    summary: pd.DataFrame, efficiency: pd.DataFrame, output_path: Path
) -> None:
    patch = summary.loc[summary["evaluation_unit"] == "patch"].copy()
    order = EVALUATED_VARIANTS
    labels = [
        f"{ARCHITECTURE_LABELS[a].replace('Color', '')}\n{REPRESENTATION_LABELS[r]}"
        for a, r in order
    ]
    colors = ["#4C78A8"] * 3 + ["#59A14F"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for axis, analyte in zip(axes[0], ANALYTES):
        indexed = patch.loc[patch["analyte"] == analyte].set_index(
            ["architecture", "representation"]
        )
        values = [100 * indexed.loc[key, "normalized_mae_mean"] for key in order]
        errors = [100 * indexed.loc[key, "normalized_mae_std"] for key in order]
        bars = axis.bar(range(len(order)), values, yerr=errors, capsize=3, color=colors)
        axis.set_xticks(range(len(order)), labels, rotation=25, ha="right")
        axis.set_title(ANALYTE_LABELS[analyte], fontweight="bold")
        axis.set_ylabel("Normalized MAE (% of range)")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    resource_order = order
    resource_index = efficiency.set_index(["architecture", "representation"])
    for axis, metric, title, unit in [
        (axes[1, 0], "parameter_count", "Model parameters", "Parameters"),
        (axes[1, 1], "macs", "Multiply-accumulate operations", "MACs / patch (M)"),
    ]:
        values = [float(resource_index.loc[key, metric]) for key in resource_order]
        if metric == "macs":
            values = [value / 1e6 for value in values]
        bars = axis.bar(range(len(order)), values, color=colors)
        axis.set_xticks(range(len(order)), labels, rotation=25, ha="right")
        axis.set_title(title, fontweight="bold")
        axis.set_ylabel(unit)
        axis.set_yscale("log")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values):
            label = f"{value / 1000:.1f}k" if metric == "parameter_count" else f"{value:.1f}M"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle(
        "48 × 48 CNN color-channel and architecture reduction trade-offs",
        fontsize=15,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.92, hspace=0.48, wspace=0.22)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    prepare_output_dir(args.output_dir, args.overwrite)
    set_reproducible_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = pd.read_csv(args.features)
    split_assignments = pd.read_csv(args.splits)
    metadata = features[["patch_id", "analyte", "crop_file", "circle_radius_px"]].copy()
    split_columns = [
        "outer_fold",
        "patch_id",
        "analyte",
        "concentration_order",
        "concentration_mg_ml",
        "well_id",
        "dl_role",
    ]
    data = split_assignments[split_columns].merge(
        metadata, on=["patch_id", "analyte"], how="left", validate="many_to_one"
    )
    if data.isna().any().any():
        raise ValueError("Lightweight CNN modeling table contains missing values")

    analytes = ["glucose"] if args.smoke_test else ANALYTES
    outer_folds = [1] if args.smoke_test else list(range(1, N_SPLITS + 1))
    architectures = ["tiny"] if args.smoke_test else ARCHITECTURES
    representations = ["g"] if args.smoke_test else REPRESENTATIONS
    maximum_epochs = 3 if args.smoke_test else 100
    patience = 2 if args.smoke_test else 15
    batch_size = 128

    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    concentration_frames: list[pd.DataFrame] = []

    for analyte in analytes:
        analyte_metadata = metadata.loc[metadata["analyte"] == analyte]
        tensors_by_representation = {
            representation: preload_images(
                analyte_metadata, args.patch_root, representation
            )
            for representation in representations
        }
        analyte_max = float(
            data.loc[data["analyte"] == analyte, "concentration_mg_ml"].max()
        )
        for outer_fold in outer_folds:
            fold_data = data.loc[
                (data["analyte"] == analyte) & (data["outer_fold"] == outer_fold)
            ].copy()
            train_rows = fold_data.loc[fold_data["dl_role"] == "train"].reset_index(drop=True)
            validation_rows = fold_data.loc[
                fold_data["dl_role"] == "validation"
            ].reset_index(drop=True)
            test_rows = fold_data.loc[fold_data["dl_role"] == "test"].reset_index(drop=True)
            concentration_range = analyte_max

            for architecture in architectures:
                for representation in representations:
                    if (architecture, representation) not in TRAIN_VARIANTS:
                        continue
                    candidates: list[dict[str, Any]] = []
                    for target_mode in EXPERIMENT_TARGET_MODES:
                        seed = experiment_seed(analyte, outer_fold, target_mode)
                        result = train_candidate(
                            train_rows,
                            validation_rows,
                            tensors_by_representation[representation],
                            architecture,
                            representation,
                            target_mode,
                            device,
                            seed,
                            maximum_epochs,
                            patience,
                            batch_size,
                        )
                        candidates.append(result)
                        selection_rows.append(
                            {
                                "analyte": analyte,
                                "outer_fold": outer_fold,
                                "architecture": architecture,
                                "representation": representation,
                                "target_transform": target_mode,
                                "validation_mae": result["best_validation_mae"],
                                "best_epoch": result["best_epoch"],
                                "epochs_run": result["epochs_run"],
                                "training_seconds": result["training_seconds"],
                            }
                        )
                    selected = min(candidates, key=lambda item: item["best_validation_mae"])
                    for candidate in candidates:
                        selected_target = int(candidate is selected)
                        history = pd.DataFrame(candidate["history"])
                        history.insert(0, "analyte", analyte)
                        history.insert(1, "outer_fold", outer_fold)
                        history.insert(2, "architecture", architecture)
                        history.insert(3, "representation", representation)
                        history.insert(4, "target_transform", candidate["target_mode"])
                        history.insert(5, "selected_target", selected_target)
                        history_frames.append(history)

                    model = build_model(architecture, representation).to(
                        device=device, memory_format=torch.channels_last
                    )
                    model.load_state_dict(selected["state_dict"])
                    transform: TargetTransform = selected["target_transform"]
                    test_loader = create_loader(
                        test_rows,
                        tensors_by_representation[representation],
                        transform,
                        False,
                        batch_size,
                        int(selected["seed"]),
                        False,
                        device,
                    )
                    prediction_raw = predict_loader(model, test_loader, transform, device)
                    prediction = np.clip(prediction_raw, 0.0, analyte_max)
                    frame = test_rows[
                        [
                            "patch_id",
                            "well_id",
                            "concentration_order",
                            "concentration_mg_ml",
                        ]
                    ].rename(columns={"concentration_mg_ml": "actual_concentration"})
                    frame.insert(0, "analyte", analyte)
                    frame.insert(1, "architecture", architecture)
                    frame.insert(2, "representation", representation)
                    frame.insert(3, "model", ARCHITECTURE_LABELS[architecture])
                    frame.insert(4, "outer_fold", outer_fold)
                    frame.insert(5, "target_transform", selected["target_mode"])
                    frame["prediction_raw"] = prediction_raw
                    frame["prediction"] = prediction
                    prediction_frames.append(frame)

                    checkpoint_path = (
                        args.output_dir
                        / "checkpoints"
                        / f"{analyte}_{architecture}_{representation}_fold{outer_fold}.pt"
                    )
                    torch.save(
                        {
                            "state_dict": selected["state_dict"],
                            "analyte": analyte,
                            "architecture": architecture,
                            "representation": representation,
                            "outer_fold": outer_fold,
                            "target_transform": selected["target_mode"],
                            "target_mean": transform.mean,
                            "target_std": transform.std,
                            "image_size": MODEL_IMAGE_SIZE,
                            "roi_radius_fraction": ROI_RADIUS_FRACTION,
                        },
                        checkpoint_path,
                    )

                    common = {
                        "analyte": analyte,
                        "architecture": architecture,
                        "representation": representation,
                        "model": ARCHITECTURE_LABELS[architecture],
                        "outer_fold": outer_fold,
                        "target_transform": selected["target_mode"],
                        "best_epoch": int(selected["best_epoch"]),
                        "epochs_run": int(selected["epochs_run"]),
                        "training_seconds": float(selected["training_seconds"]),
                        "validation_mae": float(selected["best_validation_mae"]),
                        "parameter_count": sum(p.numel() for p in model.parameters()),
                        "model_size_bytes": checkpoint_path.stat().st_size,
                    }
                    patch_metrics = calculate_metrics(
                        frame["actual_concentration"].to_numpy(),
                        frame["prediction_raw"].to_numpy(),
                        frame["prediction"].to_numpy(),
                        concentration_range,
                    )
                    patch_metrics.update(common)
                    patch_metrics.update(
                        {"evaluation_unit": "patch", "n_observations": len(frame)}
                    )

                    concentration = (
                        frame.groupby(
                            ["concentration_order", "actual_concentration"],
                            as_index=False,
                        )
                        .agg(
                            prediction_raw=("prediction_raw", "median"),
                            prediction=("prediction", "median"),
                            patch_count=("patch_id", "size"),
                        )
                        .sort_values("concentration_order")
                    )
                    concentration_metrics = calculate_metrics(
                        concentration["actual_concentration"].to_numpy(),
                        concentration["prediction_raw"].to_numpy(),
                        concentration["prediction"].to_numpy(),
                        concentration_range,
                    )
                    concentration_metrics.update(common)
                    concentration_metrics.update(
                        {
                            "evaluation_unit": "concentration_median",
                            "n_observations": len(concentration),
                        }
                    )
                    metric_frames.append(pd.DataFrame([patch_metrics, concentration_metrics]))
                    concentration.insert(0, "analyte", analyte)
                    concentration.insert(1, "architecture", architecture)
                    concentration.insert(2, "representation", representation)
                    concentration.insert(3, "outer_fold", outer_fold)
                    concentration_frames.append(concentration)
                    print(
                        f"{analyte} fold={outer_fold} {architecture}/{representation}: "
                        f"target={selected['target_mode']} val={selected['best_validation_mae']:.4f} "
                        f"epoch={selected['best_epoch']}",
                        flush=True,
                    )

    metrics = pd.concat(metric_frames, ignore_index=True, sort=False)
    predictions = pd.concat(prediction_frames, ignore_index=True, sort=False)
    history = pd.concat(history_frames, ignore_index=True, sort=False)
    selection = pd.DataFrame(selection_rows)
    concentrations = pd.concat(concentration_frames, ignore_index=True, sort=False)
    summary = summarize_metrics(metrics)

    efficiency_rows: list[dict[str, Any]] = []
    for architecture, representation in EVALUATED_VARIANTS:
        model = build_model(architecture, representation).eval()
        parameters = sum(parameter.numel() for parameter in model.parameters())
        input_channels = INPUT_CHANNELS[representation]
        median_ms, p95_ms = benchmark_inference(model, input_channels, device)
        efficiency_rows.append(
            {
                "architecture": architecture,
                "representation": representation,
                "input_channels": input_channels,
                "input_tensor_bytes": 4
                * input_channels
                * MODEL_IMAGE_SIZE
                * MODEL_IMAGE_SIZE,
                "parameter_count": parameters,
                "parameter_bytes_float32": 4 * parameters,
                "macs": count_macs(model.cpu(), input_channels),
                "gpu_batch1_latency_median_ms": median_ms,
                "gpu_batch1_latency_p95_ms": p95_ms,
            }
        )
    efficiency = pd.DataFrame(efficiency_rows)
    baseline_efficiency = efficiency.loc[
        (efficiency["architecture"] == "tiny")
        & (efficiency["representation"] == "rgb")
    ].iloc[0]
    for column in ["input_tensor_bytes", "parameter_count", "macs"]:
        efficiency[f"{column}_reduction_percent_vs_tiny_rgb"] = 100 * (
            1 - efficiency[column] / float(baseline_efficiency[column])
        )
    original_model = TinyColorCNN(input_channels=3).eval()
    original_reference = {
        "input_tensor_bytes": 4 * 3 * IMAGE_SIZE * IMAGE_SIZE,
        "parameter_count": sum(p.numel() for p in original_model.parameters()),
        "macs": count_macs(original_model, 3, IMAGE_SIZE),
    }
    for column, reference in original_reference.items():
        efficiency[f"{column}_reduction_percent_vs_original_96_rgb"] = 100 * (
            1 - efficiency[column] / float(reference)
        )

    metrics.to_csv(args.output_dir / "cnn_lightweight_fold_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "cnn_lightweight_performance_summary.csv", index=False)
    predictions.to_csv(args.output_dir / "cnn_lightweight_predictions.csv", index=False)
    concentrations.to_csv(
        args.output_dir / "cnn_lightweight_concentration_predictions.csv", index=False
    )
    history.to_csv(args.output_dir / "cnn_lightweight_training_history.csv", index=False)
    selection.to_csv(args.output_dir / "cnn_lightweight_target_selection.csv", index=False)
    efficiency.to_csv(args.output_dir / "cnn_lightweight_efficiency.csv", index=False)
    if not args.smoke_test:
        plot_results(
            summary,
            efficiency,
            args.output_dir / "figures" / "cnn_lightweight_tradeoffs.png",
        )

    config = {
        "random_seed": RANDOM_SEED,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "outer_folds": outer_folds,
        "architectures": ARCHITECTURES,
        "representations": REPRESENTATIONS,
        "input_channels": INPUT_CHANNELS,
        "original_reference_image_size": IMAGE_SIZE,
        "lightweight_experiment_image_size": MODEL_IMAGE_SIZE,
        "roi_radius_fraction": ROI_RADIUS_FRACTION,
        "target_transform_candidates": EXPERIMENT_TARGET_MODES,
        "maximum_epochs": maximum_epochs,
        "early_stopping_patience": patience,
        "batch_size": batch_size,
        "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)",
        "loss": "SmoothL1Loss on train-standardized target",
        "precision": "CUDA automatic mixed precision (float16); float32 on CPU",
        "memory_format": "channels_last",
        "augmentation": "random 90-degree rotations and horizontal/vertical flips only",
        "hue_encoding": "S*sin(H), S*cos(H)",
        "lite_architecture": "narrow depthwise-separable convolutions",
        "evaluated_variants": EVALUATED_VARIANTS,
        "smoke_test": args.smoke_test,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved lightweight CNN results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

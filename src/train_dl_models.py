"""Train lightweight image regressors with the fixed group-aware splits."""

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


RANDOM_SEED = 240920
N_SPLITS = 5
IMAGE_SIZE = 96
ROI_RADIUS_FRACTION = 0.70
INPUT_MODES = ["roi_masked", "full_patch"]
TARGET_MODES = ["raw", "log1p"]
ANALYTE_LABELS = {"glucose": "Glucose", "ketone": "Ketone"}
MODE_LABELS = {"roi_masked": "Central ROI", "full_patch": "Full patch"}


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
        "--output-dir",
        type=Path,
        default=Path("outputs/modeling/dl"),
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


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def candidate_seed(analyte: str, outer_fold: int, input_mode: str, target_mode: str) -> int:
    return (
        RANDOM_SEED
        + (0 if analyte == "glucose" else 10_000)
        + outer_fold * 100
        + INPUT_MODES.index(input_mode) * 10
        + TARGET_MODES.index(target_mode)
    )


def load_image_tensor(
    image_path: Path,
    circle_radius_px: float,
    input_mode: str,
) -> torch.Tensor:
    with Image.open(image_path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    height, width = array.shape[:2]
    if input_mode == "roi_masked":
        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0
        y_grid, x_grid = np.ogrid[:height, :width]
        radius = ROI_RADIUS_FRACTION * float(circle_radius_px)
        mask = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2 <= radius**2
        array[~mask] = 0.5
    tensor = torch.from_numpy(np.transpose(array, (2, 0, 1))).unsqueeze(0)
    tensor = F.interpolate(
        tensor,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return (tensor - 0.5) / 0.25


def preload_images(
    metadata: pd.DataFrame,
    patch_root: Path,
    input_mode: str,
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for row in metadata.itertuples(index=False):
        image_path = patch_root / str(row.crop_file)
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing patch image: {image_path}")
        tensors[str(row.patch_id)] = load_image_tensor(
            image_path,
            float(row.circle_radius_px),
            input_mode,
        )
    return tensors


class PatchDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        tensors: dict[str, torch.Tensor],
        target_values: np.ndarray,
        augment: bool,
    ) -> None:
        self.patch_ids = rows["patch_id"].astype(str).tolist()
        self.tensors = tensors
        self.targets = torch.as_tensor(target_values, dtype=torch.float32)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.tensors[self.patch_ids[index]]
        if self.augment:
            rotation = int(torch.randint(0, 4, (1,)).item())
            image = torch.rot90(image, rotation, dims=(1, 2))
            if bool(torch.randint(0, 2, (1,)).item()):
                image = torch.flip(image, dims=(2,))
            if bool(torch.randint(0, 2, (1,)).item()):
                image = torch.flip(image, dims=(1,))
        return image, self.targets[index]


class TinyColorCNN(nn.Module):
    def __init__(self, dropout: float = 0.25, input_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.features(inputs)).squeeze(1)


@dataclass
class TargetTransform:
    mode: str
    mean: float
    std: float

    @classmethod
    def fit(cls, values: np.ndarray, mode: str) -> "TargetTransform":
        transformed = np.log1p(values) if mode == "log1p" else values.astype(float)
        standard_deviation = float(np.std(transformed))
        if standard_deviation <= 0:
            raise ValueError("Training target has zero variance")
        return cls(mode=mode, mean=float(np.mean(transformed)), std=standard_deviation)

    def encode(self, values: np.ndarray) -> np.ndarray:
        transformed = np.log1p(values) if self.mode == "log1p" else values.astype(float)
        return ((transformed - self.mean) / self.std).astype(np.float32)

    def decode(self, values: np.ndarray) -> np.ndarray:
        transformed = values.astype(float) * self.std + self.mean
        return np.expm1(transformed) if self.mode == "log1p" else transformed


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
    target = target_transform.encode(rows["concentration_mg_ml"].to_numpy(dtype=float))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        PatchDataset(rows, tensors, target, augment),
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
        images = images.to(device, non_blocking=True)
        values = model(images).detach().cpu().numpy()
        predictions.append(values)
    standardized = np.concatenate(predictions)
    return target_transform.decode(standardized)


def train_candidate(
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    tensors: dict[str, torch.Tensor],
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
        train_rows,
        tensors,
        transform,
        augment=True,
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
        device=device,
    )
    validation_loader = create_loader(
        validation_rows,
        tensors,
        transform,
        augment=False,
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
        device=device,
    )
    model = TinyColorCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
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

    start_time = time.perf_counter()
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(images)
            total_count += len(images)

        validation_raw = predict_loader(model, validation_loader, transform, device)
        validation_prediction = np.clip(validation_raw, 0.0, concentration_max)
        validation_mae = float(mean_absolute_error(validation_actual, validation_prediction))
        training_loss = total_loss / max(total_count, 1)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "validation_mae": validation_mae,
                "learning_rate": current_lr,
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
    training_seconds = time.perf_counter() - start_time
    return {
        "state_dict": best_state,
        "target_transform": transform,
        "best_validation_mae": best_validation_mae,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "history": history,
        "training_seconds": training_seconds,
    }


def calculate_metrics(
    actual: np.ndarray,
    prediction_raw: np.ndarray,
    prediction: np.ndarray,
    concentration_range: float,
) -> dict[str, float]:
    mae = mean_absolute_error(actual, prediction)
    rmse = math.sqrt(mean_squared_error(actual, prediction))
    rho = (
        np.nan
        if np.unique(actual).size < 2 or np.unique(prediction).size < 2
        else spearmanr(actual, prediction).statistic
    )
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2_score(actual, prediction)),
        "spearman_rho": float(rho),
        "normalized_mae": float(mae / concentration_range),
        "normalized_rmse": float(rmse / concentration_range),
        "mean_absolute_log1p_error": float(
            mean_absolute_error(np.log1p(actual), np.log1p(prediction))
        ),
        "unclipped_mae": float(mean_absolute_error(actual, prediction_raw)),
        "clipped_fraction": float(np.mean(prediction_raw != prediction)),
    }


def evaluation_rows(
    prediction_frame: pd.DataFrame,
    analyte: str,
    input_mode: str,
    outer_fold: int,
    target_mode: str,
    best_epoch: int,
    epochs_run: int,
    training_seconds: float,
    inference_seconds: float,
    parameter_count: int,
    model_size_bytes: int,
    validation_mae: float,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    concentration_range = float(
        prediction_frame["actual_concentration"].max()
        - prediction_frame["actual_concentration"].min()
    )
    common = {
        "analyte": analyte,
        "input_mode": input_mode,
        "model": "TinyColorCNN",
        "outer_fold": outer_fold,
        "target_transform": target_mode,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "inference_ms_per_patch": 1000 * inference_seconds / len(prediction_frame),
        "parameter_count": parameter_count,
        "model_size_bytes": model_size_bytes,
        "validation_mae": validation_mae,
    }
    patch_metrics = calculate_metrics(
        prediction_frame["actual_concentration"].to_numpy(),
        prediction_frame["prediction_raw"].to_numpy(),
        prediction_frame["prediction"].to_numpy(),
        concentration_range,
    )
    patch_metrics.update(common)
    patch_metrics.update(
        {"evaluation_unit": "patch", "n_observations": len(prediction_frame)}
    )

    concentration = (
        prediction_frame.groupby(
            ["concentration_order", "actual_concentration"], as_index=False
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
    concentration.insert(0, "analyte", analyte)
    concentration.insert(1, "input_mode", input_mode)
    concentration.insert(2, "model", "TinyColorCNN")
    concentration.insert(3, "outer_fold", outer_fold)
    concentration.insert(4, "target_transform", target_mode)
    return [patch_metrics, concentration_metrics], concentration


def plot_training_curves(history: pd.DataFrame, output_path: Path) -> None:
    selected = history.loc[
        (history["selected_target"] == 1) & (history["input_mode"] == "roi_masked")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=False)
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, N_SPLITS))
    for axis, analyte in zip(axes, ["glucose", "ketone"]):
        subset = selected.loc[selected["analyte"] == analyte]
        for color, fold in zip(colors, range(1, N_SPLITS + 1)):
            fold_data = subset.loc[subset["outer_fold"] == fold]
            axis.plot(
                fold_data["epoch"],
                fold_data["validation_mae"],
                color=color,
                linewidth=1.6,
                label=f"Fold {fold}",
            )
        axis.set_title(ANALYTE_LABELS[analyte], fontweight="bold")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Validation MAE (mg/mL)")
        axis.grid(color="#D9D9D9", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle(
        "TinyColorCNN validation curves using the central ROI",
        fontsize=15,
        fontweight="bold",
    )
    fig.subplots_adjust(bottom=0.18, top=0.85, wspace=0.25)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_input_comparison(summary: pd.DataFrame, output_path: Path) -> None:
    selected = summary.loc[summary["evaluation_unit"] == "patch"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    colors = ["#4C78A8", "#F2CF5B"]
    for axis, analyte in zip(axes, ["glucose", "ketone"]):
        subset = selected.loc[selected["analyte"] == analyte].set_index("input_mode")
        values = [100 * subset.loc[mode, "normalized_mae_mean"] for mode in INPUT_MODES]
        errors = [100 * subset.loc[mode, "normalized_mae_std"] for mode in INPUT_MODES]
        bars = axis.bar(
            range(len(INPUT_MODES)),
            values,
            yerr=errors,
            capsize=4,
            color=colors,
            width=0.62,
        )
        axis.set_xticks(range(len(INPUT_MODES)), [MODE_LABELS[mode] for mode in INPUT_MODES])
        axis.set_title(ANALYTE_LABELS[analyte], fontweight="bold")
        axis.set_xlabel("CNN input")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    axes[0].set_ylabel("Outer-fold normalized MAE (% of range, mean ± SD)")
    fig.suptitle("Effect of restricting CNN input to the sensor ROI", fontsize=15, fontweight="bold")
    fig.subplots_adjust(top=0.84, wspace=0.12)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    prepare_output_dir(args.output_dir, args.overwrite)
    set_reproducible_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = pd.read_csv(args.features)
    split_assignments = pd.read_csv(args.splits)
    metadata = features[
        ["patch_id", "analyte", "crop_file", "circle_radius_px"]
    ].copy()
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
        metadata,
        on=["patch_id", "analyte"],
        how="left",
        validate="many_to_one",
    )
    if data.isna().any().any():
        raise ValueError("DL modeling table contains missing values")

    analytes = ["glucose"] if args.smoke_test else ["glucose", "ketone"]
    outer_folds = [1] if args.smoke_test else list(range(1, N_SPLITS + 1))
    input_modes = ["roi_masked"] if args.smoke_test else INPUT_MODES
    maximum_epochs = 3 if args.smoke_test else 120
    patience = 2 if args.smoke_test else 18
    batch_size = 64

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    concentration_rows: list[pd.DataFrame] = []
    history_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    for analyte in analytes:
        analyte_metadata = metadata.loc[metadata["analyte"] == analyte]
        image_tensors = {
            mode: preload_images(analyte_metadata, args.patch_root, mode)
            for mode in input_modes
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

            for input_mode in input_modes:
                candidates: list[dict[str, Any]] = []
                for target_mode in TARGET_MODES:
                    seed = candidate_seed(analyte, outer_fold, input_mode, target_mode)
                    result = train_candidate(
                        train_rows,
                        validation_rows,
                        image_tensors[input_mode],
                        target_mode,
                        device,
                        seed,
                        maximum_epochs,
                        patience,
                        batch_size,
                    )
                    result["target_mode"] = target_mode
                    result["seed"] = seed
                    candidates.append(result)
                    selection_rows.append(
                        {
                            "analyte": analyte,
                            "outer_fold": outer_fold,
                            "input_mode": input_mode,
                            "target_transform": target_mode,
                            "validation_mae": result["best_validation_mae"],
                            "best_epoch": result["best_epoch"],
                            "epochs_run": result["epochs_run"],
                            "training_seconds": result["training_seconds"],
                        }
                    )
                selected = min(candidates, key=lambda item: item["best_validation_mae"])
                selected_mode = str(selected["target_mode"])
                for candidate in candidates:
                    chosen = int(candidate is selected)
                    for history in candidate["history"]:
                        history_rows.append(
                            {
                                "analyte": analyte,
                                "outer_fold": outer_fold,
                                "input_mode": input_mode,
                                "target_transform": candidate["target_mode"],
                                "selected_target": chosen,
                                **history,
                            }
                        )

                model = TinyColorCNN().to(device)
                model.load_state_dict(selected["state_dict"])
                transform: TargetTransform = selected["target_transform"]
                test_loader = create_loader(
                    test_rows,
                    image_tensors[input_mode],
                    transform,
                    augment=False,
                    batch_size=batch_size,
                    seed=int(selected["seed"]),
                    shuffle=False,
                    device=device,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                inference_start = time.perf_counter()
                prediction_raw = predict_loader(model, test_loader, transform, device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                inference_seconds = time.perf_counter() - inference_start
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
                frame.insert(1, "input_mode", input_mode)
                frame.insert(2, "model", "TinyColorCNN")
                frame.insert(3, "outer_fold", outer_fold)
                frame.insert(4, "target_transform", selected_mode)
                frame["prediction_raw"] = prediction_raw
                frame["prediction"] = prediction

                checkpoint_path = (
                    args.output_dir
                    / "checkpoints"
                    / f"{analyte}_{input_mode}_fold{outer_fold}.pt"
                )
                checkpoint = {
                    "state_dict": selected["state_dict"],
                    "analyte": analyte,
                    "input_mode": input_mode,
                    "outer_fold": outer_fold,
                    "target_transform": selected_mode,
                    "target_mean": transform.mean,
                    "target_std": transform.std,
                    "image_size": IMAGE_SIZE,
                    "roi_radius_fraction": ROI_RADIUS_FRACTION,
                }
                torch.save(checkpoint, checkpoint_path)
                model_size_bytes = checkpoint_path.stat().st_size
                parameter_count = sum(parameter.numel() for parameter in model.parameters())
                metrics, concentrations = evaluation_rows(
                    frame,
                    analyte,
                    input_mode,
                    outer_fold,
                    selected_mode,
                    int(selected["best_epoch"]),
                    int(selected["epochs_run"]),
                    float(selected["training_seconds"]),
                    inference_seconds,
                    parameter_count,
                    model_size_bytes,
                    float(selected["best_validation_mae"]),
                )
                metric_rows.extend(metrics)
                prediction_rows.append(frame)
                concentration_rows.append(concentrations)
                print(
                    f"{analyte} fold={outer_fold} input={input_mode}: "
                    f"target={selected_mode}, val MAE={selected['best_validation_mae']:.4f}, "
                    f"epoch={selected['best_epoch']}",
                    flush=True,
                )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    concentration_predictions = pd.concat(concentration_rows, ignore_index=True)
    history = pd.DataFrame(history_rows)
    selection = pd.DataFrame(selection_rows)
    metric_columns = [
        "mae",
        "rmse",
        "r2",
        "spearman_rho",
        "normalized_mae",
        "normalized_rmse",
        "mean_absolute_log1p_error",
        "unclipped_mae",
        "clipped_fraction",
        "best_epoch",
        "epochs_run",
        "training_seconds",
        "inference_seconds",
        "inference_ms_per_patch",
        "parameter_count",
        "model_size_bytes",
        "validation_mae",
    ]
    summary = (
        metrics.groupby(
            ["analyte", "input_mode", "model", "evaluation_unit"], as_index=False
        )[metric_columns]
        .agg(["mean", "std"])
    )
    summary.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]

    metrics.to_csv(args.output_dir / "dl_fold_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "dl_performance_summary.csv", index=False)
    predictions.to_csv(args.output_dir / "dl_predictions.csv", index=False)
    concentration_predictions.to_csv(
        args.output_dir / "dl_concentration_predictions.csv", index=False
    )
    history.to_csv(args.output_dir / "training_history.csv", index=False)
    selection.to_csv(args.output_dir / "target_transform_selection.csv", index=False)
    if not args.smoke_test:
        plot_training_curves(
            history,
            args.output_dir / "figures" / "dl_training_curves.png",
        )
        plot_input_comparison(
            summary,
            args.output_dir / "figures" / "dl_input_comparison.png",
        )
    config = {
        "random_seed": RANDOM_SEED,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "outer_folds": outer_folds,
        "input_modes": input_modes,
        "target_transform_candidates": TARGET_MODES,
        "image_size": IMAGE_SIZE,
        "roi_radius_fraction": ROI_RADIUS_FRACTION,
        "augmentation": "random 90-degree rotations and horizontal/vertical flips only",
        "color_augmentation": False,
        "maximum_epochs": maximum_epochs,
        "early_stopping_patience": patience,
        "batch_size": batch_size,
        "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)",
        "loss": "SmoothL1Loss on train-standardized target",
        "reported_prediction": "clipped to 0 and analyte training maximum",
        "evaluation_units": ["patch", "concentration_median"],
        "smoke_test": args.smoke_test,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved DL results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

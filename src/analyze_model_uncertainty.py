"""Quantify paired model differences and color-normalization sensitivity."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RANDOM_SEED = 240920
BOOTSTRAP_REPEATS = 20_000
ANALYTE_RANGES = {"glucose": 20.0, "ketone": 10.0}
ANALYTE_LABELS = {"glucose": "Glucose", "ketone": "Ketone"}
COMPARISONS = ["CNN central ROI", "CNN full patch"]
FEATURE_SETS = [
    "RGB_primary",
    "Chromaticity_secondary",
    "Background_adjusted_secondary",
    "HSV_primary",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/modeling/comparison/headline_model_predictions.csv"),
    )
    parser.add_argument(
        "--ml-metrics",
        type=Path,
        default=Path("outputs/modeling/ml/ml_fold_metrics.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/modeling/uncertainty"),
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


def paired_bootstrap(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(RANDOM_SEED)
    for analyte, concentration_range in ANALYTE_RANGES.items():
        analyte_data = predictions.loc[predictions["analyte"] == analyte].copy()
        analyte_data["absolute_error"] = np.abs(
            analyte_data["prediction"] - analyte_data["actual_concentration"]
        )
        well_errors = (
            analyte_data.groupby(["well_id", "comparison_model"])["absolute_error"]
            .mean()
            .unstack("comparison_model")
            .sort_index()
        )
        if set(well_errors.columns) != {
            "Nested-selected ML",
            "CNN central ROI",
            "CNN full patch",
        }:
            raise ValueError(f"Unexpected model set for {analyte}: {well_errors.columns}")
        if len(well_errors) != 96 or well_errors.isna().any().any():
            raise ValueError(f"Expected 96 complete well clusters for {analyte}")
        ml_error = well_errors["Nested-selected ML"].to_numpy(dtype=float)
        sampled_indices = rng.integers(
            0, len(well_errors), size=(BOOTSTRAP_REPEATS, len(well_errors))
        )
        ml_bootstrap = ml_error[sampled_indices].mean(axis=1)
        for comparison in COMPARISONS:
            cnn_error = well_errors[comparison].to_numpy(dtype=float)
            point_difference = float(cnn_error.mean() - ml_error.mean())
            bootstrap_difference = (
                cnn_error[sampled_indices].mean(axis=1) - ml_bootstrap
            )
            lower, upper = np.quantile(bootstrap_difference, [0.025, 0.975])
            rows.append(
                {
                    "analyte": analyte,
                    "comparison": comparison,
                    "cluster_unit": "well_id",
                    "clusters": len(well_errors),
                    "bootstrap_repeats": BOOTSTRAP_REPEATS,
                    "mae_difference_cnn_minus_ml": point_difference,
                    "ci95_lower": float(lower),
                    "ci95_upper": float(upper),
                    "normalized_difference_percent": 100
                    * point_difference
                    / concentration_range,
                    "normalized_ci95_lower_percent": 100
                    * float(lower)
                    / concentration_range,
                    "normalized_ci95_upper_percent": 100
                    * float(upper)
                    / concentration_range,
                    "bootstrap_fraction_positive": float(
                        np.mean(bootstrap_difference > 0)
                    ),
                }
            )
    return pd.DataFrame(rows)


def normalization_sensitivity(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics.loc[
        (metrics["evaluation_unit"] == "patch")
        & (metrics["model"] == "ExtraTrees")
        & (metrics["feature_set"].isin(FEATURE_SETS))
    ].copy()
    summary = (
        selected.groupby(["analyte", "feature_set", "model"], as_index=False)
        .agg(
            folds=("outer_fold", "nunique"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            r2_mean=("r2", "mean"),
        )
    )
    rgb_reference = summary.loc[
        summary["feature_set"] == "RGB_primary", ["analyte", "mae_mean"]
    ].rename(columns={"mae_mean": "rgb_mae_reference"})
    summary = summary.merge(rgb_reference, on="analyte", validate="many_to_one")
    summary["mae_change_vs_rgb"] = summary["mae_mean"] - summary["rgb_mae_reference"]
    summary["normalized_mae_percent"] = summary.apply(
        lambda row: 100 * row["mae_mean"] / ANALYTE_RANGES[str(row["analyte"])],
        axis=1,
    )
    return summary.sort_values(["analyte", "mae_mean"]).reset_index(drop=True)


def plot_bootstrap(summary: pd.DataFrame, output_path: Path) -> None:
    order = [
        ("glucose", "CNN central ROI"),
        ("glucose", "CNN full patch"),
        ("ketone", "CNN central ROI"),
        ("ketone", "CNN full patch"),
    ]
    labels = [
        f"{ANALYTE_LABELS[analyte]} · {comparison.replace('CNN ', '')}"
        for analyte, comparison in order
    ]
    points: list[float] = []
    lower_errors: list[float] = []
    upper_errors: list[float] = []
    for analyte, comparison in order:
        row = summary.loc[
            (summary["analyte"] == analyte)
            & (summary["comparison"] == comparison)
        ].iloc[0]
        point = float(row["normalized_difference_percent"])
        lower = float(row["normalized_ci95_lower_percent"])
        upper = float(row["normalized_ci95_upper_percent"])
        points.append(point)
        lower_errors.append(point - lower)
        upper_errors.append(upper - point)
    y = np.arange(len(order))
    colors = ["#477db3", "#e0a52e", "#477db3", "#e0a52e"]
    fig, axis = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    axis.axvline(0, color="#333333", linewidth=1.2)
    axis.errorbar(
        points,
        y,
        xerr=np.array([lower_errors, upper_errors]),
        fmt="none",
        ecolor="#333333",
        capsize=5,
        linewidth=1.8,
    )
    axis.scatter(points, y, c=colors, s=85, zorder=3)
    for x, y_value in zip(points, y, strict=True):
        axis.text(x + 0.025, y_value + 0.15, f"{x:.2f}%p", fontsize=10)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("CNN MAE − nested-selected ML MAE (% of concentration range)")
    fig.suptitle(
        "Paired well-position bootstrap of MAE differences",
        fontsize=17,
        fontweight="bold",
    )
    axis.set_title(
        "Positive values favor ML · 20,000 resamples of 96 well_id clusters",
        fontsize=10,
        color="#555555",
        pad=8,
    )
    axis.grid(axis="x", alpha=0.25)
    axis.margins(y=0.15)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    prepare_output_dir(args.output_dir, args.overwrite)
    predictions = pd.read_csv(args.predictions)
    metrics = pd.read_csv(args.ml_metrics)
    bootstrap = paired_bootstrap(predictions)
    sensitivity = normalization_sensitivity(metrics)
    bootstrap.to_csv(args.output_dir / "paired_mae_bootstrap.csv", index=False)
    sensitivity.to_csv(
        args.output_dir / "color_normalization_sensitivity.csv", index=False
    )
    plot_bootstrap(
        bootstrap, args.output_dir / "figures" / "paired_mae_bootstrap.png"
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "random_seed": RANDOM_SEED,
                "bootstrap_repeats": BOOTSTRAP_REPEATS,
                "cluster_unit": "well_id",
                "paired_models": ["Nested-selected ML", *COMPARISONS],
                "uncertainty_scope": (
                    "conditional uncertainty across the 96 well positions; "
                    "not uncertainty across independent images or plates"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

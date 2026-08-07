"""Compare nested-selected ML and lightweight CNN models on fixed outer folds."""

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


PRIMARY_FEATURE_SETS = {"RGB_primary", "HSV_primary", "RGB_HSV_combined"}
MODEL_ORDER = [
    "Nested-selected ML",
    "CNN central ROI",
    "CNN full patch",
    "Background only",
    "Mean baseline",
]
HEADLINE_ORDER = ["Nested-selected ML", "CNN central ROI", "CNN full patch"]
MODEL_COLORS = {
    "Mean baseline": "#A0A0A0",
    "Background only": "#9C755F",
    "Nested-selected ML": "#4C78A8",
    "CNN central ROI": "#59A14F",
    "CNN full patch": "#F2CF5B",
}
ANALYTE_LABELS = {"glucose": "Glucose", "ketone": "Ketone"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ml-dir", type=Path, default=Path("outputs/modeling/ml")
    )
    parser.add_argument(
        "--dl-dir", type=Path, default=Path("outputs/modeling/dl")
    )
    parser.add_argument(
        "--selected-ml-dir",
        type=Path,
        default=Path("outputs/modeling/selected_ml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/modeling/comparison"),
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


def select_ml_pipeline(ml_dir: Path) -> pd.DataFrame:
    parameters = pd.read_csv(ml_dir / "ml_best_parameters.csv")
    return (
        parameters.loc[parameters["feature_set"].isin(PRIMARY_FEATURE_SETS)]
        .sort_values(["analyte", "outer_fold", "inner_best_mae"])
        .groupby(["analyte", "outer_fold"], as_index=False)
        .first()
    )


def selected_ml_rows(table: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    keys = selected[["analyte", "outer_fold", "feature_set", "model"]]
    return keys.merge(
        table,
        on=["analyte", "outer_fold", "feature_set", "model"],
        how="left",
        validate="one_to_many",
    )


def add_comparison_label(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    output = frame.copy()
    output["comparison_model"] = label
    return output


def build_comparison_metrics(
    ml_metrics: pd.DataFrame,
    dl_metrics: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    selected_ml = add_comparison_label(
        selected_ml_rows(ml_metrics, selected), "Nested-selected ML"
    )
    baseline = add_comparison_label(
        ml_metrics.loc[
            (ml_metrics["feature_set"] == "Baseline")
            & (ml_metrics["model"] == "DummyMean")
        ],
        "Mean baseline",
    )
    background = add_comparison_label(
        ml_metrics.loc[
            (ml_metrics["feature_set"] == "Background_negative_control")
            & (ml_metrics["model"] == "ExtraTrees")
        ],
        "Background only",
    )
    cnn_roi = add_comparison_label(
        dl_metrics.loc[dl_metrics["input_mode"] == "roi_masked"],
        "CNN central ROI",
    )
    cnn_full = add_comparison_label(
        dl_metrics.loc[dl_metrics["input_mode"] == "full_patch"],
        "CNN full patch",
    )
    keep_columns = [
        "analyte",
        "outer_fold",
        "evaluation_unit",
        "comparison_model",
        "mae",
        "rmse",
        "r2",
        "spearman_rho",
        "normalized_mae",
        "normalized_rmse",
        "mean_absolute_log1p_error",
        "unclipped_mae",
        "clipped_fraction",
        "n_observations",
    ]
    return pd.concat(
        [baseline, background, selected_ml, cnn_roi, cnn_full],
        ignore_index=True,
    )[keep_columns]


def summarize_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
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
    ]
    summary = (
        metrics.groupby(
            ["analyte", "comparison_model", "evaluation_unit"], as_index=False
        )[metric_columns]
        .agg(["mean", "std"])
    )
    summary.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    return summary


def build_headline_predictions(
    ml_predictions: pd.DataFrame,
    dl_predictions: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    selected_ml = add_comparison_label(
        selected_ml_rows(ml_predictions, selected), "Nested-selected ML"
    )
    cnn_roi = add_comparison_label(
        dl_predictions.loc[dl_predictions["input_mode"] == "roi_masked"],
        "CNN central ROI",
    )
    cnn_full = add_comparison_label(
        dl_predictions.loc[dl_predictions["input_mode"] == "full_patch"],
        "CNN full patch",
    )
    keep = [
        "analyte",
        "outer_fold",
        "comparison_model",
        "patch_id",
        "well_id",
        "concentration_order",
        "actual_concentration",
        "prediction_raw",
        "prediction",
    ]
    return pd.concat([selected_ml, cnn_roi, cnn_full], ignore_index=True)[keep]


def build_headline_concentrations(
    ml_concentration: pd.DataFrame,
    dl_concentration: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    selected_ml = add_comparison_label(
        selected_ml_rows(ml_concentration, selected), "Nested-selected ML"
    )
    cnn_roi = add_comparison_label(
        dl_concentration.loc[dl_concentration["input_mode"] == "roi_masked"],
        "CNN central ROI",
    )
    cnn_full = add_comparison_label(
        dl_concentration.loc[dl_concentration["input_mode"] == "full_patch"],
        "CNN full patch",
    )
    keep = [
        "analyte",
        "outer_fold",
        "comparison_model",
        "concentration_order",
        "actual_concentration",
        "prediction_raw",
        "prediction",
        "patch_count",
    ]
    return pd.concat([selected_ml, cnn_roi, cnn_full], ignore_index=True)[keep]


def build_error_profile(predictions: pd.DataFrame) -> pd.DataFrame:
    working = predictions.copy()
    working["absolute_error"] = np.abs(
        working["actual_concentration"] - working["prediction"]
    )
    return (
        working.groupby(
            [
                "analyte",
                "comparison_model",
                "concentration_order",
                "actual_concentration",
            ],
            as_index=False,
        )["absolute_error"]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
    )


def build_efficiency(
    selected_ml_efficiency: pd.DataFrame,
    dl_metrics: pd.DataFrame,
) -> pd.DataFrame:
    ml = selected_ml_efficiency.copy()
    ml["comparison_model"] = "Nested-selected ML"
    ml["parameter_count"] = np.nan
    ml["training_seconds"] = ml["fit_seconds"]
    dl = dl_metrics.loc[dl_metrics["evaluation_unit"] == "patch"].copy()
    dl["comparison_model"] = np.where(
        dl["input_mode"] == "roi_masked", "CNN central ROI", "CNN full patch"
    )
    dl["tree_count"] = np.nan
    dl["total_tree_nodes"] = np.nan
    keep = [
        "analyte",
        "outer_fold",
        "comparison_model",
        "training_seconds",
        "inference_ms_per_patch",
        "model_size_bytes",
        "parameter_count",
        "tree_count",
        "total_tree_nodes",
    ]
    return pd.concat([ml, dl], ignore_index=True)[keep]


def plot_performance(summary: pd.DataFrame, output_path: Path) -> None:
    selected = summary.loc[summary["evaluation_unit"] == "patch"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for axis, analyte in zip(axes, ["glucose", "ketone"]):
        subset = selected.loc[selected["analyte"] == analyte].set_index(
            "comparison_model"
        )
        values = [100 * subset.loc[model, "normalized_mae_mean"] for model in MODEL_ORDER]
        errors = [100 * subset.loc[model, "normalized_mae_std"] for model in MODEL_ORDER]
        y = np.arange(len(MODEL_ORDER))
        axis.errorbar(
            values,
            y,
            xerr=errors,
            fmt="none",
            ecolor="#333333",
            capsize=4,
            linewidth=1.5,
        )
        axis.scatter(
            values,
            y,
            s=90,
            color=[MODEL_COLORS[model] for model in MODEL_ORDER],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        axis.set_yticks(y, MODEL_ORDER)
        axis.tick_params(axis="y", labelleft=True)
        axis.set_title(ANALYTE_LABELS[analyte], fontweight="bold")
        axis.set_xlabel("Normalized MAE (% of concentration range)")
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        for value, position in zip(values, y):
            axis.text(value, position - 0.22, f"{value:.1f}%", ha="center", fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Outer-fold performance: color-feature ML versus image CNN",
        fontsize=15,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.86, wspace=0.32)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_observed_predicted(concentrations: pd.DataFrame, output_path: Path) -> None:
    summarized = (
        concentrations.groupby(
            [
                "analyte",
                "comparison_model",
                "concentration_order",
                "actual_concentration",
            ],
            as_index=False,
        )["prediction"]
        .agg(["mean", "std"])
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    for axis, analyte in zip(axes, ["glucose", "ketone"]):
        analyte_data = summarized.loc[summarized["analyte"] == analyte]
        actual_table = (
            analyte_data[["concentration_order", "actual_concentration"]]
            .drop_duplicates()
            .sort_values("concentration_order")
        )
        x = np.arange(len(actual_table))
        axis.plot(
            x,
            actual_table["actual_concentration"],
            linestyle="--",
            color="#222222",
            linewidth=2,
            label="Actual concentration",
        )
        for model in HEADLINE_ORDER:
            model_data = (
                analyte_data.loc[analyte_data["comparison_model"] == model]
                .sort_values("concentration_order")
            )
            axis.errorbar(
                x,
                model_data["mean"],
                yerr=model_data["std"],
                marker="o",
                markersize=4,
                linewidth=1.5,
                capsize=2,
                color=MODEL_COLORS[model],
                label=model,
            )
        labels = [f"{value:g}" for value in actual_table["actual_concentration"]]
        axis.set_xticks(x, labels, rotation=45, ha="right")
        axis.set_title(ANALYTE_LABELS[analyte], fontweight="bold")
        axis.set_xlabel("Actual concentration (mg/mL)")
        axis.set_ylabel("Predicted concentration (mg/mL)")
        axis.grid(color="#D9D9D9", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(
        "Concentration-level median predictions across five outer folds",
        fontsize=15,
        fontweight="bold",
    )
    fig.subplots_adjust(bottom=0.25, top=0.86, wspace=0.25)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_error_profile(error_profile: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=False)
    for axis, analyte in zip(axes, ["glucose", "ketone"]):
        analyte_data = error_profile.loc[error_profile["analyte"] == analyte]
        concentrations = (
            analyte_data[["concentration_order", "actual_concentration"]]
            .drop_duplicates()
            .sort_values("concentration_order")
        )
        x = np.arange(len(concentrations))
        for model in HEADLINE_ORDER:
            model_data = (
                analyte_data.loc[analyte_data["comparison_model"] == model]
                .sort_values("concentration_order")
            )
            axis.plot(
                x,
                model_data["mean"],
                marker="o",
                linewidth=1.7,
                color=MODEL_COLORS[model],
                label=model,
            )
        axis.set_xticks(
            x,
            [f"{value:g}" for value in concentrations["actual_concentration"]],
            rotation=45,
            ha="right",
        )
        axis.set_title(ANALYTE_LABELS[analyte], fontweight="bold")
        axis.set_xlabel("Actual concentration (mg/mL)")
        axis.set_ylabel("Patch-level MAE (mg/mL)")
        axis.grid(color="#D9D9D9", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Concentration-specific prediction error", fontsize=15, fontweight="bold")
    fig.subplots_adjust(bottom=0.24, top=0.86, wspace=0.24)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    prepare_output_dir(args.output_dir, args.overwrite)
    selected = select_ml_pipeline(args.ml_dir)
    ml_metrics = pd.read_csv(args.ml_dir / "ml_fold_metrics.csv")
    ml_predictions = pd.read_csv(args.ml_dir / "ml_predictions.csv")
    ml_concentration = pd.read_csv(args.ml_dir / "ml_concentration_predictions.csv")
    dl_metrics = pd.read_csv(args.dl_dir / "dl_fold_metrics.csv")
    dl_predictions = pd.read_csv(args.dl_dir / "dl_predictions.csv")
    dl_concentration = pd.read_csv(args.dl_dir / "dl_concentration_predictions.csv")
    selected_ml_efficiency = pd.read_csv(
        args.selected_ml_dir / "selected_ml_model_efficiency.csv"
    )

    comparison_metrics = build_comparison_metrics(ml_metrics, dl_metrics, selected)
    comparison_summary = summarize_comparison(comparison_metrics)
    headline_predictions = build_headline_predictions(
        ml_predictions, dl_predictions, selected
    )
    headline_concentrations = build_headline_concentrations(
        ml_concentration, dl_concentration, selected
    )
    error_profile = build_error_profile(headline_predictions)
    efficiency = build_efficiency(selected_ml_efficiency, dl_metrics)

    selected.to_csv(args.output_dir / "selected_ml_folds.csv", index=False)
    comparison_metrics.to_csv(
        args.output_dir / "model_comparison_fold_metrics.csv", index=False
    )
    comparison_summary.to_csv(
        args.output_dir / "model_comparison_summary.csv", index=False
    )
    headline_predictions.to_csv(
        args.output_dir / "headline_model_predictions.csv", index=False
    )
    headline_concentrations.to_csv(
        args.output_dir / "headline_concentration_predictions.csv", index=False
    )
    error_profile.to_csv(
        args.output_dir / "concentration_error_summary.csv", index=False
    )
    efficiency.to_csv(args.output_dir / "model_efficiency.csv", index=False)

    plot_performance(
        comparison_summary,
        args.output_dir / "figures" / "model_performance_comparison.png",
    )
    plot_observed_predicted(
        headline_concentrations,
        args.output_dir / "figures" / "observed_vs_predicted.png",
    )
    plot_error_profile(
        error_profile,
        args.output_dir / "figures" / "concentration_error_profiles.png",
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "headline_ml_selection": "lowest inner MAE among primary feature/model candidates in each outer fold",
                "headline_dl_models": ["roi_masked", "full_patch"],
                "performance_unit": "patch-level outer-fold metrics",
                "error_bars": "standard deviation across five outer folds",
                "known_limitation": "different well positions from the same source image remain across train and test",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved model comparison to {args.output_dir}")


if __name__ == "__main__":
    main()

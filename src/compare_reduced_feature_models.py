"""Compare RGB models with parsimonious G-only and circular Hue-only models."""

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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from train_ml_models import (
    MODEL_ORDER,
    RANDOM_SEED,
    build_search,
    inner_cv_indices,
    model_specifications,
    serializable_parameters,
    target_mode,
)


ANALYTES = ["glucose", "ketone"]
ANALYTE_LABELS = {"glucose": "Glucose", "ketone": "Ketone"}
ANALYTE_RANGES = {"glucose": 20.0, "ketone": 10.0}
REDUCED_FEATURES = {
    "G_only": ["g_median"],
    "Hue_only": ["h_sin_weighted", "h_cos_weighted"],
}
FEATURE_ORDER = ["G_only", "RGB_primary", "Hue_only", "HSV_primary"]
FEATURE_LABELS = {
    "G_only": "G only",
    "RGB_primary": "RGB",
    "Hue_only": "Hue only",
    "HSV_primary": "HSV",
}


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
        "--existing-metrics",
        type=Path,
        default=Path("outputs/modeling/ml/ml_fold_metrics.csv"),
    )
    parser.add_argument(
        "--existing-parameters",
        type=Path,
        default=Path("outputs/modeling/ml/ml_best_parameters.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/modeling/reduced_features"),
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


def train_reduced_models(
    features: pd.DataFrame,
    splits: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for analyte in ANALYTES:
        analyte_max = ANALYTE_RANGES[analyte]
        for outer_fold in range(1, 6):
            fold_split = splits.loc[
                (splits["analyte"] == analyte)
                & (splits["outer_fold"] == outer_fold)
            ]
            train_assignment = fold_split.loc[
                fold_split["ml_role"] == "train", ["patch_id", "inner_fold"]
            ]
            test_assignment = fold_split.loc[
                fold_split["ml_role"] == "test", ["patch_id"]
            ]
            train_data = train_assignment.merge(
                features, on="patch_id", how="left", validate="one_to_one"
            )
            test_data = test_assignment.merge(
                features, on="patch_id", how="left", validate="one_to_one"
            )
            inner_splits = inner_cv_indices(train_data)
            actual = test_data["concentration_mg_ml"].to_numpy(dtype=float)
            for feature_set, selected_features in REDUCED_FEATURES.items():
                for model_name, (estimator, grid) in model_specifications(False).items():
                    search = build_search(estimator, grid, inner_splits)
                    search.fit(
                        train_data[selected_features],
                        train_data["concentration_mg_ml"],
                    )
                    raw = search.predict(test_data[selected_features])
                    prediction = np.clip(raw, 0.0, analyte_max)
                    mae = float(mean_absolute_error(actual, prediction))
                    rmse = float(math.sqrt(mean_squared_error(actual, prediction)))
                    metric_rows.append(
                        {
                            "analyte": analyte,
                            "outer_fold": outer_fold,
                            "feature_set": feature_set,
                            "model": model_name,
                            "feature_count": len(selected_features),
                            "target_transform": target_mode(search.best_estimator_),
                            "inner_best_mae": -float(search.best_score_),
                            "mae": mae,
                            "rmse": rmse,
                            "r2": float(r2_score(actual, prediction)),
                            "normalized_mae": mae / analyte_max,
                        }
                    )
                    parameter_rows.append(
                        {
                            "analyte": analyte,
                            "outer_fold": outer_fold,
                            "feature_set": feature_set,
                            "model": model_name,
                            "inner_best_mae": -float(search.best_score_),
                            "best_parameters": json.dumps(
                                serializable_parameters(search.best_params_)
                            ),
                        }
                    )
                    frame = test_data[
                        [
                            "patch_id",
                            "well_id",
                            "concentration_order",
                            "concentration_mg_ml",
                        ]
                    ].rename(columns={"concentration_mg_ml": "actual_concentration"})
                    frame.insert(0, "analyte", analyte)
                    frame.insert(1, "outer_fold", outer_fold)
                    frame.insert(2, "feature_set", feature_set)
                    frame.insert(3, "model", model_name)
                    frame["prediction_raw"] = raw
                    frame["prediction"] = prediction
                    prediction_rows.append(frame)
                    print(
                        f"Trained {analyte} fold={outer_fold} "
                        f"{feature_set} {model_name}",
                        flush=True,
                    )
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(parameter_rows),
        pd.concat(prediction_rows, ignore_index=True),
    )


def algorithm_summary(
    reduced_metrics: pd.DataFrame,
    existing_metrics: pd.DataFrame,
) -> pd.DataFrame:
    existing = existing_metrics.loc[
        (existing_metrics["evaluation_unit"] == "patch")
        & (existing_metrics["feature_set"].isin(["RGB_primary", "HSV_primary"]))
        & (existing_metrics["model"].isin(MODEL_ORDER)),
        [
            "analyte",
            "outer_fold",
            "feature_set",
            "model",
            "mae",
            "rmse",
            "r2",
            "normalized_mae",
        ],
    ].copy()
    reduced = reduced_metrics[
        [
            "analyte",
            "outer_fold",
            "feature_set",
            "model",
            "mae",
            "rmse",
            "r2",
            "normalized_mae",
        ]
    ].copy()
    combined = pd.concat([existing, reduced], ignore_index=True)
    return (
        combined.groupby(["analyte", "feature_set", "model"], as_index=False)
        .agg(
            folds=("outer_fold", "nunique"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            normalized_mae_mean=("normalized_mae", "mean"),
        )
        .sort_values(["analyte", "feature_set", "mae_mean"])
        .reset_index(drop=True)
    )


def select_existing_feature_family(
    feature_set: str,
    metrics: pd.DataFrame,
    parameters: pd.DataFrame,
) -> pd.DataFrame:
    chosen = (
        parameters.loc[parameters["feature_set"] == feature_set]
        .sort_values(["analyte", "outer_fold", "inner_best_mae"])
        .groupby(["analyte", "outer_fold"], as_index=False)
        .first()[["analyte", "outer_fold", "feature_set", "model", "inner_best_mae"]]
    )
    patch_metrics = metrics.loc[
        (metrics["evaluation_unit"] == "patch")
        & (metrics["feature_set"] == feature_set)
    ]
    return chosen.merge(
        patch_metrics,
        on=["analyte", "outer_fold", "feature_set", "model"],
        how="left",
        validate="one_to_one",
        suffixes=("_selection", ""),
    )


def nested_feature_family_summary(
    reduced_metrics: pd.DataFrame,
    existing_metrics: pd.DataFrame,
    existing_parameters: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_frames: list[pd.DataFrame] = []
    for feature_set in ["RGB_primary", "HSV_primary"]:
        selected_frames.append(
            select_existing_feature_family(
                feature_set, existing_metrics, existing_parameters
            )
        )
    for feature_set in REDUCED_FEATURES:
        selected_frames.append(
            reduced_metrics.loc[reduced_metrics["feature_set"] == feature_set]
            .sort_values(["analyte", "outer_fold", "inner_best_mae"])
            .groupby(["analyte", "outer_fold"], as_index=False)
            .first()
        )
    selected = pd.concat(selected_frames, ignore_index=True, sort=False)
    selected = selected[
        [
            "analyte",
            "outer_fold",
            "feature_set",
            "model",
            "inner_best_mae",
            "mae",
            "rmse",
            "r2",
            "normalized_mae",
        ]
    ].sort_values(["analyte", "feature_set", "outer_fold"])
    summary = (
        selected.groupby(["analyte", "feature_set"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            r2_mean=("r2", "mean"),
            normalized_mae_mean=("normalized_mae", "mean"),
            selected_models=("model", lambda values: ", ".join(sorted(set(values)))),
        )
    )
    return selected, summary


def plot_feature_families(
    summary: pd.DataFrame,
    output_path: Path,
    title: str = "Nested-CV comparison of single-channel and full color feature sets",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3), constrained_layout=True)
    colors = ["#62a76b", "#3b7fb6", "#d38b43", "#8b63a8"]
    for axis, analyte in zip(axes, ANALYTES, strict=True):
        data = (
            summary.loc[summary["analyte"] == analyte]
            .set_index("feature_set")
            .reindex(FEATURE_ORDER)
        )
        values = 100 * data["normalized_mae_mean"].to_numpy(dtype=float)
        errors = 100 * data["mae_std"].to_numpy(dtype=float) / ANALYTE_RANGES[analyte]
        x = np.arange(len(FEATURE_ORDER))
        axis.bar(x, values, yerr=errors, color=colors, capsize=4, width=0.68)
        label_pad = max(float(np.max(values + errors)) * 0.025, 0.04)
        for x_value, y_value, error in zip(x, values, errors, strict=True):
            axis.text(
                x_value,
                y_value + error + label_pad,
                f"{y_value:.2f}%",
                ha="center",
                fontsize=10,
            )
        axis.set_xticks(x, [FEATURE_LABELS[item] for item in FEATURE_ORDER])
        axis.set_ylabel("Normalized MAE (% of concentration range)")
        axis.set_title(ANALYTE_LABELS[analyte], fontweight="bold")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(title, fontsize=17, fontweight="bold")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    prepare_output_dir(args.output_dir, args.overwrite)
    features = pd.read_csv(args.features)
    splits = pd.read_csv(args.splits)
    existing_metrics = pd.read_csv(args.existing_metrics)
    existing_parameters = pd.read_csv(args.existing_parameters)
    reduced_metrics, reduced_parameters, reduced_predictions = train_reduced_models(
        features, splits
    )
    algorithms = algorithm_summary(reduced_metrics, existing_metrics)
    selected, feature_summary = nested_feature_family_summary(
        reduced_metrics, existing_metrics, existing_parameters
    )
    reduced_metrics.to_csv(args.output_dir / "reduced_feature_fold_metrics.csv", index=False)
    reduced_parameters.to_csv(
        args.output_dir / "reduced_feature_best_parameters.csv", index=False
    )
    reduced_predictions.to_csv(
        args.output_dir / "reduced_feature_predictions.csv", index=False
    )
    algorithms.to_csv(args.output_dir / "algorithm_feature_summary.csv", index=False)
    selected.to_csv(
        args.output_dir / "nested_selected_feature_family_folds.csv", index=False
    )
    feature_summary.to_csv(
        args.output_dir / "nested_selected_feature_family_summary.csv", index=False
    )
    plot_feature_families(
        feature_summary,
        args.output_dir / "figures" / "single_vs_full_color_features.png",
    )
    random_forest_summary = algorithms.loc[
        algorithms["model"] == "RandomForest"
    ].copy()
    plot_feature_families(
        random_forest_summary,
        args.output_dir
        / "figures"
        / "random_forest_single_vs_full_color_features.png",
        title="Random Forest comparison of single-channel and full color features",
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "random_seed": RANDOM_SEED,
                "outer_folds": 5,
                "inner_folds": 5,
                "G_only": ["g_median"],
                "Hue_only": ["h_sin_weighted", "h_cos_weighted"],
                "hue_note": "two circular components treated as one conceptual Hue feature",
                "selection_rule": "lowest inner-fold MAE within each feature family and outer fold",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

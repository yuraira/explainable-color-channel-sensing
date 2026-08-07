"""Measure Extra Trees performance and deployment cost after feature reduction."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from refit_selected_ml_models import build_estimator


RANDOM_SEED = 240920
ANALYTES = ["glucose", "ketone"]
ANALYTE_LABELS = {"glucose": "Glucose", "ketone": "Ketone"}
ANALYTE_RANGES = {"glucose": 20.0, "ketone": 10.0}
FEATURE_ORDER = ["G_only", "RGB_primary", "Hue_only", "HSV_primary"]
FEATURE_LABELS = {
    "G_only": "G only",
    "RGB_primary": "RGB",
    "Hue_only": "Hue only",
    "HSV_primary": "HSV",
}
FEATURES = {
    "G_only": ["g_median"],
    "RGB_primary": ["r_median", "g_median", "b_median"],
    "Hue_only": ["h_sin_weighted", "h_cos_weighted"],
    "HSV_primary": [
        "h_sin_weighted",
        "h_cos_weighted",
        "s_median",
        "v_median",
    ],
}
CONCEPTUAL_CHANNELS = {
    "G_only": 1,
    "RGB_primary": 3,
    "Hue_only": 1,
    "HSV_primary": 3,
}
INFERENCE_REPEATS = 50


def load_parameters() -> pd.DataFrame:
    primary = pd.read_csv("outputs/modeling/ml/ml_best_parameters.csv")
    primary = primary.loc[
        (primary["model"] == "ExtraTrees")
        & primary["feature_set"].isin(["RGB_primary", "HSV_primary"])
    ]
    reduced = pd.read_csv(
        "outputs/modeling/reduced_features/reduced_feature_best_parameters.csv"
    )
    reduced = reduced.loc[
        (reduced["model"] == "ExtraTrees")
        & reduced["feature_set"].isin(["G_only", "Hue_only"])
    ]
    return pd.concat([primary, reduced], ignore_index=True)


def measure_fold_efficiency() -> pd.DataFrame:
    features = pd.read_csv("outputs/color_features/features.csv")
    splits = pd.read_csv("outputs/data_splits/nested_split_assignments.csv")
    parameters = load_parameters()
    rows: list[dict[str, object]] = []
    for analyte in ANALYTES:
        for feature_set in FEATURE_ORDER:
            selected_features = FEATURES[feature_set]
            for outer_fold in range(1, 6):
                parameter_row = parameters.loc[
                    (parameters["analyte"] == analyte)
                    & (parameters["outer_fold"] == outer_fold)
                    & (parameters["feature_set"] == feature_set)
                ]
                if len(parameter_row) != 1:
                    raise ValueError(
                        f"Expected one parameter row for {analyte}, {feature_set}, "
                        f"fold {outer_fold}; found {len(parameter_row)}"
                    )
                best_parameters = json.loads(parameter_row.iloc[0]["best_parameters"])
                estimator = build_estimator("ExtraTrees", best_parameters)
                fold_split = splits.loc[
                    (splits["analyte"] == analyte)
                    & (splits["outer_fold"] == outer_fold)
                ]
                train_ids = fold_split.loc[
                    fold_split["ml_role"] == "train", "patch_id"
                ]
                test_ids = fold_split.loc[
                    fold_split["ml_role"] == "test", "patch_id"
                ]
                train_data = pd.DataFrame({"patch_id": train_ids}).merge(
                    features, on="patch_id", how="left", validate="one_to_one"
                )
                test_data = pd.DataFrame({"patch_id": test_ids}).merge(
                    features, on="patch_id", how="left", validate="one_to_one"
                )
                fit_start = time.perf_counter()
                estimator.fit(
                    train_data[selected_features],
                    train_data["concentration_mg_ml"],
                )
                fit_seconds = time.perf_counter() - fit_start
                estimator.predict(test_data[selected_features])
                timings: list[float] = []
                for _ in range(INFERENCE_REPEATS):
                    predict_start = time.perf_counter()
                    estimator.predict(test_data[selected_features])
                    timings.append(time.perf_counter() - predict_start)
                fitted_model = estimator.regressor_.named_steps["model"]
                node_counts = np.asarray(
                    [tree.tree_.node_count for tree in fitted_model.estimators_],
                    dtype=int,
                )
                depths = np.asarray(
                    [tree.tree_.max_depth for tree in fitted_model.estimators_],
                    dtype=int,
                )
                with tempfile.TemporaryDirectory() as temporary_directory:
                    model_path = Path(temporary_directory) / "model.joblib"
                    joblib.dump(estimator, model_path, compress=3)
                    model_size_bytes = model_path.stat().st_size
                rows.append(
                    {
                        "analyte": analyte,
                        "outer_fold": outer_fold,
                        "feature_set": feature_set,
                        "conceptual_channel_count": CONCEPTUAL_CHANNELS[feature_set],
                        "numeric_feature_count": len(selected_features),
                        "float32_input_bytes_per_patch": 4 * len(selected_features),
                        "tree_count": len(fitted_model.estimators_),
                        "total_tree_nodes": int(node_counts.sum()),
                        "mean_tree_depth": float(depths.mean()),
                        "model_size_bytes": model_size_bytes,
                        "fit_seconds": fit_seconds,
                        "inference_ms_per_patch": float(
                            1000 * np.median(timings) / len(test_data)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    performance = pd.read_csv(
        "outputs/modeling/reduced_features/algorithm_feature_summary.csv"
    )
    performance = performance.loc[performance["model"] == "ExtraTrees"]
    efficiency = (
        folds.groupby(["analyte", "feature_set"], as_index=False)
        .agg(
            folds=("outer_fold", "nunique"),
            conceptual_channel_count=("conceptual_channel_count", "first"),
            numeric_feature_count=("numeric_feature_count", "first"),
            float32_input_bytes_per_patch=(
                "float32_input_bytes_per_patch",
                "first",
            ),
            model_size_bytes_mean=("model_size_bytes", "mean"),
            model_size_bytes_std=("model_size_bytes", "std"),
            total_tree_nodes_mean=("total_tree_nodes", "mean"),
            total_tree_nodes_std=("total_tree_nodes", "std"),
            mean_tree_depth=("mean_tree_depth", "mean"),
            fit_seconds_mean=("fit_seconds", "mean"),
            inference_ms_per_patch_median=("inference_ms_per_patch", "median"),
        )
    )
    keep = [
        "analyte",
        "feature_set",
        "mae_mean",
        "mae_std",
        "r2_mean",
        "r2_std",
        "normalized_mae_mean",
    ]
    summary = efficiency.merge(
        performance[keep],
        on=["analyte", "feature_set"],
        how="left",
        validate="one_to_one",
    )
    baseline_map = {
        "G_only": "RGB_primary",
        "RGB_primary": "RGB_primary",
        "Hue_only": "HSV_primary",
        "HSV_primary": "HSV_primary",
    }
    summary["full_feature_set"] = summary["feature_set"].map(baseline_map)
    baseline = summary[
        [
            "analyte",
            "feature_set",
            "numeric_feature_count",
            "float32_input_bytes_per_patch",
            "model_size_bytes_mean",
            "total_tree_nodes_mean",
            "inference_ms_per_patch_median",
        ]
    ].rename(
        columns={
            "feature_set": "full_feature_set",
            "numeric_feature_count": "full_numeric_feature_count",
            "float32_input_bytes_per_patch": "full_input_bytes",
            "model_size_bytes_mean": "full_model_size_bytes",
            "total_tree_nodes_mean": "full_tree_nodes",
            "inference_ms_per_patch_median": "full_inference_ms",
        }
    )
    summary = summary.merge(
        baseline,
        on=["analyte", "full_feature_set"],
        how="left",
        validate="many_to_one",
    )
    summary["numeric_input_reduction_percent"] = 100 * (
        1 - summary["numeric_feature_count"] / summary["full_numeric_feature_count"]
    )
    summary["model_size_change_percent"] = 100 * (
        summary["model_size_bytes_mean"] / summary["full_model_size_bytes"] - 1
    )
    summary["tree_node_change_percent"] = 100 * (
        summary["total_tree_nodes_mean"] / summary["full_tree_nodes"] - 1
    )
    summary["inference_time_change_percent"] = 100 * (
        summary["inference_ms_per_patch_median"] / summary["full_inference_ms"] - 1
    )
    return summary.sort_values(["analyte", "feature_set"]).reset_index(drop=True)


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), constrained_layout=True)
    colors = ["#62A76B", "#3B7FB6", "#D38B43", "#8B63A8"]
    for column, analyte in enumerate(ANALYTES):
        data = (
            summary.loc[summary["analyte"] == analyte]
            .set_index("feature_set")
            .reindex(FEATURE_ORDER)
        )
        x = np.arange(len(FEATURE_ORDER))
        performance_axis = axes[0, column]
        values = 100 * data["normalized_mae_mean"].to_numpy(dtype=float)
        errors = 100 * data["mae_std"].to_numpy(dtype=float) / ANALYTE_RANGES[analyte]
        performance_axis.bar(x, values, yerr=errors, capsize=4, color=colors)
        performance_axis.set_title(
            f"{ANALYTE_LABELS[analyte]} prediction error", fontweight="bold"
        )
        performance_axis.set_ylabel("Normalized MAE (% of range)")
        performance_axis.set_xticks(x, [FEATURE_LABELS[item] for item in FEATURE_ORDER])
        performance_axis.grid(axis="y", alpha=0.25)
        size_axis = axes[1, column]
        sizes = data["model_size_bytes_mean"].to_numpy(dtype=float) / (1024**2)
        size_errors = data["model_size_bytes_std"].to_numpy(dtype=float) / (1024**2)
        size_axis.bar(x, sizes, yerr=size_errors, capsize=4, color=colors)
        size_axis.set_title(
            f"{ANALYTE_LABELS[analyte]} serialized model size", fontweight="bold"
        )
        size_axis.set_ylabel("Model size (MiB)")
        size_axis.set_xticks(x, [FEATURE_LABELS[item] for item in FEATURE_ORDER])
        size_axis.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Extra Trees: feature reduction trades input size for prediction accuracy",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    output_dir = Path("outputs/modeling/feature_efficiency")
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    folds = measure_fold_efficiency()
    summary = summarize(folds)
    folds.to_csv(output_dir / "extra_trees_efficiency_folds.csv", index=False)
    summary.to_csv(output_dir / "extra_trees_efficiency_summary.csv", index=False)
    plot_summary(
        summary,
        figures_dir / "extra_trees_feature_reduction_efficiency.png",
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "random_seed": RANDOM_SEED,
                "tree_count": 150,
                "outer_folds": 5,
                "inference_repeats": INFERENCE_REPEATS,
                "serialization": "joblib compress=3",
                "input_storage_assumption": "float32 numeric feature vector only",
                "hue_note": "one conceptual Hue channel encoded as sine and cosine",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

"""Train color-feature regression models with nested group-aware CV."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVR


RANDOM_SEED = 240920
N_SPLITS = 5
PRIMARY_FEATURE_SETS = ["RGB_primary", "HSV_primary", "RGB_HSV_combined"]
SECONDARY_FEATURE_SETS = [
    "Chromaticity_secondary",
    "Background_adjusted_secondary",
    "Background_negative_control",
]
MODEL_ORDER = ["Ridge", "ElasticNet", "SVR", "RandomForest", "ExtraTrees"]
ANALYTE_LABELS = {"glucose": "Glucose", "ketone": "Ketone"}
FEATURE_LABELS = {
    "RGB_primary": "RGB",
    "HSV_primary": "HSV",
    "RGB_HSV_combined": "RGB+HSV",
    "Chromaticity_secondary": "Chromaticity",
    "Background_adjusted_secondary": "Background-adjusted",
    "Background_negative_control": "Background only",
    "Baseline": "Mean baseline",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("outputs/color_features/features.csv"),
    )
    parser.add_argument(
        "--feature-sets",
        type=Path,
        default=Path("outputs/feature_validation/model_feature_sets.csv"),
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("outputs/data_splits/nested_split_assignments.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/modeling/ml"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--debug-one-fold", action="store_true")
    return parser.parse_args()


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {path}. Use --overwrite to replace it."
            )
        shutil.rmtree(path)
    (path / "figures").mkdir(parents=True, exist_ok=True)


def identity_transformer() -> FunctionTransformer:
    return FunctionTransformer(validate=True)


def log_transformer() -> FunctionTransformer:
    return FunctionTransformer(
        func=np.log1p,
        inverse_func=np.expm1,
        validate=True,
        check_inverse=True,
    )


def model_specifications(smoke_test: bool) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    tree_count = 30 if smoke_test else 150
    specifications: dict[str, tuple[Any, dict[str, list[Any]]]] = {
        "Ridge": (
            Ridge(),
            {"regressor__model__alpha": [0.1, 1.0, 10.0]},
        ),
        "ElasticNet": (
            ElasticNet(max_iter=30_000, random_state=RANDOM_SEED),
            {
                "regressor__model__alpha": [0.001, 0.01, 0.1],
                "regressor__model__l1_ratio": [0.2, 0.8],
            },
        ),
        "SVR": (
            SVR(kernel="rbf", gamma="scale"),
            {
                "regressor__model__C": [1.0, 10.0, 100.0],
                "regressor__model__epsilon": [0.01, 0.1],
            },
        ),
        "RandomForest": (
            RandomForestRegressor(
                n_estimators=tree_count,
                random_state=RANDOM_SEED,
                n_jobs=1,
            ),
            {
                "regressor__model__max_depth": [None, 8],
                "regressor__model__min_samples_leaf": [1, 3],
            },
        ),
        "ExtraTrees": (
            ExtraTreesRegressor(
                n_estimators=tree_count,
                random_state=RANDOM_SEED,
                n_jobs=1,
            ),
            {
                "regressor__model__max_depth": [None, 8],
                "regressor__model__min_samples_leaf": [1, 3],
            },
        ),
    }
    if smoke_test:
        return {"Ridge": specifications["Ridge"]}
    return specifications


def build_search(
    estimator: Any,
    model_grid: dict[str, list[Any]],
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
) -> GridSearchCV:
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", estimator),
        ]
    )
    regressor = TransformedTargetRegressor(
        regressor=pipeline,
        transformer=identity_transformer(),
    )
    parameter_grid = []
    for transformer in [identity_transformer(), log_transformer()]:
        grid = {key: value for key, value in model_grid.items()}
        grid["transformer"] = [transformer]
        parameter_grid.append(grid)
    return GridSearchCV(
        regressor,
        param_grid=parameter_grid,
        scoring="neg_mean_absolute_error",
        cv=inner_splits,
        refit=True,
        n_jobs=1,
        error_score="raise",
        return_train_score=False,
    )


def target_mode(best_estimator: TransformedTargetRegressor) -> str:
    transformer = best_estimator.transformer_
    return "log1p" if getattr(transformer, "func", None) is np.log1p else "raw"


def serializable_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in parameters.items():
        if key == "transformer":
            output["target_transform"] = (
                "log1p" if getattr(value, "func", None) is np.log1p else "raw"
            )
        elif isinstance(value, (str, int, float, bool)) or value is None:
            output[key.replace("regressor__model__", "")] = value
        else:
            output[key.replace("regressor__model__", "")] = str(value)
    return output


def inner_cv_indices(train_data: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    folds = train_data["inner_fold"].to_numpy(dtype=int)
    if set(np.unique(folds)) != set(range(1, N_SPLITS + 1)):
        raise ValueError(f"Expected inner folds 1 through 5, found {sorted(set(folds))}")
    return [
        (np.flatnonzero(folds != fold), np.flatnonzero(folds == fold))
        for fold in range(1, N_SPLITS + 1)
    ]


def calculate_metrics(
    actual: np.ndarray,
    predicted_raw: np.ndarray,
    predicted: np.ndarray,
    concentration_range: float,
) -> dict[str, float]:
    mae = mean_absolute_error(actual, predicted)
    rmse = math.sqrt(mean_squared_error(actual, predicted))
    rho = (
        np.nan
        if np.unique(actual).size < 2 or np.unique(predicted).size < 2
        else spearmanr(actual, predicted).statistic
    )
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2_score(actual, predicted)),
        "spearman_rho": float(rho),
        "normalized_mae": float(mae / concentration_range),
        "normalized_rmse": float(rmse / concentration_range),
        "mean_absolute_log1p_error": float(
            mean_absolute_error(np.log1p(actual), np.log1p(predicted))
        ),
        "unclipped_mae": float(mean_absolute_error(actual, predicted_raw)),
        "clipped_fraction": float(np.mean(predicted_raw != predicted)),
    }


def evaluate_predictions(
    prediction_frame: pd.DataFrame,
    feature_set: str,
    model: str,
    outer_fold: int,
    analyte: str,
    fit_seconds: float,
    predict_seconds: float,
    inner_best_mae: float | None,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    concentration_range = float(
        prediction_frame["actual_concentration"].max()
        - prediction_frame["actual_concentration"].min()
    )
    patch_metrics = calculate_metrics(
        prediction_frame["actual_concentration"].to_numpy(),
        prediction_frame["prediction_raw"].to_numpy(),
        prediction_frame["prediction"].to_numpy(),
        concentration_range,
    )
    patch_metrics.update(
        {
            "analyte": analyte,
            "feature_set": feature_set,
            "model": model,
            "outer_fold": outer_fold,
            "evaluation_unit": "patch",
            "n_observations": len(prediction_frame),
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "inner_best_mae": inner_best_mae,
        }
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
    concentration_metrics.update(
        {
            "analyte": analyte,
            "feature_set": feature_set,
            "model": model,
            "outer_fold": outer_fold,
            "evaluation_unit": "concentration_median",
            "n_observations": len(concentration),
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "inner_best_mae": inner_best_mae,
        }
    )
    concentration.insert(0, "analyte", analyte)
    concentration.insert(1, "feature_set", feature_set)
    concentration.insert(2, "model", model)
    concentration.insert(3, "outer_fold", outer_fold)
    return [patch_metrics, concentration_metrics], concentration


def plot_ml_comparison(summary: pd.DataFrame, output_path: Path) -> None:
    primary = summary.loc[
        (summary["evaluation_unit"] == "patch")
        & summary["feature_set"].isin(PRIMARY_FEATURE_SETS)
        & summary["model"].isin(MODEL_ORDER)
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)
    colors = {
        "Ridge": "#4C78A8",
        "ElasticNet": "#72B7B2",
        "SVR": "#F2CF5B",
        "RandomForest": "#E45756",
        "ExtraTrees": "#B279A2",
    }
    for axis, analyte in zip(axes, ["glucose", "ketone"]):
        subset = primary.loc[primary["analyte"] == analyte]
        x = np.arange(len(PRIMARY_FEATURE_SETS), dtype=float)
        offsets = np.linspace(-0.26, 0.26, len(MODEL_ORDER))
        for offset, model in zip(offsets, MODEL_ORDER):
            model_data = (
                subset.loc[subset["model"] == model]
                .set_index("feature_set")
                .reindex(PRIMARY_FEATURE_SETS)
            )
            axis.errorbar(
                x + offset,
                100 * model_data["normalized_mae_mean"],
                yerr=100 * model_data["normalized_mae_std"],
                marker="o",
                linewidth=1.5,
                capsize=3,
                color=colors[model],
                label=model,
            )
        axis.set_xticks(x, [FEATURE_LABELS[item] for item in PRIMARY_FEATURE_SETS])
        axis.set_title(ANALYTE_LABELS[analyte], fontweight="bold")
        axis.set_xlabel("Feature set")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Outer-fold normalized MAE (% of range, mean ± SD)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle(
        "Nested cross-validation of color-feature ML models",
        fontsize=15,
        fontweight="bold",
    )
    fig.subplots_adjust(bottom=0.20, top=0.86, wspace=0.12)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    prepare_output_dir(args.output_dir, args.overwrite)
    features = pd.read_csv(args.features)
    split_assignments = pd.read_csv(args.splits)
    feature_rows = pd.read_csv(args.feature_sets)
    feature_sets = (
        feature_rows.groupby("feature_set", sort=False)["feature"].apply(list).to_dict()
    )
    requested_feature_sets = PRIMARY_FEATURE_SETS + SECONDARY_FEATURE_SETS
    missing_feature_sets = sorted(set(requested_feature_sets) - set(feature_sets))
    if missing_feature_sets:
        raise ValueError(f"Missing feature-set definitions: {missing_feature_sets}")

    split_columns = [
        "outer_fold",
        "patch_id",
        "analyte",
        "concentration_order",
        "concentration_mg_ml",
        "well_id",
        "inner_fold",
        "ml_role",
    ]
    data = split_assignments[split_columns].merge(
        features[["patch_id"] + sorted(set(sum(feature_sets.values(), [])))],
        on="patch_id",
        how="left",
        validate="many_to_one",
    )
    if data.isna().any().any():
        raise ValueError("Merged modeling table contains missing values")

    model_specs = model_specifications(args.smoke_test)
    limited_run = args.smoke_test or args.debug_one_fold
    analytes = ["glucose"] if limited_run else ["glucose", "ketone"]
    outer_folds = [1] if limited_run else list(range(1, N_SPLITS + 1))
    primary_sets = ["RGB_primary"] if limited_run else PRIMARY_FEATURE_SETS
    secondary_sets = [] if limited_run else SECONDARY_FEATURE_SETS

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    concentration_rows: list[pd.DataFrame] = []
    best_parameter_rows: list[dict[str, Any]] = []

    for analyte in analytes:
        analyte_max = float(features.loc[features["analyte"] == analyte, "concentration_mg_ml"].max())
        for outer_fold in outer_folds:
            fold_data = data.loc[
                (data["analyte"] == analyte) & (data["outer_fold"] == outer_fold)
            ].copy()
            train_data = fold_data.loc[fold_data["ml_role"] == "train"].reset_index(drop=True)
            test_data = fold_data.loc[fold_data["ml_role"] == "test"].reset_index(drop=True)
            splits = inner_cv_indices(train_data)

            dummy_start = time.perf_counter()
            dummy = DummyRegressor(strategy="mean")
            dummy.fit(np.zeros((len(train_data), 1)), train_data["concentration_mg_ml"])
            dummy_fit_seconds = time.perf_counter() - dummy_start
            predict_start = time.perf_counter()
            dummy_raw = dummy.predict(np.zeros((len(test_data), 1)))
            dummy_predict_seconds = time.perf_counter() - predict_start
            dummy_prediction = np.clip(dummy_raw, 0.0, analyte_max)
            dummy_frame = test_data[
                [
                    "patch_id",
                    "well_id",
                    "concentration_order",
                    "concentration_mg_ml",
                ]
            ].rename(columns={"concentration_mg_ml": "actual_concentration"})
            dummy_frame.insert(0, "analyte", analyte)
            dummy_frame.insert(1, "feature_set", "Baseline")
            dummy_frame.insert(2, "model", "DummyMean")
            dummy_frame.insert(3, "outer_fold", outer_fold)
            dummy_frame["prediction_raw"] = dummy_raw
            dummy_frame["prediction"] = dummy_prediction
            metrics, concentrations = evaluate_predictions(
                dummy_frame,
                "Baseline",
                "DummyMean",
                outer_fold,
                analyte,
                dummy_fit_seconds,
                dummy_predict_seconds,
                None,
            )
            metric_rows.extend(metrics)
            prediction_rows.append(dummy_frame)
            concentration_rows.append(concentrations)

            combinations = [
                (feature_set, model_name)
                for feature_set in primary_sets
                for model_name in model_specs
            ] + [(feature_set, "ExtraTrees") for feature_set in secondary_sets]
            for feature_set, model_name in combinations:
                selected_features = feature_sets[feature_set]
                estimator, parameter_grid = model_specs[model_name]
                search = build_search(estimator, parameter_grid, splits)
                fit_start = time.perf_counter()
                search.fit(
                    train_data[selected_features],
                    train_data["concentration_mg_ml"],
                )
                fit_seconds = time.perf_counter() - fit_start
                predict_start = time.perf_counter()
                prediction_raw = search.predict(test_data[selected_features])
                predict_seconds = time.perf_counter() - predict_start
                prediction = np.clip(prediction_raw, 0.0, analyte_max)
                frame = test_data[
                    [
                        "patch_id",
                        "well_id",
                        "concentration_order",
                        "concentration_mg_ml",
                    ]
                ].rename(columns={"concentration_mg_ml": "actual_concentration"})
                frame.insert(0, "analyte", analyte)
                frame.insert(1, "feature_set", feature_set)
                frame.insert(2, "model", model_name)
                frame.insert(3, "outer_fold", outer_fold)
                frame["prediction_raw"] = prediction_raw
                frame["prediction"] = prediction
                inner_best_mae = float(-search.best_score_)
                metrics, concentrations = evaluate_predictions(
                    frame,
                    feature_set,
                    model_name,
                    outer_fold,
                    analyte,
                    fit_seconds,
                    predict_seconds,
                    inner_best_mae,
                )
                metric_rows.extend(metrics)
                prediction_rows.append(frame)
                concentration_rows.append(concentrations)
                parameters = serializable_parameters(search.best_params_)
                best_parameter_rows.append(
                    {
                        "analyte": analyte,
                        "feature_set": feature_set,
                        "model": model_name,
                        "outer_fold": outer_fold,
                        "target_transform": target_mode(search.best_estimator_),
                        "inner_best_mae": inner_best_mae,
                        "best_parameters": json.dumps(parameters, sort_keys=True),
                    }
                )
                print(
                    f"{analyte} fold={outer_fold} {feature_set} {model_name}: "
                    f"inner MAE={inner_best_mae:.4f}, target={target_mode(search.best_estimator_)}",
                    flush=True,
                )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    concentration_predictions = pd.concat(concentration_rows, ignore_index=True)
    best_parameters = pd.DataFrame(best_parameter_rows)
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
        "fit_seconds",
        "predict_seconds",
        "inner_best_mae",
    ]
    summary = (
        metrics.groupby(
            ["analyte", "feature_set", "model", "evaluation_unit"],
            as_index=False,
        )[metric_columns]
        .agg(["mean", "std"])
    )
    summary.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]

    metrics.to_csv(args.output_dir / "ml_fold_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "ml_performance_summary.csv", index=False)
    predictions.to_csv(args.output_dir / "ml_predictions.csv", index=False)
    concentration_predictions.to_csv(
        args.output_dir / "ml_concentration_predictions.csv", index=False
    )
    best_parameters.to_csv(args.output_dir / "ml_best_parameters.csv", index=False)
    plot_ml_comparison(
        summary,
        args.output_dir / "figures" / "ml_model_comparison.png",
    )
    config = {
        "random_seed": RANDOM_SEED,
        "outer_folds": outer_folds,
        "inner_folds": N_SPLITS,
        "primary_feature_sets": primary_sets,
        "secondary_feature_sets": secondary_sets,
        "models": list(model_specs),
        "inner_selection_metric": "MAE on original concentration scale",
        "target_transform_candidates": ["raw", "log1p"],
        "reported_prediction": "clipped to 0 and analyte training maximum",
        "evaluation_units": ["patch", "concentration_median"],
        "smoke_test": args.smoke_test,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved ML results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

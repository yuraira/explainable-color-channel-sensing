"""Refit the inner-CV-selected ML pipeline for efficiency and later SHAP analysis."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


RANDOM_SEED = 240920
PRIMARY_FEATURE_SETS = {"RGB_primary", "HSV_primary", "RGB_HSV_combined"}


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
        "--best-parameters",
        type=Path,
        default=Path("outputs/modeling/ml/ml_best_parameters.csv"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/modeling/ml/ml_predictions.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/modeling/selected_ml"),
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
    (path / "models").mkdir(parents=True, exist_ok=True)


def build_estimator(model_name: str, parameters: dict[str, object]) -> object:
    common = {
        "n_estimators": 150,
        "max_depth": parameters.get("max_depth"),
        "min_samples_leaf": int(parameters.get("min_samples_leaf", 1)),
        "random_state": RANDOM_SEED,
        "n_jobs": 1,
    }
    if model_name == "ExtraTrees":
        model = ExtraTreesRegressor(**common)
    elif model_name == "RandomForest":
        model = RandomForestRegressor(**common)
    else:
        raise ValueError(
            f"Selected model {model_name} is not a supported tree estimator for refit"
        )
    pipeline = Pipeline([("scale", StandardScaler()), ("model", model)])
    if parameters.get("target_transform") == "log1p":
        transformer = FunctionTransformer(
            func=np.log1p,
            inverse_func=np.expm1,
            validate=True,
            check_inverse=True,
        )
    else:
        transformer = FunctionTransformer(validate=True)
    return TransformedTargetRegressor(regressor=pipeline, transformer=transformer)


def main() -> None:
    args = parse_args()
    prepare_output_dir(args.output_dir, args.overwrite)
    features = pd.read_csv(args.features)
    feature_rows = pd.read_csv(args.feature_sets)
    feature_sets = (
        feature_rows.groupby("feature_set", sort=False)["feature"].apply(list).to_dict()
    )
    splits = pd.read_csv(args.splits)
    parameters = pd.read_csv(args.best_parameters)
    previous_predictions = pd.read_csv(args.predictions)

    selected = (
        parameters.loc[parameters["feature_set"].isin(PRIMARY_FEATURE_SETS)]
        .sort_values(["analyte", "outer_fold", "inner_best_mae"])
        .groupby(["analyte", "outer_fold"], as_index=False)
        .first()
    )
    output_rows: list[dict[str, object]] = []
    for row in selected.itertuples(index=False):
        analyte = str(row.analyte)
        outer_fold = int(row.outer_fold)
        feature_set = str(row.feature_set)
        model_name = str(row.model)
        selected_features = feature_sets[feature_set]
        best_parameters = json.loads(str(row.best_parameters))
        estimator = build_estimator(model_name, best_parameters)
        fold_splits = splits.loc[
            (splits["analyte"] == analyte) & (splits["outer_fold"] == outer_fold)
        ]
        train_ids = fold_splits.loc[fold_splits["ml_role"] == "train", "patch_id"]
        test_ids = fold_splits.loc[fold_splits["ml_role"] == "test", "patch_id"]
        train_data = (
            pd.DataFrame({"patch_id": train_ids})
            .merge(features, on="patch_id", how="left", validate="one_to_one")
        )
        test_data = (
            pd.DataFrame({"patch_id": test_ids})
            .merge(features, on="patch_id", how="left", validate="one_to_one")
        )
        fit_start = time.perf_counter()
        estimator.fit(train_data[selected_features], train_data["concentration_mg_ml"])
        fit_seconds = time.perf_counter() - fit_start
        estimator.predict(test_data[selected_features])
        timings: list[float] = []
        for _ in range(30):
            start = time.perf_counter()
            estimator.predict(test_data[selected_features])
            timings.append(time.perf_counter() - start)
        prediction_raw = estimator.predict(test_data[selected_features])
        expected = (
            previous_predictions.loc[
                (previous_predictions["analyte"] == analyte)
                & (previous_predictions["outer_fold"] == outer_fold)
                & (previous_predictions["feature_set"] == feature_set)
                & (previous_predictions["model"] == model_name),
                ["patch_id", "prediction_raw"],
            ]
            .set_index("patch_id")
            .loc[test_data["patch_id"], "prediction_raw"]
            .to_numpy()
        )
        maximum_absolute_difference = float(np.max(np.abs(prediction_raw - expected)))
        if maximum_absolute_difference > 1e-10:
            raise RuntimeError(
                f"Refit prediction mismatch for {analyte} fold {outer_fold}: "
                f"{maximum_absolute_difference}"
            )
        model_path = args.output_dir / "models" / f"{analyte}_fold{outer_fold}.joblib"
        joblib.dump(estimator, model_path, compress=3)
        fitted_model = estimator.regressor_.named_steps["model"]
        total_nodes = int(sum(tree.tree_.node_count for tree in fitted_model.estimators_))
        output_rows.append(
            {
                "analyte": analyte,
                "outer_fold": outer_fold,
                "feature_set": feature_set,
                "model": model_name,
                "target_transform": best_parameters["target_transform"],
                "feature_count": len(selected_features),
                "tree_count": len(fitted_model.estimators_),
                "total_tree_nodes": total_nodes,
                "fit_seconds": fit_seconds,
                "inference_seconds_median": float(np.median(timings)),
                "inference_ms_per_patch": float(
                    1000 * np.median(timings) / len(test_data)
                ),
                "model_size_bytes": model_path.stat().st_size,
                "prediction_max_abs_difference": maximum_absolute_difference,
                "model_file": model_path.as_posix(),
            }
        )
        print(
            f"Saved {analyte} fold={outer_fold} {feature_set} {model_name}",
            flush=True,
        )
    output = pd.DataFrame(output_rows)
    output.to_csv(args.output_dir / "selected_ml_model_efficiency.csv", index=False)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "selection_rule": "lowest inner_best_mae among primary feature/model candidates",
                "random_seed": RANDOM_SEED,
                "prediction_reproduction_tolerance": 1e-10,
                "inference_repeats": 30,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

"""Explain fixed RGB-only and HSV-only tree models across outer test folds."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import shap
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 240920
N_SPLITS = 5
N_TREES = 150
PERMUTATION_REPEATS = 30
FEATURE_SPACES = ["RGB_primary", "HSV_primary"]
ANALYTES = ["glucose", "ketone"]
ANALYTE_LABELS = {"glucose": "Glucose", "ketone": "Ketone"}
SPACE_LABELS = {"RGB_primary": "RGB-only", "HSV_primary": "HSV-only"}
MODEL_BY_ANALYTE = {"glucose": "ExtraTrees", "ketone": "RandomForest"}
GROUPS = {
    "RGB_primary": {
        "R": ["r_median"],
        "G": ["g_median"],
        "B": ["b_median"],
    },
    "HSV_primary": {
        "Hue": ["h_sin_weighted", "h_cos_weighted"],
        "Saturation": ["s_median"],
        "Value": ["v_median"],
    },
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
        default=Path("outputs/explainability/ml"),
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


def inner_cv_indices(train_data: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    folds = train_data["inner_fold"].to_numpy(dtype=int)
    if set(np.unique(folds)) != set(range(1, N_SPLITS + 1)):
        raise ValueError(f"Expected inner folds 1 through 5, found {sorted(set(folds))}")
    return [
        (np.flatnonzero(folds != fold), np.flatnonzero(folds == fold))
        for fold in range(1, N_SPLITS + 1)
    ]


def build_search(model_name: str, inner_splits: list[tuple[np.ndarray, np.ndarray]]) -> GridSearchCV:
    common = {
        "n_estimators": N_TREES,
        "random_state": RANDOM_SEED,
        "n_jobs": 1,
    }
    if model_name == "ExtraTrees":
        model = ExtraTreesRegressor(**common)
    elif model_name == "RandomForest":
        model = RandomForestRegressor(**common)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    pipeline = Pipeline([("scale", StandardScaler()), ("model", model)])
    return GridSearchCV(
        pipeline,
        param_grid={
            "model__max_depth": [None, 8],
            "model__min_samples_leaf": [1, 3],
        },
        scoring="neg_mean_absolute_error",
        cv=inner_splits,
        refit=True,
        n_jobs=1,
        error_score="raise",
    )


def contiguous_hue_degrees(values: np.ndarray) -> np.ndarray:
    """Map circular degrees to the shortest contiguous interval for display/ranking."""
    values = np.mod(np.asarray(values, dtype=float), 360.0)
    if values.size < 2:
        return values
    ordered = np.sort(values)
    gaps = np.diff(np.r_[ordered, ordered[0] + 360.0])
    cut = ordered[(int(np.argmax(gaps)) + 1) % len(ordered)]
    return np.mod(values - cut, 360.0) + cut


def group_feature_values(data: pd.DataFrame, group: str) -> np.ndarray:
    if group == "Hue":
        return contiguous_hue_degrees(data["h_circular_mean_deg"].to_numpy(dtype=float))
    column = {
        "R": "r_median",
        "G": "g_median",
        "B": "b_median",
        "Saturation": "s_median",
        "Value": "v_median",
    }[group]
    return data[column].to_numpy(dtype=float)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return math.nan
    return float(spearmanr(x, y).statistic)


def grouped_permutation_importance(
    estimator: Pipeline,
    test_data: pd.DataFrame,
    feature_columns: list[str],
    groups: dict[str, list[str]],
    actual: np.ndarray,
    seed: int,
) -> dict[str, tuple[float, float]]:
    baseline = mean_absolute_error(actual, estimator.predict(test_data[feature_columns]))
    rng = np.random.default_rng(seed)
    results: dict[str, tuple[float, float]] = {}
    base = test_data[feature_columns].copy()
    for group, columns in groups.items():
        increases: list[float] = []
        for _ in range(PERMUTATION_REPEATS):
            permuted = base.copy()
            order = rng.permutation(len(permuted))
            permuted.loc[:, columns] = base[columns].to_numpy()[order]
            permuted_mae = mean_absolute_error(actual, estimator.predict(permuted))
            increases.append(float(permuted_mae - baseline))
        results[group] = (float(np.mean(increases)), float(np.std(increases, ddof=1)))
    return results


def observed_directions(features: pd.DataFrame) -> dict[tuple[str, str], float]:
    output: dict[tuple[str, str], float] = {}
    for analyte in ANALYTES:
        data = features.loc[features["analyte"] == analyte].copy()
        for group in ["R", "G", "B", "Hue", "Saturation", "Value"]:
            values = group_feature_values(data, group)
            data[f"_{group}_value"] = values
            medians = (
                data.groupby("concentration_mg_ml", as_index=False)[f"_{group}_value"]
                .median()
                .sort_values("concentration_mg_ml")
            )
            output[(analyte, group)] = safe_spearman(
                medians["concentration_mg_ml"].to_numpy(dtype=float),
                medians[f"_{group}_value"].to_numpy(dtype=float),
            )
    return output


def plot_importance(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    colors = {"TreeSHAP": "#3678b5", "Permutation": "#e08c3e"}
    for row_index, analyte in enumerate(ANALYTES):
        for column_index, feature_space in enumerate(FEATURE_SPACES):
            axis = axes[row_index, column_index]
            subset = summary.loc[
                (summary["analyte"] == analyte)
                & (summary["feature_space"] == feature_space)
            ].sort_values("shap_share_mean", ascending=True)
            labels = subset["feature_group"].tolist()
            y = np.arange(len(labels), dtype=float)
            height = 0.34
            axis.barh(
                y + height / 2,
                100 * subset["shap_share_mean"],
                xerr=100 * subset["shap_share_std"].fillna(0),
                height=height,
                color=colors["TreeSHAP"],
                label="TreeSHAP",
                capsize=3,
            )
            axis.barh(
                y - height / 2,
                100 * subset["permutation_share_mean"],
                xerr=100 * subset["permutation_share_std"].fillna(0),
                height=height,
                color=colors["Permutation"],
                label="Permutation",
                capsize=3,
            )
            axis.set_yticks(y, labels)
            axis.set_xlim(left=0)
            axis.grid(axis="x", alpha=0.25)
            axis.set_title(f"{ANALYTE_LABELS[analyte]} · {SPACE_LABELS[feature_space]}")
            axis.set_xlabel("Relative importance within feature space (%)")
            if row_index == 0 and column_index == 0:
                axis.legend(frameon=False, loc="lower right")
    fig.suptitle(
        "Color-channel importance on held-out outer folds",
        fontsize=18,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_shap_direction(values: pd.DataFrame, summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    cmap = plt.get_cmap("coolwarm")
    rng = np.random.default_rng(RANDOM_SEED)
    for row_index, analyte in enumerate(ANALYTES):
        for column_index, feature_space in enumerate(FEATURE_SPACES):
            axis = axes[row_index, column_index]
            subset = values.loc[
                (values["analyte"] == analyte)
                & (values["feature_space"] == feature_space)
            ].copy()
            order = (
                summary.loc[
                    (summary["analyte"] == analyte)
                    & (summary["feature_space"] == feature_space)
                ]
                .sort_values("shap_share_mean", ascending=False)["feature_group"]
                .tolist()
            )
            for y_index, group in enumerate(order):
                group_data = subset.loc[subset["feature_group"] == group].copy()
                feature = group_data["feature_value"].to_numpy(dtype=float)
                low, high = np.nanpercentile(feature, [2, 98])
                if high <= low:
                    color_value = np.full(len(feature), 0.5)
                else:
                    color_value = np.clip((feature - low) / (high - low), 0, 1)
                jitter = rng.normal(0, 0.075, size=len(group_data))
                axis.scatter(
                    group_data["normalized_shap_percent"],
                    y_index + jitter,
                    c=color_value,
                    cmap=cmap,
                    norm=Normalize(0, 1),
                    s=10,
                    alpha=0.38,
                    linewidths=0,
                    rasterized=True,
                )
            axis.axvline(0, color="#333333", linewidth=1)
            axis.set_yticks(np.arange(len(order)), order)
            axis.invert_yaxis()
            axis.grid(axis="x", alpha=0.2)
            axis.set_title(f"{ANALYTE_LABELS[analyte]} · {SPACE_LABELS[feature_space]}")
            axis.set_xlabel("SHAP contribution (% of concentration range)")
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
        ax=axes,
        fraction=0.018,
        pad=0.02,
    )
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["Low channel value", "High channel value"])
    fig.suptitle(
        "Direction of color-channel contributions on held-out outer folds",
        fontsize=18,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    prepare_output_dir(args.output_dir, args.overwrite)
    features = pd.read_csv(args.features)
    feature_rows = pd.read_csv(args.feature_sets)
    feature_sets = (
        feature_rows.groupby("feature_set", sort=False)["feature"].apply(list).to_dict()
    )
    splits = pd.read_csv(args.splits)
    observed = observed_directions(features)

    fold_rows: list[dict[str, object]] = []
    value_rows: list[pd.DataFrame] = []
    performance_rows: list[dict[str, object]] = []
    for analyte in ANALYTES:
        analyte_max = float(
            features.loc[features["analyte"] == analyte, "concentration_mg_ml"].max()
        )
        model_name = MODEL_BY_ANALYTE[analyte]
        for feature_space in FEATURE_SPACES:
            selected_features = feature_sets[feature_space]
            groups = GROUPS[feature_space]
            for outer_fold in range(1, N_SPLITS + 1):
                fold_split = splits.loc[
                    (splits["analyte"] == analyte)
                    & (splits["outer_fold"] == outer_fold)
                ]
                train_assignments = fold_split.loc[
                    fold_split["ml_role"] == "train", ["patch_id", "inner_fold"]
                ]
                test_assignments = fold_split.loc[
                    fold_split["ml_role"] == "test", ["patch_id"]
                ]
                train_data = train_assignments.merge(
                    features, on="patch_id", how="left", validate="one_to_one"
                )
                test_data = test_assignments.merge(
                    features, on="patch_id", how="left", validate="one_to_one"
                )
                search = build_search(model_name, inner_cv_indices(train_data))
                search.fit(
                    train_data[selected_features],
                    train_data["concentration_mg_ml"],
                )
                estimator = search.best_estimator_
                actual = test_data["concentration_mg_ml"].to_numpy(dtype=float)
                prediction_raw = estimator.predict(test_data[selected_features])
                prediction = np.clip(prediction_raw, 0.0, analyte_max)
                performance_rows.append(
                    {
                        "analyte": analyte,
                        "feature_space": feature_space,
                        "model": model_name,
                        "target_transform": "raw",
                        "outer_fold": outer_fold,
                        "mae": mean_absolute_error(actual, prediction),
                        "rmse": math.sqrt(mean_squared_error(actual, prediction)),
                        "r2": r2_score(actual, prediction),
                        "best_inner_mae": -float(search.best_score_),
                        "best_parameters": json.dumps(
                            {
                                key.replace("model__", ""): value
                                for key, value in search.best_params_.items()
                            }
                        ),
                    }
                )

                scaler = estimator.named_steps["scale"]
                tree_model = estimator.named_steps["model"]
                scaled_test = scaler.transform(test_data[selected_features])
                explainer = shap.TreeExplainer(tree_model)
                shap_values = np.asarray(
                    explainer.shap_values(scaled_test, check_additivity=True),
                    dtype=float,
                )
                expected_value = float(np.ravel(explainer.expected_value)[0])
                additivity_error = float(
                    np.max(np.abs(expected_value + shap_values.sum(axis=1) - prediction_raw))
                )
                if additivity_error > 1e-8:
                    raise RuntimeError(
                        f"SHAP additivity failed for {analyte} {feature_space} "
                        f"fold {outer_fold}: {additivity_error}"
                    )
                permutation = grouped_permutation_importance(
                    estimator,
                    test_data,
                    selected_features,
                    groups,
                    actual,
                    RANDOM_SEED + outer_fold + (0 if analyte == "glucose" else 100),
                )
                group_mean_abs: dict[str, float] = {}
                group_shap: dict[str, np.ndarray] = {}
                for group, columns in groups.items():
                    indices = [selected_features.index(column) for column in columns]
                    signed = shap_values[:, indices].sum(axis=1)
                    group_shap[group] = signed
                    group_mean_abs[group] = float(np.mean(np.abs(signed)))
                shap_total = sum(group_mean_abs.values())
                permutation_positive = {
                    group: max(0.0, permutation[group][0]) for group in groups
                }
                permutation_total = sum(permutation_positive.values())
                for group in groups:
                    feature_values = group_feature_values(test_data, group)
                    signed = group_shap[group]
                    direction_rho = safe_spearman(feature_values, signed)
                    observed_rho = observed[(analyte, group)]
                    aligned = (
                        math.nan
                        if math.isnan(direction_rho) or math.isnan(observed_rho)
                        else bool(np.sign(direction_rho) == np.sign(observed_rho))
                    )
                    fold_rows.append(
                        {
                            "analyte": analyte,
                            "feature_space": feature_space,
                            "model": model_name,
                            "target_transform": "raw",
                            "outer_fold": outer_fold,
                            "feature_group": group,
                            "mean_absolute_shap": group_mean_abs[group],
                            "shap_share": group_mean_abs[group] / shap_total,
                            "permutation_mae_increase": permutation[group][0],
                            "permutation_mae_increase_std": permutation[group][1],
                            "permutation_share": (
                                permutation_positive[group] / permutation_total
                                if permutation_total > 0
                                else 0.0
                            ),
                            "shap_direction_rho": direction_rho,
                            "observed_concentration_rho": observed_rho,
                            "direction_aligned": aligned,
                            "shap_additivity_max_error": additivity_error,
                        }
                    )
                    group_frame = pd.DataFrame(
                        {
                            "analyte": analyte,
                            "feature_space": feature_space,
                            "model": model_name,
                            "outer_fold": outer_fold,
                            "patch_id": test_data["patch_id"].to_numpy(),
                            "actual_concentration": actual,
                            "prediction": prediction,
                            "feature_group": group,
                            "feature_value": feature_values,
                            "shap_value": signed,
                            "normalized_shap_percent": 100 * signed / analyte_max,
                        }
                    )
                    value_rows.append(group_frame)
                print(
                    f"Explained {analyte} {feature_space} fold={outer_fold}",
                    flush=True,
                )

    folds = pd.DataFrame(fold_rows)
    values = pd.concat(value_rows, ignore_index=True)
    performance = pd.DataFrame(performance_rows)
    rank_frame = folds.copy()
    rank_frame["shap_rank"] = rank_frame.groupby(
        ["analyte", "feature_space", "outer_fold"]
    )["mean_absolute_shap"].rank(method="min", ascending=False)
    summary = (
        rank_frame.groupby(["analyte", "feature_space", "model", "feature_group"])
        .agg(
            shap_share_mean=("shap_share", "mean"),
            shap_share_std=("shap_share", "std"),
            permutation_share_mean=("permutation_share", "mean"),
            permutation_share_std=("permutation_share", "std"),
            permutation_mae_increase_mean=("permutation_mae_increase", "mean"),
            shap_direction_rho_mean=("shap_direction_rho", "mean"),
            shap_direction_rho_min=("shap_direction_rho", "min"),
            shap_direction_rho_max=("shap_direction_rho", "max"),
            observed_concentration_rho=("observed_concentration_rho", "first"),
            top_rank_fold_count=("shap_rank", lambda item: int(np.sum(item == 1))),
            direction_aligned_fold_count=(
                "direction_aligned",
                lambda item: int(np.sum(item.astype(bool))),
            ),
        )
        .reset_index()
    )
    folds.to_csv(args.output_dir / "ml_explainability_fold_results.csv", index=False)
    values.to_csv(args.output_dir / "ml_grouped_shap_values.csv", index=False)
    performance.to_csv(args.output_dir / "ml_explainability_model_performance.csv", index=False)
    summary.to_csv(args.output_dir / "ml_explainability_summary.csv", index=False)
    plot_importance(summary, args.output_dir / "figures" / "color_channel_importance.png")
    plot_shap_direction(values, summary, args.output_dir / "figures" / "shap_direction_summary.png")
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "random_seed": RANDOM_SEED,
                "outer_folds": N_SPLITS,
                "trees": N_TREES,
                "permutation_repeats": PERMUTATION_REPEATS,
                "target_transform": "raw for SHAP values in concentration units",
                "feature_spaces": FEATURE_SPACES,
                "model_by_analyte": MODEL_BY_ANALYTE,
                "hue_grouping": "h_sin_weighted and h_cos_weighted summed as one additive group",
                "evaluation_data": "held-out outer test patches only",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

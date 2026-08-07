"""Validate extracted color features before model training.

This stage treats the 96 wells in each source image as technical patch-level
observations. Concentration trends are summarized at source-image level,
position effects are calculated after within-image robust centering, and the
white substrate annulus is evaluated as a negative-control feature group.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


RANDOM_SEED = 240920
BOOTSTRAP_ITERATIONS = 5000

RGB_FEATURES = ["r_median", "g_median", "b_median"]
HSV_FEATURES = [
    "h_sin_weighted",
    "h_cos_weighted",
    "s_median",
    "v_median",
]
CHROMATICITY_FEATURES = [
    "r_chromaticity_median",
    "g_chromaticity_median",
    "b_chromaticity_median",
]
BACKGROUND_FEATURES = ["r_bg_median", "g_bg_median", "b_bg_median"]
POSITION_FEATURES = [
    "r_median",
    "g_median",
    "b_median",
    "s_median",
    "v_median",
    "intensity_median",
]
SUMMARY_FEATURES = (
    RGB_FEATURES
    + [
        "h_sin_weighted",
        "h_cos_weighted",
        "s_median",
        "v_median",
        "hue_unwrapped_deg",
    ]
    + CHROMATICITY_FEATURES
    + ["intensity_median"]
    + BACKGROUND_FEATURES
)
ADJACENT_FEATURES = RGB_FEATURES + ["s_median", "v_median", "hue_unwrapped_deg"]

FEATURE_LABELS = {
    "r_median": "R median",
    "g_median": "G median",
    "b_median": "B median",
    "s_median": "S median",
    "v_median": "V median",
    "h_sin_weighted": "sin(H)",
    "h_cos_weighted": "cos(H)",
    "hue_unwrapped_deg": "Hue (unwrapped degree)",
    "r_chromaticity_median": "r chromaticity",
    "g_chromaticity_median": "g chromaticity",
    "b_chromaticity_median": "b chromaticity",
    "intensity_median": "Intensity median",
    "r_bg_median": "Background R",
    "g_bg_median": "Background G",
    "b_bg_median": "Background B",
}

ANALYTE_LABELS = {"glucose": "Glucose", "ketone": "Ketone"}
CHANNEL_COLORS = {
    "r_median": "#C74343",
    "g_median": "#3D8F5B",
    "b_median": "#3D6FB6",
    "s_median": "#D38A2E",
    "v_median": "#526D82",
    "hue_unwrapped_deg": "#8C5AA6",
    "r_bg_median": "#C74343",
    "g_bg_median": "#3D8F5B",
    "b_bg_median": "#3D6FB6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("outputs/color_features/features.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/feature_validation"),
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
    path.mkdir(parents=True)
    (path / "figures").mkdir()


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Malgun Gothic", "Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.8g")


def concentration_label(value: float) -> str:
    return f"{value:g}"


def add_unwrapped_hue(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = features.copy()
    output["hue_unwrapped_deg"] = np.nan
    image_rows: list[dict[str, object]] = []

    for analyte, analyte_frame in output.groupby("analyte", sort=False):
        ordered_images = (
            analyte_frame[
                ["image_id", "concentration_order", "concentration_mg_ml"]
            ]
            .drop_duplicates()
            .sort_values("concentration_order")
        )
        raw_angles: list[float] = []
        for image_id in ordered_images["image_id"]:
            group = analyte_frame.loc[analyte_frame["image_id"] == image_id]
            sin_value = float(group["h_sin_weighted"].median())
            cos_value = float(group["h_cos_weighted"].median())
            raw_angles.append(math.degrees(math.atan2(sin_value, cos_value)) % 360.0)

        unwrapped_angles = np.degrees(np.unwrap(np.radians(raw_angles)))
        for (_, image_meta), raw_angle, unwrapped_angle in zip(
            ordered_images.iterrows(), raw_angles, unwrapped_angles, strict=True
        ):
            image_id = str(image_meta["image_id"])
            mask = output["image_id"] == image_id
            patch_angles = (
                np.degrees(
                    np.arctan2(
                        output.loc[mask, "h_sin_weighted"].to_numpy(),
                        output.loc[mask, "h_cos_weighted"].to_numpy(),
                    )
                )
                % 360.0
            )
            local_delta = (patch_angles - raw_angle + 180.0) % 360.0 - 180.0
            output.loc[mask, "hue_unwrapped_deg"] = unwrapped_angle + local_delta
            image_rows.append(
                {
                    "image_id": image_id,
                    "analyte": analyte,
                    "concentration_order": int(image_meta["concentration_order"]),
                    "concentration_mg_ml": float(image_meta["concentration_mg_ml"]),
                    "hue_circular_mean_deg": raw_angle,
                    "hue_unwrapped_deg": float(unwrapped_angle),
                }
            )

    return output, pd.DataFrame(image_rows)


def summarize_concentrations(features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouping = [
        "analyte",
        "image_id",
        "concentration_order",
        "concentration_mg_ml",
    ]
    for keys, group in features.groupby(grouping, sort=True):
        analyte, image_id, order, concentration = keys
        for feature in SUMMARY_FEATURES:
            values = group[feature].to_numpy(dtype=float)
            q1, median, q3 = np.percentile(values, [25, 50, 75])
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1))
            iqr = float(q3 - q1)
            rows.append(
                {
                    "analyte": analyte,
                    "image_id": image_id,
                    "concentration_order": int(order),
                    "concentration_mg_ml": float(concentration),
                    "feature": feature,
                    "feature_label": FEATURE_LABELS[feature],
                    "n_patches": int(values.size),
                    "mean": mean,
                    "std": std,
                    "q1": float(q1),
                    "median": float(median),
                    "q3": float(q3),
                    "iqr": iqr,
                    "cv_std_over_abs_mean": (
                        std / abs(mean) if abs(mean) > np.finfo(float).eps else np.nan
                    ),
                    "robust_cv_iqr_over_abs_median": (
                        iqr / abs(median)
                        if abs(median) > np.finfo(float).eps
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_trends(concentration_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (analyte, feature), group in concentration_summary.groupby(
        ["analyte", "feature"], sort=True
    ):
        group = group.sort_values("concentration_order")
        orders = group["concentration_order"].to_numpy(dtype=float)
        values = group["median"].to_numpy(dtype=float)
        rho, p_value = spearmanr(orders, values)
        differences = np.diff(values)
        endpoint_change = float(values[-1] - values[0])
        endpoint_direction = float(np.sign(endpoint_change))
        if differences.size and endpoint_direction != 0:
            consistent = np.mean(
                (np.sign(differences) == endpoint_direction) | (differences == 0)
            )
        else:
            consistent = np.nan
        total_path = float(np.sum(np.abs(differences)))
        last_step_fraction = (
            float(abs(differences[-1]) / total_path)
            if differences.size and total_path > 0
            else np.nan
        )
        rows.append(
            {
                "analyte": analyte,
                "feature": feature,
                "feature_label": FEATURE_LABELS[feature],
                "n_concentrations": int(group.shape[0]),
                "spearman_rho_vs_concentration_order": float(rho),
                "spearman_p_descriptive": float(p_value),
                "endpoint_change_max_minus_zero": endpoint_change,
                "monotonic_adjacent_fraction": float(consistent),
                "last_step_fraction_of_total_path": last_step_fraction,
                "interpretation_scope": "source-image-level descriptive trend",
            }
        )
    return pd.DataFrame(rows)


def summarize_adjacent_overlap(features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for analyte, analyte_frame in features.groupby("analyte", sort=True):
        concentrations = (
            analyte_frame[["concentration_order", "concentration_mg_ml"]]
            .drop_duplicates()
            .sort_values("concentration_order")
        )
        concentration_records = list(concentrations.itertuples(index=False))
        for feature in ADJACENT_FEATURES:
            for first, second in zip(
                concentration_records[:-1], concentration_records[1:], strict=True
            ):
                first_values = analyte_frame.loc[
                    analyte_frame["concentration_order"] == first.concentration_order,
                    feature,
                ].to_numpy(dtype=float)
                second_values = analyte_frame.loc[
                    analyte_frame["concentration_order"] == second.concentration_order,
                    feature,
                ].to_numpy(dtype=float)
                first_q1, first_median, first_q3 = np.percentile(
                    first_values, [25, 50, 75]
                )
                second_q1, second_median, second_q3 = np.percentile(
                    second_values, [25, 50, 75]
                )
                overlap_width = max(
                    0.0, min(first_q3, second_q3) - max(first_q1, second_q1)
                )
                union_width = max(first_q3, second_q3) - min(first_q1, second_q1)
                mean_iqr = ((first_q3 - first_q1) + (second_q3 - second_q1)) / 2
                rows.append(
                    {
                        "analyte": analyte,
                        "feature": feature,
                        "feature_label": FEATURE_LABELS[feature],
                        "concentration_1_mg_ml": float(first.concentration_mg_ml),
                        "concentration_2_mg_ml": float(second.concentration_mg_ml),
                        "median_1": float(first_median),
                        "median_2": float(second_median),
                        "absolute_median_gap": float(
                            abs(second_median - first_median)
                        ),
                        "median_gap_over_mean_iqr": (
                            float(abs(second_median - first_median) / mean_iqr)
                            if mean_iqr > 0
                            else np.nan
                        ),
                        "iqr_overlap_over_union": (
                            float(overlap_width / union_width)
                            if union_width > 0
                            else np.nan
                        ),
                        "interpretation_scope": "technical-patch distribution overlap",
                    }
                )
    return pd.DataFrame(rows)


def bootstrap_median_ci(
    values: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan, np.nan
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_ITERATIONS, values.size))
    medians = np.median(values[indices], axis=1)
    lower, upper = np.percentile(medians, [2.5, 97.5])
    return float(lower), float(upper)


def calculate_position_bias(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    residual_rows: list[pd.DataFrame] = []
    for image_id, group in features.groupby("image_id", sort=True):
        base = group[
            [
                "patch_id",
                "image_id",
                "analyte",
                "concentration_order",
                "concentration_mg_ml",
                "well_id",
                "grid_row",
                "grid_col",
            ]
        ].copy()
        for feature in POSITION_FEATURES:
            values = group[feature].to_numpy(dtype=float)
            q1, median, q3 = np.percentile(values, [25, 50, 75])
            scale = float(q3 - q1)
            if scale <= np.finfo(float).eps:
                scale = float(np.std(values, ddof=1))
            if scale <= np.finfo(float).eps:
                base[f"{feature}_residual_iqr"] = 0.0
            else:
                base[f"{feature}_residual_iqr"] = (values - median) / scale
        residual_rows.append(base)
    residuals = pd.concat(residual_rows, ignore_index=True)

    well_rows: list[dict[str, object]] = []
    bias_rows: list[dict[str, object]] = []
    edge_mask = (
        (residuals["grid_row"].isin([1, 8]))
        | (residuals["grid_col"].isin([1, 12]))
    )

    for analyte, analyte_frame in residuals.groupby("analyte", sort=True):
        for feature in POSITION_FEATURES:
            residual_column = f"{feature}_residual_iqr"
            well_summary = (
                analyte_frame.groupby(
                    ["well_id", "grid_row", "grid_col"], sort=True
                )[residual_column]
                .median()
                .reset_index(name="median_residual_iqr_units")
            )
            for row in well_summary.itertuples(index=False):
                well_rows.append(
                    {
                        "analyte": analyte,
                        "feature": feature,
                        "feature_label": FEATURE_LABELS[feature],
                        "well_id": row.well_id,
                        "grid_row": int(row.grid_row),
                        "grid_col": int(row.grid_col),
                        "median_residual_iqr_units": float(
                            row.median_residual_iqr_units
                        ),
                        "n_source_images": int(
                            analyte_frame["image_id"].nunique()
                        ),
                    }
                )

            image_effects: list[float] = []
            for _, image_group in analyte_frame.groupby("image_id", sort=True):
                image_edge = (
                    (image_group["grid_row"].isin([1, 8]))
                    | (image_group["grid_col"].isin([1, 12]))
                )
                edge_value = float(image_group.loc[image_edge, residual_column].median())
                interior_value = float(
                    image_group.loc[~image_edge, residual_column].median()
                )
                image_effects.append(edge_value - interior_value)
            effects = np.asarray(image_effects, dtype=float)
            ci_lower, ci_upper = bootstrap_median_ci(effects, rng)

            row_values = (
                well_summary.groupby("grid_row")["median_residual_iqr_units"]
                .median()
                .sort_index()
            )
            col_values = (
                well_summary.groupby("grid_col")["median_residual_iqr_units"]
                .median()
                .sort_index()
            )
            row_rho, _ = spearmanr(row_values.index, row_values.values)
            col_rho, _ = spearmanr(col_values.index, col_values.values)
            max_abs = float(
                well_summary["median_residual_iqr_units"].abs().max()
            )
            bias_rows.append(
                {
                    "analyte": analyte,
                    "feature": feature,
                    "feature_label": FEATURE_LABELS[feature],
                    "n_source_images": int(analyte_frame["image_id"].nunique()),
                    "median_edge_minus_interior_iqr_units": float(
                        np.median(effects)
                    ),
                    "bootstrap_95ci_lower": ci_lower,
                    "bootstrap_95ci_upper": ci_upper,
                    "max_abs_well_median_iqr_units": max_abs,
                    "row_spearman_rho_descriptive": float(row_rho),
                    "column_spearman_rho_descriptive": float(col_rho),
                    "interpretation_scope": "within-image centered position effect",
                }
            )

    return residuals, pd.DataFrame(well_rows), pd.DataFrame(bias_rows)


def summarize_feature_correlations(
    concentration_summary: pd.DataFrame,
) -> pd.DataFrame:
    primary_features = RGB_FEATURES + HSV_FEATURES
    rows: list[dict[str, object]] = []
    for analyte, group in concentration_summary.groupby("analyte", sort=True):
        wide = group.loc[group["feature"].isin(primary_features)].pivot(
            index="image_id", columns="feature", values="median"
        )
        correlation = wide[primary_features].corr(method="spearman")
        for first in primary_features:
            for second in primary_features:
                rows.append(
                    {
                        "analyte": analyte,
                        "feature_1": first,
                        "feature_2": second,
                        "spearman_rho_image_medians": float(
                            correlation.loc[first, second]
                        ),
                        "n_source_images": int(wide.shape[0]),
                    }
                )
    return pd.DataFrame(rows)


def feature_set_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(set_name: str, role: str, features: list[str], rationale: str) -> None:
        for feature in features:
            rows.append(
                {
                    "feature_set": set_name,
                    "analysis_role": role,
                    "feature": feature,
                    "rationale": rationale,
                }
            )

    add(
        "RGB_primary",
        "primary",
        RGB_FEATURES,
        "robust central tendency with direct channel interpretation",
    )
    add(
        "HSV_primary",
        "primary",
        HSV_FEATURES,
        "circular Hue representation plus saturation and brightness",
    )
    add(
        "RGB_HSV_combined",
        "secondary",
        RGB_FEATURES + HSV_FEATURES,
        "correlated color spaces; secondary comparison only",
    )
    add(
        "Chromaticity_secondary",
        "secondary",
        ["r_chromaticity_median", "g_chromaticity_median"],
        "two chromaticity components; B omitted to reduce compositional redundancy",
    )
    add(
        "Background_adjusted_secondary",
        "secondary",
        ["r_delta_bg", "g_delta_bg", "b_delta_bg"],
        "sensitivity analysis for local illumination adjustment",
    )
    add(
        "Background_negative_control",
        "negative_control",
        BACKGROUND_FEATURES,
        "tests whether the substrate alone carries concentration-associated image information",
    )
    add(
        "Texture_QC_secondary",
        "secondary",
        ["r_iqr", "g_iqr", "b_iqr", "s_iqr", "v_iqr"],
        "patch heterogeneity; excluded from parsimonious primary models",
    )
    return pd.DataFrame(rows)


def draw_boxplot_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    feature: str,
    analyte: str,
    color: str,
) -> None:
    ordered = (
        frame[["concentration_order", "concentration_mg_ml"]]
        .drop_duplicates()
        .sort_values("concentration_order")
    )
    arrays = [
        frame.loc[frame["concentration_order"] == row.concentration_order, feature]
        .dropna()
        .to_numpy(dtype=float)
        for row in ordered.itertuples(index=False)
    ]
    positions = np.arange(len(arrays))
    box = ax.boxplot(
        arrays,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1F1F1F", "linewidth": 1.2},
        whiskerprops={"color": "#777777", "linewidth": 0.8},
        capprops={"color": "#777777", "linewidth": 0.8},
    )
    for patch in box["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor(color)
    medians = [float(np.median(values)) for values in arrays]
    ax.plot(positions, medians, color=color, marker="o", linewidth=1.6, markersize=3)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [concentration_label(value) for value in ordered["concentration_mg_ml"]],
        rotation=45,
        ha="right",
    )
    ax.set_title(f"{ANALYTE_LABELS[analyte]} · {FEATURE_LABELS[feature]}")
    ax.set_xlabel("Concentration (mg/mL)")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)


def plot_concentration_panels(
    features: pd.DataFrame,
    feature_group: list[str],
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, len(feature_group), figsize=(14, 7.2), squeeze=False)
    for row_index, analyte in enumerate(("glucose", "ketone")):
        frame = features.loc[features["analyte"] == analyte]
        for column_index, feature in enumerate(feature_group):
            draw_boxplot_panel(
                axes[row_index, column_index],
                frame,
                feature,
                analyte,
                CHANNEL_COLORS[feature],
            )
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    fig.text(
        0.5,
        0.005,
        "Boxes describe 96 technical patches in one source image per concentration.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_variability(summary: pd.DataFrame, output_path: Path) -> None:
    features = ["r_median", "g_median", "b_median", "s_median", "v_median"]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    positions = np.arange(len(features), dtype=float)
    offsets = {"glucose": -0.18, "ketone": 0.18}
    colors = {"glucose": "#3B82A0", "ketone": "#B7658B"}
    for analyte in ("glucose", "ketone"):
        arrays = [
            summary.loc[
                (summary["analyte"] == analyte) & (summary["feature"] == feature),
                "robust_cv_iqr_over_abs_median",
            ]
            .dropna()
            .to_numpy(dtype=float)
            for feature in features
        ]
        box = ax.boxplot(
            arrays,
            positions=positions + offsets[analyte],
            widths=0.30,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "#202020", "linewidth": 1.2},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(colors[analyte])
            patch.set_alpha(0.55)
            patch.set_edgecolor(colors[analyte])
        ax.plot([], [], color=colors[analyte], linewidth=8, alpha=0.55, label=ANALYTE_LABELS[analyte])
    ax.set_xticks(positions)
    ax.set_xticklabels([FEATURE_LABELS[feature] for feature in features])
    ax.set_ylabel("Within-image IQR / |median|")
    ax.set_title("Patch-to-patch variability within each source image")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    fig.text(
        0.5,
        0.01,
        "Each point represents one source image; variation is technical within-image variation.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_position_heatmaps(well_summary: pd.DataFrame, output_path: Path) -> None:
    display_features = ["r_median", "g_median", "b_median", "v_median"]
    subset = well_summary.loc[well_summary["feature"].isin(display_features)]
    limit = float(
        max(
            0.5,
            np.percentile(
                np.abs(subset["median_residual_iqr_units"].to_numpy(dtype=float)),
                98,
            ),
        )
    )
    fig, axes = plt.subplots(2, 4, figsize=(15.2, 6.7), squeeze=False)
    image = None
    for row_index, analyte in enumerate(("glucose", "ketone")):
        for column_index, feature in enumerate(display_features):
            ax = axes[row_index, column_index]
            frame = subset.loc[
                (subset["analyte"] == analyte) & (subset["feature"] == feature)
            ]
            matrix = frame.pivot(
                index="grid_row",
                columns="grid_col",
                values="median_residual_iqr_units",
            ).sort_index(ascending=True)
            image = ax.imshow(
                matrix.to_numpy(),
                cmap="RdBu_r",
                norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
                aspect="auto",
            )
            ax.set_title(f"{ANALYTE_LABELS[analyte]} · {FEATURE_LABELS[feature]}")
            ax.set_xticks(np.arange(12))
            ax.set_xticklabels(np.arange(1, 13))
            ax.set_yticks(np.arange(8))
            ax.set_yticklabels(list("ABCDEFGH"))
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")
    fig.suptitle(
        "Sensor-array position bias after removing each image's concentration level",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    fig.subplots_adjust(left=0.05, right=0.90, bottom=0.08, top=0.90, wspace=0.28, hspace=0.38)
    if image is not None:
        colorbar_axis = fig.add_axes([0.925, 0.16, 0.014, 0.67])
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Median within-image residual (IQR units)")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_background_control(
    concentration_summary: pd.DataFrame, output_path: Path
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.8), squeeze=False)
    for row_index, analyte in enumerate(("glucose", "ketone")):
        analyte_frame = concentration_summary.loc[
            concentration_summary["analyte"] == analyte
        ]
        for column_index, feature in enumerate(BACKGROUND_FEATURES):
            ax = axes[row_index, column_index]
            frame = analyte_frame.loc[analyte_frame["feature"] == feature].sort_values(
                "concentration_order"
            )
            positions = np.arange(frame.shape[0])
            ax.plot(
                positions,
                frame["median"],
                color=CHANNEL_COLORS[feature],
                marker="o",
                linewidth=1.5,
            )
            ax.set_xticks(positions)
            ax.set_xticklabels(
                [concentration_label(value) for value in frame["concentration_mg_ml"]],
                rotation=45,
                ha="right",
            )
            ax.set_title(f"{ANALYTE_LABELS[analyte]} · {FEATURE_LABELS[feature]}")
            ax.set_xlabel("Concentration (mg/mL)")
            ax.set_ylabel("Background intensity (0-255)")
            ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    fig.suptitle(
        "White-substrate RGB as a negative control",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        0.005,
        "A concentration trend here indicates image-level lighting/background information, not sensor chemistry.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_feature_correlations(correlations: pd.DataFrame, output_path: Path) -> None:
    features = RGB_FEATURES + HSV_FEATURES
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), squeeze=False)
    for index, analyte in enumerate(("glucose", "ketone")):
        ax = axes[0, index]
        frame = correlations.loc[correlations["analyte"] == analyte]
        matrix = frame.pivot(
            index="feature_1", columns="feature_2", values="spearman_rho_image_medians"
        ).loc[features, features]
        image = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(features)))
        ax.set_xticklabels([FEATURE_LABELS[value] for value in features], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(features)))
        ax.set_yticklabels([FEATURE_LABELS[value] for value in features])
        ax.set_title(f"{ANALYTE_LABELS[analyte]} (source-image medians)")
        for row in range(len(features)):
            for column in range(len(features)):
                value = matrix.iloc[row, column]
                ax.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(value) > 0.55 else "#202020",
                )
    fig.suptitle("Redundancy among primary color features", fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.89, bottom=0.22, top=0.84, wspace=0.32)
    colorbar_axis = fig.add_axes([0.92, 0.22, 0.015, 0.58])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Spearman rho")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def json_safe(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def build_validation_summary(
    features: pd.DataFrame,
    trends: pd.DataFrame,
    position_bias: pd.DataFrame,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "random_seed": RANDOM_SEED,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "patch_rows": int(features.shape[0]),
        "source_images": int(features["image_id"].nunique()),
        "qc_status_counts": features["qc_status"].value_counts().to_dict(),
        "interpretation": {
            "patches": "technical observations within source images",
            "concentration_trends": "source-image-level descriptive associations",
            "position_bias": "within-image centered residuals in IQR units",
            "background": "negative control only; excluded from primary models",
        },
        "analytes": {},
    }
    for analyte in ("glucose", "ketone"):
        analyte_trends = trends.loc[trends["analyte"] == analyte]
        sensor_trends = analyte_trends.loc[
            ~analyte_trends["feature"].isin(BACKGROUND_FEATURES)
        ].copy()
        sensor_trends["abs_rho"] = sensor_trends[
            "spearman_rho_vs_concentration_order"
        ].abs()
        background_trends = analyte_trends.loc[
            analyte_trends["feature"].isin(BACKGROUND_FEATURES)
        ].copy()
        background_trends["abs_rho"] = background_trends[
            "spearman_rho_vs_concentration_order"
        ].abs()
        position = position_bias.loc[position_bias["analyte"] == analyte].copy()
        position = position.sort_values(
            "max_abs_well_median_iqr_units", ascending=False
        )
        summary["analytes"][analyte] = {
            "source_images": int(
                features.loc[features["analyte"] == analyte, "image_id"].nunique()
            ),
            "strongest_sensor_trends": sensor_trends.nlargest(5, "abs_rho")[
                ["feature", "spearman_rho_vs_concentration_order"]
            ].to_dict("records"),
            "strongest_background_trend": background_trends.nlargest(1, "abs_rho")[
                ["feature", "spearman_rho_vs_concentration_order"]
            ].to_dict("records"),
            "largest_position_effect": position.head(1)[
                [
                    "feature",
                    "max_abs_well_median_iqr_units",
                    "median_edge_minus_interior_iqr_units",
                ]
            ].to_dict("records"),
        }
    return json_safe(summary)


def main() -> None:
    args = parse_args()
    features_path = args.features.resolve()
    output_dir = args.output_dir.resolve()
    prepare_output_dir(output_dir, args.overwrite)
    configure_plot_style()

    features = pd.read_csv(features_path, encoding="utf-8-sig")
    required = {
        "patch_id",
        "image_id",
        "analyte",
        "concentration_order",
        "concentration_mg_ml",
        "well_id",
        "grid_row",
        "grid_col",
        "qc_status",
        *RGB_FEATURES,
        *HSV_FEATURES,
        *CHROMATICITY_FEATURES,
        *BACKGROUND_FEATURES,
        *POSITION_FEATURES,
    }
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    if features["patch_id"].duplicated().any():
        raise ValueError("Duplicate patch_id values found")

    validation_features, hue_summary = add_unwrapped_hue(features)
    concentration_summary = summarize_concentrations(validation_features)
    trends = summarize_trends(concentration_summary)
    adjacent_overlap = summarize_adjacent_overlap(validation_features)
    residuals, well_summary, position_bias = calculate_position_bias(
        validation_features
    )
    correlations = summarize_feature_correlations(concentration_summary)
    feature_sets = feature_set_table()
    background_control = trends.loc[
        trends["feature"].isin(BACKGROUND_FEATURES)
    ].reset_index(drop=True)
    variability = concentration_summary.loc[
        concentration_summary["feature"].isin(
            ["r_median", "g_median", "b_median", "s_median", "v_median"]
        )
    ].reset_index(drop=True)

    write_csv(concentration_summary, output_dir / "concentration_summary.csv")
    write_csv(trends, output_dir / "concentration_trends.csv")
    write_csv(adjacent_overlap, output_dir / "adjacent_concentration_overlap.csv")
    write_csv(variability, output_dir / "within_image_variability.csv")
    write_csv(hue_summary, output_dir / "hue_unwrapped_summary.csv")
    write_csv(residuals, output_dir / "position_residuals.csv")
    write_csv(well_summary, output_dir / "position_well_summary.csv")
    write_csv(position_bias, output_dir / "position_bias_summary.csv")
    write_csv(background_control, output_dir / "background_control_summary.csv")
    write_csv(correlations, output_dir / "feature_correlation_summary.csv")
    write_csv(feature_sets, output_dir / "model_feature_sets.csv")

    figures = output_dir / "figures"
    plot_concentration_panels(
        validation_features,
        RGB_FEATURES,
        figures / "concentration_trends_rgb.png",
        "RGB distributions across measured concentrations",
    )
    plot_concentration_panels(
        validation_features,
        ["hue_unwrapped_deg", "s_median", "v_median"],
        figures / "concentration_trends_hsv.png",
        "HSV distributions across measured concentrations",
    )
    plot_variability(
        concentration_summary, figures / "within_image_variability.png"
    )
    plot_position_heatmaps(
        well_summary, figures / "position_bias_heatmaps.png"
    )
    plot_background_control(
        concentration_summary, figures / "background_negative_control.png"
    )
    plot_feature_correlations(
        correlations, figures / "feature_redundancy_heatmap.png"
    )

    summary = build_validation_summary(features, trends, position_bias)
    summary["outputs"] = {
        "tables": 11,
        "figures": 6,
        "primary_feature_sets": ["RGB_primary", "HSV_primary"],
        "secondary_feature_sets": [
            "RGB_HSV_combined",
            "Chromaticity_secondary",
            "Background_adjusted_secondary",
            "Texture_QC_secondary",
        ],
        "negative_control": "Background_negative_control",
    }
    with (output_dir / "validation_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

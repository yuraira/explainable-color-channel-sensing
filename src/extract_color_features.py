"""Extract reproducible color features from detected colorimetric patches.

The detector creates lossless PNG crops and a manifest containing each circle's
centre and radius. This script measures a conservative central ROI, excludes
fixed-rule white specular pixels, and records a local substrate annulus as an
optional illumination reference. Raw and background-normalized features remain
separate so downstream experiments can compare them without redefining the ROI.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ROI_RADIUS_FRACTION = 0.70
BACKGROUND_INNER_RADIUS_FRACTION = 1.05
BACKGROUND_OUTER_RADIUS_FRACTION = 1.16
HIGHLIGHT_V_THRESHOLD = 0.96
HIGHLIGHT_S_THRESHOLD = 0.30
DARK_V_THRESHOLD = 0.10
QC_MAX_HIGHLIGHT_FRACTION = 0.15
QC_MIN_VALID_FRACTION = 0.80

METADATA_FIELDS = [
    "patch_id",
    "image_id",
    "analyte",
    "concentration_order",
    "concentration_mg_ml",
    "well_id",
    "grid_row",
    "grid_col",
    "crop_file",
    "circle_radius_px",
]

RGB_STATS_FIELDS = [
    f"{channel}_{stat}"
    for channel in ("r", "g", "b")
    for stat in ("mean", "median", "std", "iqr")
]

HSV_STATS_FIELDS = [
    f"{channel}_{stat}"
    for channel in ("s", "v")
    for stat in ("mean", "median", "std", "iqr")
]

HUE_FIELDS = [
    "h_circular_mean_deg",
    "h_circular_std_deg",
    "h_resultant_length",
    "h_sin_weighted",
    "h_cos_weighted",
]

DERIVED_COLOR_FIELDS = [
    "r_chromaticity_mean",
    "r_chromaticity_median",
    "g_chromaticity_mean",
    "g_chromaticity_median",
    "b_chromaticity_mean",
    "b_chromaticity_median",
    "intensity_mean",
    "intensity_median",
]

BACKGROUND_FIELDS = [
    "r_bg_median",
    "g_bg_median",
    "b_bg_median",
    "r_delta_bg",
    "g_delta_bg",
    "b_delta_bg",
    "r_ratio_bg",
    "g_ratio_bg",
    "b_ratio_bg",
]

QC_FIELDS = [
    "roi_pixel_count",
    "valid_pixel_count",
    "highlight_pixel_count",
    "highlight_fraction",
    "valid_fraction",
    "dark_fraction",
    "background_pixel_count",
    "qc_status",
    "qc_reason",
]

FEATURE_FIELDS = (
    METADATA_FIELDS
    + RGB_STATS_FIELDS
    + HSV_STATS_FIELDS
    + HUE_FIELDS
    + DERIVED_COLOR_FIELDS
    + BACKGROUND_FIELDS
    + QC_FIELDS
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--patch-manifest",
        type=Path,
        default=Path("outputs/patch_detection/patch_manifest.csv"),
    )
    parser.add_argument(
        "--patch-root",
        type=Path,
        default=Path("outputs/patch_detection"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/color_features"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing feature-output directory.",
    )
    return parser.parse_args()


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {path}. Use --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True)
    (path / "qc").mkdir()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def distribution_stats(values: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    q25, q75 = np.percentile(values, [25, 75])
    return (
        float(np.mean(values)),
        float(np.median(values)),
        float(np.std(values, ddof=0)),
        float(q75 - q25),
    )


def circular_hue_stats(
    hue_deg: np.ndarray, saturation: np.ndarray
) -> tuple[float, float, float, float, float]:
    angles = np.deg2rad(np.asarray(hue_deg, dtype=np.float64))
    weights = np.asarray(saturation, dtype=np.float64)
    weight_sum = float(np.sum(weights))
    if weight_sum <= np.finfo(float).eps:
        weights = np.ones_like(weights)
        weight_sum = float(weights.size)

    sin_component = float(np.sum(weights * np.sin(angles)) / weight_sum)
    cos_component = float(np.sum(weights * np.cos(angles)) / weight_sum)
    resultant = float(np.hypot(sin_component, cos_component))
    mean_deg = float(np.degrees(np.arctan2(sin_component, cos_component)) % 360.0)
    clipped_resultant = float(np.clip(resultant, np.finfo(float).eps, 1.0))
    circular_std_deg = float(
        np.degrees(np.sqrt(-2.0 * np.log(clipped_resultant)))
    )
    return mean_deg, circular_std_deg, resultant, sin_component, cos_component


def rounded(value: float) -> float:
    return round(float(value), 6)


def extract_one(
    manifest_row: dict[str, str], patch_root: Path
) -> tuple[dict[str, object], np.ndarray, dict[str, object]]:
    crop_path = patch_root / Path(manifest_row["crop_file"])
    bgr = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read crop: {crop_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    height, width = rgb.shape[:2]
    center_x = float(manifest_row["center_x_px"]) - float(manifest_row["crop_x0_px"])
    center_y = float(manifest_row["center_y_px"]) - float(manifest_row["crop_y0_px"])
    radius = float(manifest_row["circle_radius_px"])

    yy, xx = np.ogrid[:height, :width]
    distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    roi_mask = distance <= ROI_RADIUS_FRACTION * radius
    background_mask = (
        (distance >= BACKGROUND_INNER_RADIUS_FRACTION * radius)
        & (distance <= BACKGROUND_OUTER_RADIUS_FRACTION * radius)
    )

    saturation_all = hsv[..., 1].astype(np.float64) / 255.0
    value_all = hsv[..., 2].astype(np.float64) / 255.0
    highlight_mask = (
        roi_mask
        & (value_all >= HIGHLIGHT_V_THRESHOLD)
        & (saturation_all <= HIGHLIGHT_S_THRESHOLD)
    )
    valid_mask = roi_mask & ~highlight_mask

    roi_count = int(np.count_nonzero(roi_mask))
    valid_count = int(np.count_nonzero(valid_mask))
    highlight_count = int(np.count_nonzero(highlight_mask))
    background_count = int(np.count_nonzero(background_mask))
    if roi_count == 0 or valid_count == 0 or background_count == 0:
        raise ValueError(f"Empty measurement mask for {manifest_row['patch_id']}")

    rgb_values = rgb[valid_mask].astype(np.float64)
    saturation = saturation_all[valid_mask]
    value = value_all[valid_mask]
    hue_deg = hsv[..., 0][valid_mask].astype(np.float64) * 2.0

    result: dict[str, object] = {
        "patch_id": manifest_row["patch_id"],
        "image_id": manifest_row["image_id"],
        "analyte": manifest_row["analyte"],
        "concentration_order": int(manifest_row["concentration_order"]),
        "concentration_mg_ml": float(manifest_row["concentration_mg_ml"]),
        "well_id": manifest_row["well_id"],
        "grid_row": int(manifest_row["grid_row"]),
        "grid_col": int(manifest_row["grid_col"]),
        "crop_file": manifest_row["crop_file"],
        "circle_radius_px": rounded(radius),
    }

    channel_medians: dict[str, float] = {}
    for index, channel in enumerate(("r", "g", "b")):
        mean, median, std, iqr = distribution_stats(rgb_values[:, index])
        result[f"{channel}_mean"] = rounded(mean)
        result[f"{channel}_median"] = rounded(median)
        result[f"{channel}_std"] = rounded(std)
        result[f"{channel}_iqr"] = rounded(iqr)
        channel_medians[channel] = median

    for channel, values in (("s", saturation), ("v", value)):
        mean, median, std, iqr = distribution_stats(values)
        result[f"{channel}_mean"] = rounded(mean)
        result[f"{channel}_median"] = rounded(median)
        result[f"{channel}_std"] = rounded(std)
        result[f"{channel}_iqr"] = rounded(iqr)

    hue_stats = circular_hue_stats(hue_deg, saturation)
    for field, value_item in zip(HUE_FIELDS, hue_stats, strict=True):
        result[field] = rounded(value_item)

    rgb_sum = np.sum(rgb_values, axis=1)
    safe_sum = np.maximum(rgb_sum, np.finfo(float).eps)
    for index, channel in enumerate(("r", "g", "b")):
        chromaticity = rgb_values[:, index] / safe_sum
        result[f"{channel}_chromaticity_mean"] = rounded(np.mean(chromaticity))
        result[f"{channel}_chromaticity_median"] = rounded(np.median(chromaticity))

    intensity = rgb_sum / 3.0
    result["intensity_mean"] = rounded(np.mean(intensity))
    result["intensity_median"] = rounded(np.median(intensity))

    background_rgb = rgb[background_mask].astype(np.float64)
    for index, channel in enumerate(("r", "g", "b")):
        background_median = float(np.median(background_rgb[:, index]))
        result[f"{channel}_bg_median"] = rounded(background_median)
        result[f"{channel}_delta_bg"] = rounded(
            channel_medians[channel] - background_median
        )
        result[f"{channel}_ratio_bg"] = rounded(
            channel_medians[channel] / max(background_median, np.finfo(float).eps)
        )

    highlight_fraction = highlight_count / roi_count
    valid_fraction = valid_count / roi_count
    dark_fraction = float(np.count_nonzero(roi_mask & (value_all <= DARK_V_THRESHOLD))) / roi_count

    qc_reasons: list[str] = []
    if highlight_fraction > QC_MAX_HIGHLIGHT_FRACTION:
        qc_reasons.append("high_specular_fraction")
    if valid_fraction < QC_MIN_VALID_FRACTION:
        qc_reasons.append("low_valid_fraction")
    qc_status = "review" if qc_reasons else "pass"

    result.update(
        {
            "roi_pixel_count": roi_count,
            "valid_pixel_count": valid_count,
            "highlight_pixel_count": highlight_count,
            "highlight_fraction": rounded(highlight_fraction),
            "valid_fraction": rounded(valid_fraction),
            "dark_fraction": rounded(dark_fraction),
            "background_pixel_count": background_count,
            "qc_status": qc_status,
            "qc_reason": ";".join(qc_reasons),
        }
    )

    visual = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    visual[highlight_mask] = (0, 0, 255)
    center = (int(round(center_x)), int(round(center_y)))
    cv2.circle(
        visual,
        center,
        int(round(ROI_RADIUS_FRACTION * radius)),
        (0, 180, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.circle(
        visual,
        center,
        int(round(BACKGROUND_INNER_RADIUS_FRACTION * radius)),
        (255, 120, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.circle(
        visual,
        center,
        int(round(BACKGROUND_OUTER_RADIUS_FRACTION * radius)),
        (255, 120, 0),
        1,
        cv2.LINE_AA,
    )

    qc_info = {
        "patch_id": result["patch_id"],
        "highlight_fraction": highlight_fraction,
        "qc_status": qc_status,
    }
    return result, visual, qc_info


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["image_id"])].append(row)

    output: list[dict[str, object]] = []
    for image_id in sorted(grouped):
        group = grouped[image_id]
        highlights = np.array([float(row["highlight_fraction"]) for row in group])
        valids = np.array([float(row["valid_fraction"]) for row in group])
        output.append(
            {
                "image_id": image_id,
                "analyte": group[0]["analyte"],
                "concentration_order": group[0]["concentration_order"],
                "concentration_mg_ml": group[0]["concentration_mg_ml"],
                "patch_count": len(group),
                "qc_pass_count": sum(row["qc_status"] == "pass" for row in group),
                "qc_review_count": sum(row["qc_status"] == "review" for row in group),
                "highlight_fraction_mean": rounded(np.mean(highlights)),
                "highlight_fraction_median": rounded(np.median(highlights)),
                "highlight_fraction_max": rounded(np.max(highlights)),
                "valid_fraction_min": rounded(np.min(valids)),
            }
        )
    return output


def make_qc_contact_sheet(
    visuals: dict[str, tuple[np.ndarray, dict[str, object]]], output_path: Path
) -> None:
    selected_ids = [
        f"{image_id}_w{well_id}"
        for image_id in (
            "glucose_01_c0",
            "glucose_11_c20",
            "ketone_01_c0",
            "ketone_08_c10",
        )
        for well_id in ("A01", "E06", "H12")
    ]
    panels: list[np.ndarray] = []
    for patch_id in selected_ids:
        visual, info = visuals[patch_id]
        right_padding = max(6, 310 - visual.shape[1])
        panel = cv2.copyMakeBorder(
            visual,
            30,
            6,
            6,
            right_padding,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
        label = f"{patch_id} | highlight {100 * info['highlight_fraction']:.1f}%"
        cv2.putText(
            panel,
            label,
            (7, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        panels.append(panel)

    target_height = max(panel.shape[0] for panel in panels)
    target_width = max(panel.shape[1] for panel in panels)
    normalized = []
    for panel in panels:
        bottom = target_height - panel.shape[0]
        right = target_width - panel.shape[1]
        normalized.append(
            cv2.copyMakeBorder(
                panel,
                0,
                bottom,
                0,
                right,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            )
        )
    rows = [np.hstack(normalized[start : start + 3]) for start in range(0, 12, 3)]
    contact_sheet = np.vstack(rows)
    legend = np.full((38, contact_sheet.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        legend,
        "Green: central ROI | Blue: local-background annulus | Red: excluded highlight pixels",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    contact_sheet = np.vstack([legend, contact_sheet])
    cv2.imwrite(str(output_path), contact_sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])


def make_top_highlight_contact_sheet(
    top_visuals: list[tuple[float, str, np.ndarray, dict[str, object]]],
    output_path: Path,
) -> None:
    panels: list[np.ndarray] = []
    for _, patch_id, visual, info in top_visuals:
        right_padding = max(6, 310 - visual.shape[1])
        panel = cv2.copyMakeBorder(
            visual,
            30,
            6,
            6,
            right_padding,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
        label = f"{patch_id} | highlight {100 * info['highlight_fraction']:.1f}%"
        cv2.putText(
            panel,
            label,
            (7, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        panels.append(panel)

    target_height = max(panel.shape[0] for panel in panels)
    target_width = max(panel.shape[1] for panel in panels)
    normalized = []
    for panel in panels:
        normalized.append(
            cv2.copyMakeBorder(
                panel,
                0,
                target_height - panel.shape[0],
                0,
                target_width - panel.shape[1],
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            )
        )
    rows = [np.hstack(normalized[start : start + 3]) for start in range(0, 6, 3)]
    contact_sheet = np.vstack(rows)
    legend = np.full((38, contact_sheet.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        legend,
        "Six highest highlight fractions | Red: pixels excluded by the fixed specular rule",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    contact_sheet = np.vstack([legend, contact_sheet])
    cv2.imwrite(str(output_path), contact_sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])


def main() -> None:
    args = parse_args()
    manifest_path = args.patch_manifest.resolve()
    patch_root = args.patch_root.resolve()
    output_dir = args.output_dir.resolve()
    prepare_output_dir(output_dir, args.overwrite)

    manifest_rows = read_manifest(manifest_path)
    feature_rows: list[dict[str, object]] = []
    representative_visuals: dict[str, tuple[np.ndarray, dict[str, object]]] = {}
    top_highlight_visuals: list[
        tuple[float, str, np.ndarray, dict[str, object]]
    ] = []
    representative_wells = {"A01", "E06", "H12"}
    representative_images = {
        "glucose_01_c0",
        "glucose_11_c20",
        "ketone_01_c0",
        "ketone_08_c10",
    }

    for index, row in enumerate(manifest_rows, start=1):
        feature_row, visual, qc_info = extract_one(row, patch_root)
        feature_rows.append(feature_row)
        if (
            row["image_id"] in representative_images
            and row["well_id"] in representative_wells
        ):
            representative_visuals[row["patch_id"]] = (visual, qc_info)
        top_highlight_visuals.append(
            (
                float(qc_info["highlight_fraction"]),
                row["patch_id"],
                visual,
                qc_info,
            )
        )
        top_highlight_visuals.sort(key=lambda item: item[0], reverse=True)
        del top_highlight_visuals[6:]
        if index % 250 == 0 or index == len(manifest_rows):
            print(f"Processed {index}/{len(manifest_rows)} patches")

    feature_path = output_dir / "features.csv"
    write_csv(feature_path, feature_rows, FEATURE_FIELDS)

    summary_rows = build_summary(feature_rows)
    summary_fields = [
        "image_id",
        "analyte",
        "concentration_order",
        "concentration_mg_ml",
        "patch_count",
        "qc_pass_count",
        "qc_review_count",
        "highlight_fraction_mean",
        "highlight_fraction_median",
        "highlight_fraction_max",
        "valid_fraction_min",
    ]
    write_csv(output_dir / "feature_extraction_summary.csv", summary_rows, summary_fields)

    make_qc_contact_sheet(
        representative_visuals, output_dir / "qc" / "roi_qc_contact_sheet.jpg"
    )
    make_top_highlight_contact_sheet(
        top_highlight_visuals,
        output_dir / "qc" / "highest_highlight_qc_contact_sheet.jpg",
    )

    config = {
        "input_manifest": str(manifest_path),
        "patch_root": str(patch_root),
        "output_directory": str(output_dir),
        "patch_count": len(feature_rows),
        "feature_column_count_including_metadata_and_qc": len(FEATURE_FIELDS),
        "roi_radius_fraction": ROI_RADIUS_FRACTION,
        "background_annulus_radius_fractions": [
            BACKGROUND_INNER_RADIUS_FRACTION,
            BACKGROUND_OUTER_RADIUS_FRACTION,
        ],
        "specular_highlight_rule": {
            "v_greater_than_or_equal": HIGHLIGHT_V_THRESHOLD,
            "s_less_than_or_equal": HIGHLIGHT_S_THRESHOLD,
        },
        "dark_pixel_reporting_threshold_v": DARK_V_THRESHOLD,
        "qc_review_rules": {
            "highlight_fraction_greater_than": QC_MAX_HIGHLIGHT_FRACTION,
            "valid_fraction_less_than": QC_MIN_VALID_FRACTION,
        },
        "hue_method": "saturation-weighted circular statistics",
        "color_scale": {
            "rgb": "0-255",
            "hue": "degrees 0-360",
            "saturation_value": "0-1",
        },
        "notes": [
            "No resizing, white balancing, or color correction is applied before feature extraction.",
            "QC review flags are retained in the feature table and do not automatically exclude patches.",
            "Background-normalized features are auxiliary and remain separate from raw RGB/HSV features.",
        ],
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    print(f"Wrote {feature_path}")
    print(
        "QC review patches:",
        sum(row["qc_status"] == "review" for row in feature_rows),
    )


if __name__ == "__main__":
    main()

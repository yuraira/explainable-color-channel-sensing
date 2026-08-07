"""Verify feature-validation tables, figures, and core scientific invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


EXPECTED_ANALYTE_IMAGES = {"glucose": 11, "ketone": 8}
EXPECTED_TABLES = [
    "concentration_summary.csv",
    "concentration_trends.csv",
    "adjacent_concentration_overlap.csv",
    "within_image_variability.csv",
    "hue_unwrapped_summary.csv",
    "position_residuals.csv",
    "position_well_summary.csv",
    "position_bias_summary.csv",
    "background_control_summary.csv",
    "feature_correlation_summary.csv",
    "model_feature_sets.csv",
]
EXPECTED_FIGURES = [
    "concentration_trends_rgb.png",
    "concentration_trends_hsv.png",
    "within_image_variability.png",
    "position_bias_heatmaps.png",
    "background_negative_control.png",
    "feature_redundancy_heatmap.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/feature_validation"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    errors: list[str] = []
    warnings: list[str] = []

    for name in EXPECTED_TABLES:
        if not (input_dir / name).is_file():
            errors.append(f"Missing table: {name}")
    for name in EXPECTED_FIGURES:
        path = input_dir / "figures" / name
        if not path.is_file():
            errors.append(f"Missing figure: {name}")
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                if image.width < 1000 or image.height < 500:
                    warnings.append(
                        f"Figure may be too small: {name} ({image.width}x{image.height})"
                    )
        except Exception as exc:  # pragma: no cover - verification diagnostic
            errors.append(f"Unreadable figure {name}: {exc}")

    if not errors:
        concentration = pd.read_csv(
            input_dir / "concentration_summary.csv", encoding="utf-8-sig"
        )
        trends = pd.read_csv(
            input_dir / "concentration_trends.csv", encoding="utf-8-sig"
        )
        residuals = pd.read_csv(
            input_dir / "position_residuals.csv", encoding="utf-8-sig"
        )
        well = pd.read_csv(
            input_dir / "position_well_summary.csv", encoding="utf-8-sig"
        )
        feature_sets = pd.read_csv(
            input_dir / "model_feature_sets.csv", encoding="utf-8-sig"
        )

        if concentration["image_id"].nunique() != 19:
            errors.append("Concentration summary does not contain 19 source images")
        image_counts = (
            concentration[["analyte", "image_id"]]
            .drop_duplicates()["analyte"]
            .value_counts()
            .to_dict()
        )
        if image_counts != EXPECTED_ANALYTE_IMAGES:
            errors.append(f"Unexpected analyte image counts: {image_counts}")
        if not (concentration["n_patches"] == 96).all():
            errors.append("Not every concentration summary uses 96 technical patches")
        if residuals.shape[0] != 1824:
            errors.append(f"Expected 1824 position residual rows, found {residuals.shape[0]}")
        expected_well_rows = 2 * 6 * 96
        if well.shape[0] != expected_well_rows:
            errors.append(
                f"Expected {expected_well_rows} position-well rows, found {well.shape[0]}"
            )
        rho = trends["spearman_rho_vs_concentration_order"].to_numpy(dtype=float)
        if np.any(~np.isfinite(rho)) or np.any(np.abs(rho) > 1):
            errors.append("Invalid Spearman rho values")
        if set(feature_sets.loc[feature_sets["analysis_role"] == "primary", "feature_set"]) != {
            "RGB_primary",
            "HSV_primary",
        }:
            errors.append("Primary feature sets are not defined as expected")
        if feature_sets.loc[
            feature_sets["feature_set"] == "Background_negative_control"
        ].shape[0] != 3:
            errors.append("Background negative-control feature set is incomplete")

    report = {
        "tables_verified": len(EXPECTED_TABLES),
        "figures_verified": len(EXPECTED_FIGURES),
        "errors": errors,
        "warnings": warnings,
    }
    with (input_dir / "verification_report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

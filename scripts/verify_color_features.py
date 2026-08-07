"""Verify the completeness, identity, numeric ranges, and QC of features.csv."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_ROWS = 1824
EXPECTED_ANALYTE_COUNTS = {"glucose": 1056, "ketone": 768}
EXPECTED_IMAGE_COUNTS = {"glucose": 11, "ketone": 8}
EXPECTED_WELLS = {
    f"{row}{column:02d}" for row in "ABCDEFGH" for column in range(1, 13)
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("outputs/color_features/features.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/color_features/verification_report.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.features.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames or []

    errors: list[str] = []
    warnings: list[str] = []
    if len(rows) != EXPECTED_ROWS:
        errors.append(f"Expected {EXPECTED_ROWS} rows, found {len(rows)}")

    patch_ids = [row["patch_id"] for row in rows]
    if len(set(patch_ids)) != len(patch_ids):
        errors.append("Duplicate patch_id values found")

    analyte_counts = Counter(row["analyte"] for row in rows)
    if dict(analyte_counts) != EXPECTED_ANALYTE_COUNTS:
        errors.append(f"Unexpected analyte counts: {dict(analyte_counts)}")

    image_groups: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        image_groups[row["image_id"]].append(row)
    if len(image_groups) != sum(EXPECTED_IMAGE_COUNTS.values()):
        errors.append(f"Expected 19 image groups, found {len(image_groups)}")
    for image_id, group in image_groups.items():
        if len(group) != 96:
            errors.append(f"{image_id} has {len(group)} rows instead of 96")
        wells = {row["well_id"] for row in group}
        if wells != EXPECTED_WELLS:
            errors.append(f"{image_id} has an invalid well set")

    text_fields = {
        "patch_id",
        "image_id",
        "analyte",
        "well_id",
        "crop_file",
        "qc_status",
        "qc_reason",
    }
    numeric_fields = [field for field in fields if field not in text_fields]
    numeric_values: defaultdict[str, list[float]] = defaultdict(list)
    for row_number, row in enumerate(rows, start=2):
        for field in numeric_fields:
            try:
                value = float(row[field])
            except (TypeError, ValueError):
                errors.append(f"Non-numeric {field} at CSV row {row_number}")
                continue
            if not math.isfinite(value):
                errors.append(f"Non-finite {field} at CSV row {row_number}")
            numeric_values[field].append(value)

    def check_range(field: str, lower: float, upper: float) -> None:
        values = numeric_values[field]
        if values and (min(values) < lower or max(values) > upper):
            errors.append(
                f"{field} outside [{lower}, {upper}]: {min(values)} to {max(values)}"
            )

    for channel in ("r", "g", "b"):
        for stat in ("mean", "median", "std", "iqr"):
            check_range(f"{channel}_{stat}", 0.0, 255.0)
        check_range(f"{channel}_bg_median", 0.0, 255.0)
        check_range(f"{channel}_chromaticity_mean", 0.0, 1.0)
        check_range(f"{channel}_chromaticity_median", 0.0, 1.0)
        if min(numeric_values[f"{channel}_ratio_bg"]) < 0.0:
            errors.append(f"Negative {channel}_ratio_bg value")

    for channel in ("s", "v"):
        for stat in ("mean", "median", "std", "iqr"):
            check_range(f"{channel}_{stat}", 0.0, 1.0)
    check_range("h_circular_mean_deg", 0.0, 360.0)
    check_range("h_resultant_length", 0.0, 1.0)
    check_range("h_sin_weighted", -1.0, 1.0)
    check_range("h_cos_weighted", -1.0, 1.0)
    for field in ("highlight_fraction", "valid_fraction", "dark_fraction"):
        check_range(field, 0.0, 1.0)

    qc_counts = Counter(row["qc_status"] for row in rows)
    unexpected_qc = set(qc_counts) - {"pass", "review"}
    if unexpected_qc:
        errors.append(f"Unexpected QC status values: {sorted(unexpected_qc)}")
    if qc_counts.get("review", 0):
        warnings.append(
            f"{qc_counts['review']} patches are marked for manual review; they were not excluded."
        )

    report = {
        "feature_rows": len(rows),
        "feature_columns": len(fields),
        "unique_patch_ids": len(set(patch_ids)),
        "source_images": len(image_groups),
        "analyte_counts": dict(analyte_counts),
        "qc_counts": dict(qc_counts),
        "highlight_fraction": {
            "minimum": min(numeric_values["highlight_fraction"]),
            "maximum": max(numeric_values["highlight_fraction"]),
        },
        "valid_fraction": {
            "minimum": min(numeric_values["valid_fraction"]),
            "maximum": max(numeric_values["valid_fraction"]),
        },
        "errors": errors,
        "warnings": warnings,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

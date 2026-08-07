"""Verify patch-detection outputs and create a compact QC contact sheet."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


EXPECTED_IMAGES = 19
EXPECTED_PER_IMAGE = 96
EXPECTED_PATCHES = EXPECTED_IMAGES * EXPECTED_PER_IMAGE
EXPECTED_WELLS = {
    f"{chr(65 + row)}{col + 1:02d}"
    for row in range(8)
    for col in range(12)
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def create_contact_sheet(qc_files: list[Path], output_path: Path) -> None:
    columns = 4
    tile_width = 480
    tile_height = 390
    label_height = 26
    rows = math.ceil(len(qc_files) / columns)
    canvas = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for index, qc_file in enumerate(qc_files):
        row, col = divmod(index, columns)
        with Image.open(qc_file) as image:
            image = image.convert("RGB")
            image.thumbnail((tile_width - 12, tile_height - label_height - 12), Image.Resampling.LANCZOS)
            x = col * tile_width + (tile_width - image.width) // 2
            y = row * tile_height + label_height + (tile_height - label_height - image.height) // 2
            canvas.paste(image, (x, y))
        draw.text((col * tile_width + 8, row * tile_height + 7), qc_file.stem, fill="black", font=font)
        draw.rectangle(
            [col * tile_width, row * tile_height, (col + 1) * tile_width - 1, (row + 1) * tile_height - 1],
            outline="#B8C2CC",
            width=1,
        )

    canvas.save(output_path, format="JPEG", quality=92, optimize=True)


def main() -> None:
    args = parse_args()
    patch_manifest = read_csv(args.output_dir / "patch_manifest.csv")
    detection_summary = read_csv(args.output_dir / "detection_summary.csv")

    errors: list[str] = []
    patch_ids = [row["patch_id"] for row in patch_manifest]
    image_ids = [row["image_id"] for row in patch_manifest]
    per_image = Counter(image_ids)
    wells_by_image: dict[str, set[str]] = defaultdict(set)

    if len(detection_summary) != EXPECTED_IMAGES:
        errors.append(f"Expected {EXPECTED_IMAGES} summary rows, found {len(detection_summary)}")
    if len(patch_manifest) != EXPECTED_PATCHES:
        errors.append(f"Expected {EXPECTED_PATCHES} patch rows, found {len(patch_manifest)}")
    if len(set(patch_ids)) != len(patch_ids):
        errors.append("Duplicate patch_id values found")

    missing_crops: list[str] = []
    invalid_crops: list[str] = []
    residuals: list[float] = []
    radii: list[float] = []

    for row in patch_manifest:
        image_id = row["image_id"]
        wells_by_image[image_id].add(row["well_id"])
        crop_path = args.output_dir / row["crop_file"]
        if not crop_path.is_file():
            missing_crops.append(str(crop_path))
            continue
        try:
            with Image.open(crop_path) as image:
                image.verify()
            if int(row["crop_width_px"]) <= 0 or int(row["crop_height_px"]) <= 0:
                invalid_crops.append(row["patch_id"])
        except Exception:
            invalid_crops.append(row["patch_id"])

        center_x = float(row["center_x_px"])
        center_y = float(row["center_y_px"])
        width = int(row["canonical_width_px"])
        height = int(row["canonical_height_px"])
        radius = float(row["circle_radius_px"])
        residual = float(row["grid_residual_px"])
        radii.append(radius)
        residuals.append(residual)
        if not (0 < center_x < width and 0 < center_y < height and radius > 0):
            errors.append(f"Invalid geometry for {row['patch_id']}")

    for image_id, count in sorted(per_image.items()):
        if count != EXPECTED_PER_IMAGE:
            errors.append(f"{image_id}: expected {EXPECTED_PER_IMAGE} patches, found {count}")
        if wells_by_image[image_id] != EXPECTED_WELLS:
            errors.append(f"{image_id}: well_id set is incomplete or duplicated")

    if missing_crops:
        errors.append(f"Missing crop files: {len(missing_crops)}")
    if invalid_crops:
        errors.append(f"Unreadable or invalid crop files: {len(invalid_crops)}")

    qc_files = sorted((args.output_dir / "qc").glob("*_qc.jpg"))
    if len(qc_files) != EXPECTED_IMAGES:
        errors.append(f"Expected {EXPECTED_IMAGES} QC images, found {len(qc_files)}")
    else:
        create_contact_sheet(qc_files, args.output_dir / "qc_contact_sheet.jpg")

    analyte_counts = Counter(row["analyte"] for row in patch_manifest)
    report = {
        "status": "pass" if not errors else "fail",
        "source_images": len(detection_summary),
        "patch_records": len(patch_manifest),
        "unique_patch_ids": len(set(patch_ids)),
        "crop_files_verified": len(patch_manifest) - len(missing_crops) - len(invalid_crops),
        "qc_images": len(qc_files),
        "analyte_patch_counts": dict(analyte_counts),
        "max_grid_residual_px": round(max(residuals), 3) if residuals else None,
        "mean_grid_residual_px": round(sum(residuals) / len(residuals), 3) if residuals else None,
        "min_circle_radius_px": round(min(radii), 3) if radii else None,
        "max_circle_radius_px": round(max(radii), 3) if radii else None,
        "errors": errors,
    }
    with (args.output_dir / "verification_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Detect and extract the 8 x 12 circular sensor patches in each source image.

The detector deliberately separates geometry from downstream feature extraction:
it corrects EXIF orientation, standardizes the grid to 8 rows x 12 columns,
detects high-confidence circle candidates, fits a projective grid, and saves
lossless native-resolution crops plus auditable QC metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.optimize import linear_sum_assignment

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.cluster import DBSCAN, KMeans


GRID_ROWS = 8
GRID_COLS = 12
EXPECTED_PATCHES = GRID_ROWS * GRID_COLS
IMAGE_PATTERN = re.compile(
    r"^(?P<analyte>glucose|ketone)_(?P<order>\d{2})_c(?P<concentration>[0-9]+(?:\.[0-9]+)?)\.jpg$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DetectionResult:
    filename: str
    image_id: str
    analyte: str
    concentration_order: int
    concentration_mg_ml: float
    canonical_width_px: int
    canonical_height_px: int
    exif_transposed: bool
    canonical_rotation: str
    detection_scale: float
    hough_param2: int
    candidate_count: int
    median_radius_scaled_px: float
    mean_grid_residual_scaled_px: float
    max_grid_residual_scaled_px: float
    grid_spacing_scaled_px: float
    detected_patches: int
    qc_status: str


@dataclass(frozen=True)
class PatchRecord:
    patch_id: str
    image_id: str
    analyte: str
    concentration_order: int
    concentration_mg_ml: float
    well_id: str
    grid_row: int
    grid_col: int
    canonical_width_px: int
    canonical_height_px: int
    center_x_px: float
    center_y_px: float
    circle_radius_px: float
    crop_x0_px: int
    crop_y0_px: int
    crop_x1_px: int
    crop_y1_px: int
    crop_width_px: int
    crop_height_px: int
    crop_file: str
    grid_residual_px: float
    hough_param2: int
    candidate_count: int
    qc_status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Optional source filenames. If omitted, all matching JPG files are processed.",
    )
    parser.add_argument("--max-detection-dimension", type=int, default=1600)
    parser.add_argument("--qc-max-dimension", type=int, default=2000)
    return parser.parse_args()


def load_canonical_rgb(path: Path) -> tuple[np.ndarray, bool, str]:
    with Image.open(path) as source:
        orientation = source.getexif().get(274, 1)
        corrected = ImageOps.exif_transpose(source).convert("RGB")
        rgb = np.asarray(corrected)

    rotation = "none"
    if rgb.shape[0] > rgb.shape[1]:
        # Ketone images display as a 12-row x 8-column portrait array.
        # Rotating counter-clockwise creates the canonical 8 x 12 layout.
        rgb = cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotation = "90_ccw_after_exif"
    return rgb, orientation not in (None, 1), rotation


def dominant_spatial_component(candidates: np.ndarray) -> np.ndarray:
    radii = candidates[:, 2]
    median_radius = float(np.median(radii))
    plausible = candidates[
        (radii >= 0.68 * median_radius) & (radii <= 1.35 * median_radius)
    ]
    if len(plausible) < 4:
        return plausible

    centers = plausible[:, :2]
    distance = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(distance, np.inf)
    median_nearest = float(np.median(np.min(distance, axis=1)))
    eps = float(np.clip(1.28 * median_nearest, 60.0, 130.0))
    labels = DBSCAN(eps=eps, min_samples=3).fit_predict(centers)
    valid_labels = labels[labels >= 0]
    if len(valid_labels) == 0:
        return plausible
    label_counts = np.bincount(valid_labels)
    largest_label = int(np.argmax(label_counts))
    return plausible[labels == largest_label]


def adaptive_hough(gray: np.ndarray) -> tuple[np.ndarray, int]:
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
    best: tuple[np.ndarray, int] | None = None

    # Start conservatively and lower the accumulator threshold only until at
    # least 96 candidates are present. This minimizes false positives while
    # retaining every grid location.
    for param2 in range(45, 19, -1):
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=55,
            param1=100,
            param2=param2,
            minRadius=18,
            maxRadius=50,
        )
        if circles is None:
            continue
        raw_candidates = circles[0].astype(np.float64)
        candidates = dominant_spatial_component(raw_candidates)
        best = candidates, param2
        if len(candidates) >= EXPECTED_PATCHES:
            return candidates, param2

    if best is None:
        raise RuntimeError("Hough circle detection returned no candidates")
    candidates, param2 = best
    raise RuntimeError(
        f"Only {len(candidates)} circle candidates were found at the lowest Hough threshold ({param2}); "
        f"at least {EXPECTED_PATCHES} are required"
    )


def ordered_cluster_labels(values: np.ndarray, clusters: int) -> np.ndarray:
    model = KMeans(n_clusters=clusters, n_init=30, random_state=0)
    raw = model.fit_predict(values.reshape(-1, 1))
    order = np.argsort(model.cluster_centers_.ravel())
    remap = np.empty_like(order)
    remap[order] = np.arange(clusters)
    return remap[raw]


def project_grid(homography: np.ndarray) -> np.ndarray:
    grid = np.array(
        [[col, row] for row in range(GRID_ROWS) for col in range(GRID_COLS)],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(grid, homography).reshape(-1, 2).astype(np.float64)


def fit_grid(candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if len(candidates) < EXPECTED_PATCHES:
        raise ValueError(f"Expected at least {EXPECTED_PATCHES} candidates, received {len(candidates)}")

    centers = candidates[:, :2]
    radii = candidates[:, 2]

    # Remove implausible radii while retaining a fallback to the complete set.
    median_radius = float(np.median(radii))
    plausible = (radii >= 0.72 * median_radius) & (radii <= 1.28 * median_radius)
    if int(plausible.sum()) >= EXPECTED_PATCHES:
        centers = centers[plausible]
        radii = radii[plausible]

    col_labels = ordered_cluster_labels(centers[:, 0], GRID_COLS)
    row_labels = ordered_cluster_labels(centers[:, 1], GRID_ROWS)
    logical = np.column_stack((col_labels, row_labels)).astype(np.float32)

    homography, inlier_mask = cv2.findHomography(
        logical,
        centers.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=max(8.0, 0.25 * median_radius),
    )
    if homography is None or inlier_mask is None:
        raise RuntimeError("Projective grid fitting failed")

    predicted = project_grid(homography)
    distance = np.linalg.norm(predicted[:, None, :] - centers[None, :, :], axis=2)
    grid_indices, candidate_indices = linear_sum_assignment(distance)
    if len(grid_indices) != EXPECTED_PATCHES:
        raise RuntimeError("Could not assign one circle candidate to every grid location")

    order = np.argsort(grid_indices)
    selected_indices = candidate_indices[order]
    selected_centers = centers[selected_indices]
    selected_radii = radii[selected_indices]
    residuals = np.linalg.norm(selected_centers - predicted, axis=1)

    predicted_grid = predicted.reshape(GRID_ROWS, GRID_COLS, 2)
    horizontal = np.linalg.norm(predicted_grid[:, 1:] - predicted_grid[:, :-1], axis=2).ravel()
    vertical = np.linalg.norm(predicted_grid[1:, :] - predicted_grid[:-1, :], axis=2).ravel()
    spacing = float(np.median(np.concatenate((horizontal, vertical))))

    # Reject a fit whose assignments are too far from the projective grid.
    if float(np.max(residuals)) > 0.38 * spacing:
        raise RuntimeError(
            f"Grid assignment residual is too large: max={np.max(residuals):.1f}px, spacing={spacing:.1f}px"
        )

    return selected_centers, selected_radii, residuals, spacing


def safe_crop(image: np.ndarray, center_x: float, center_y: float, radius: float) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    margin_radius = max(4, int(math.ceil(radius * 1.18)))
    x0 = max(0, int(round(center_x)) - margin_radius)
    y0 = max(0, int(round(center_y)) - margin_radius)
    x1 = min(image.shape[1], int(round(center_x)) + margin_radius + 1)
    y1 = min(image.shape[0], int(round(center_y)) + margin_radius + 1)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        raise RuntimeError("Generated an empty crop")
    return crop, (x0, y0, x1, y1)


def save_rgb_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, format="PNG", compress_level=3)


def save_qc_image(
    path: Path,
    rgb: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    max_dimension: int,
) -> None:
    height, width = rgb.shape[:2]
    scale = min(1.0, max_dimension / max(height, width))
    display = cv2.resize(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )
    for index, ((center_x, center_y), radius) in enumerate(zip(centers, radii, strict=True)):
        row, col = divmod(index, GRID_COLS)
        well_id = f"{chr(65 + row)}{col + 1:02d}"
        x = int(round(center_x * scale))
        y = int(round(center_y * scale))
        r = max(2, int(round(radius * scale)))
        cv2.circle(display, (x, y), r, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(display, (x, y), 2, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            display,
            well_id,
            (x - r, y - r - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 80, 0),
            1,
            cv2.LINE_AA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), display, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"Failed to write QC image: {path}")


def process_image(
    source_path: Path,
    output_dir: Path,
    max_detection_dimension: int,
    qc_max_dimension: int,
) -> tuple[DetectionResult, list[PatchRecord]]:
    match = IMAGE_PATTERN.match(source_path.name)
    if not match:
        raise ValueError(f"Unexpected source filename: {source_path.name}")

    image_id = source_path.stem
    analyte = match.group("analyte").lower()
    concentration_order = int(match.group("order"))
    concentration_mg_ml = float(match.group("concentration"))

    rgb, exif_transposed, canonical_rotation = load_canonical_rgb(source_path)
    height, width = rgb.shape[:2]
    detection_scale = min(1.0, max_detection_dimension / max(height, width))
    detection_rgb = cv2.resize(
        rgb,
        None,
        fx=detection_scale,
        fy=detection_scale,
        interpolation=cv2.INTER_AREA,
    )
    detection_gray = cv2.cvtColor(detection_rgb, cv2.COLOR_RGB2GRAY)
    candidates, param2 = adaptive_hough(detection_gray)
    selected_centers, selected_radii, residuals, spacing = fit_grid(candidates)

    full_centers = selected_centers / detection_scale
    full_radii = selected_radii / detection_scale
    median_full_radius = float(np.median(full_radii))
    full_radii = np.clip(full_radii, 0.88 * median_full_radius, 1.12 * median_full_radius)

    patch_records: list[PatchRecord] = []
    for index, ((center_x, center_y), radius, residual) in enumerate(
        zip(full_centers, full_radii, residuals / detection_scale, strict=True)
    ):
        row, col = divmod(index, GRID_COLS)
        well_id = f"{chr(65 + row)}{col + 1:02d}"
        patch_id = f"{image_id}_w{well_id}"
        crop, (x0, y0, x1, y1) = safe_crop(rgb, center_x, center_y, radius)
        relative_crop = Path("crops") / analyte / image_id / f"{patch_id}.png"
        save_rgb_png(output_dir / relative_crop, crop)
        patch_records.append(
            PatchRecord(
                patch_id=patch_id,
                image_id=image_id,
                analyte=analyte,
                concentration_order=concentration_order,
                concentration_mg_ml=concentration_mg_ml,
                well_id=well_id,
                grid_row=row + 1,
                grid_col=col + 1,
                canonical_width_px=width,
                canonical_height_px=height,
                center_x_px=round(float(center_x), 3),
                center_y_px=round(float(center_y), 3),
                circle_radius_px=round(float(radius), 3),
                crop_x0_px=x0,
                crop_y0_px=y0,
                crop_x1_px=x1,
                crop_y1_px=y1,
                crop_width_px=x1 - x0,
                crop_height_px=y1 - y0,
                crop_file=relative_crop.as_posix(),
                grid_residual_px=round(float(residual), 3),
                hough_param2=param2,
                candidate_count=len(candidates),
                qc_status="pass_geometry",
            )
        )

    save_qc_image(
        output_dir / "qc" / f"{image_id}_qc.jpg",
        rgb,
        full_centers,
        full_radii,
        max_dimension=qc_max_dimension,
    )

    result = DetectionResult(
        filename=source_path.name,
        image_id=image_id,
        analyte=analyte,
        concentration_order=concentration_order,
        concentration_mg_ml=concentration_mg_ml,
        canonical_width_px=width,
        canonical_height_px=height,
        exif_transposed=exif_transposed,
        canonical_rotation=canonical_rotation,
        detection_scale=round(detection_scale, 6),
        hough_param2=param2,
        candidate_count=len(candidates),
        median_radius_scaled_px=round(float(np.median(selected_radii)), 3),
        mean_grid_residual_scaled_px=round(float(np.mean(residuals)), 3),
        max_grid_residual_scaled_px=round(float(np.max(residuals)), 3),
        grid_spacing_scaled_px=round(spacing, 3),
        detected_patches=len(patch_records),
        qc_status="pass_geometry" if len(patch_records) == EXPECTED_PATCHES else "fail_count",
    )
    return result, patch_records


def write_dataclass_csv(path: Path, records: Iterable[object]) -> None:
    records = list(records)
    if not records:
        raise ValueError(f"No records supplied for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(records[0]).keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def resolve_sources(input_dir: Path, requested_files: list[str] | None) -> list[Path]:
    if requested_files:
        sources = [input_dir / name for name in requested_files]
    else:
        sources = sorted(path for path in input_dir.glob("*.jpg") if IMAGE_PATTERN.match(path.name))
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source images: {missing}")
    if not sources:
        raise FileNotFoundError(f"No matching source images found in {input_dir}")
    return sources


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already contains files: {args.output_dir}. "
            "Use a new directory to preserve provenance."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sources = resolve_sources(args.input_dir, args.files)
    all_results: list[DetectionResult] = []
    all_patches: list[PatchRecord] = []

    for source_path in sources:
        result, patches = process_image(
            source_path,
            args.output_dir,
            max_detection_dimension=args.max_detection_dimension,
            qc_max_dimension=args.qc_max_dimension,
        )
        all_results.append(result)
        all_patches.extend(patches)
        print(
            f"{source_path.name}: patches={result.detected_patches}, "
            f"candidates={result.candidate_count}, max_residual={result.max_grid_residual_scaled_px:.2f}px"
        )

    write_dataclass_csv(args.output_dir / "detection_summary.csv", all_results)
    write_dataclass_csv(args.output_dir / "patch_manifest.csv", all_patches)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "input_dir": str(args.input_dir.resolve()),
                "output_dir": str(args.output_dir.resolve()),
                "source_files": [path.name for path in sources],
                "grid_rows": GRID_ROWS,
                "grid_cols": GRID_COLS,
                "expected_patches_per_image": EXPECTED_PATCHES,
                "max_detection_dimension": args.max_detection_dimension,
                "qc_max_dimension": args.qc_max_dimension,
                "well_id_order": "row-major A01-A12 through H01-H12",
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    expected_total = len(sources) * EXPECTED_PATCHES
    if len(all_patches) != expected_total:
        raise RuntimeError(f"Expected {expected_total} patch records, generated {len(all_patches)}")
    if len({record.patch_id for record in all_patches}) != expected_total:
        raise RuntimeError("Duplicate patch IDs were generated")
    print(f"Completed: images={len(sources)}, patches={len(all_patches)}, output={args.output_dir}")


if __name__ == "__main__":
    main()

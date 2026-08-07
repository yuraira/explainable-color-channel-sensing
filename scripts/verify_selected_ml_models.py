"""Verify refitted nested-selected ML estimators."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "modeling" / "selected_ml"


def main() -> None:
    errors: list[str] = []
    efficiency_path = OUTPUT_DIR / "selected_ml_model_efficiency.csv"
    if not efficiency_path.is_file():
        raise SystemExit("Missing selected_ml_model_efficiency.csv")
    efficiency = pd.read_csv(efficiency_path)
    model_files = sorted((OUTPUT_DIR / "models").glob("*.joblib"))
    if len(efficiency) != 10:
        errors.append(f"Expected 10 efficiency rows, found {len(efficiency)}")
    if len(model_files) != 10:
        errors.append(f"Expected 10 model files, found {len(model_files)}")
    if not efficiency.groupby("analyte").size().eq(5).all():
        errors.append("Each analyte must contain five selected models")
    if not (efficiency["tree_count"] == 150).all():
        errors.append("Every selected model must contain 150 trees")
    if efficiency["prediction_max_abs_difference"].max() > 1e-10:
        errors.append("Refitted predictions do not reproduce nested-CV outputs")
    if not (efficiency["model_size_bytes"] > 0).all():
        errors.append("A serialized model is empty")
    for model_path in model_files:
        estimator = joblib.load(model_path)
        if not hasattr(estimator, "regressor_"):
            errors.append(f"Invalid fitted estimator: {model_path.name}")
    report = {
        "models": len(model_files),
        "maximum_prediction_difference": float(
            efficiency["prediction_max_abs_difference"].max()
        ),
        "errors": errors,
    }
    (OUTPUT_DIR / "verification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

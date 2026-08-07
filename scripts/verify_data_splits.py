"""Verify group-aware outer and nested cross-validation assignments."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "outputs" / "color_features" / "features.csv"
OUTPUT_DIR = ROOT / "outputs" / "data_splits"
N_SPLITS = 5
EXPECTED_ROLES = {"train", "validation", "test"}


def check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def main() -> None:
    errors: list[str] = []
    warnings: list[str] = []
    required_files = [
        "outer_fold_assignments.csv",
        "nested_split_assignments.csv",
        "well_split_assignments.csv",
        "well_fold_matrix.csv",
        "split_summary.csv",
        "concentration_balance.csv",
        "run_config.json",
        "figures/cross_validation_split_design.png",
    ]
    for relative_path in required_files:
        check(
            (OUTPUT_DIR / relative_path).is_file(),
            f"Missing output: {relative_path}",
            errors,
        )
    if errors:
        raise SystemExit("\n".join(errors))

    source = pd.read_csv(SOURCE_PATH)
    outer = pd.read_csv(OUTPUT_DIR / "outer_fold_assignments.csv")
    nested = pd.read_csv(OUTPUT_DIR / "nested_split_assignments.csv")
    wells = pd.read_csv(OUTPUT_DIR / "well_split_assignments.csv")
    well_matrix = pd.read_csv(OUTPUT_DIR / "well_fold_matrix.csv")
    summary = pd.read_csv(OUTPUT_DIR / "split_summary.csv")
    balance = pd.read_csv(OUTPUT_DIR / "concentration_balance.csv")
    config = json.loads((OUTPUT_DIR / "run_config.json").read_text(encoding="utf-8"))

    check(len(outer) == len(source), "Outer assignment row count mismatch", errors)
    check(
        set(outer["patch_id"]) == set(source["patch_id"]),
        "Outer assignment patch_id set mismatch",
        errors,
    )
    check(
        set(outer["outer_test_fold"]) == set(range(1, N_SPLITS + 1)),
        "Outer fold labels must be 1 through 5",
        errors,
    )
    check(
        outer.groupby("well_id")["outer_test_fold"].nunique().max() == 1,
        "A well_id has multiple outer test folds",
        errors,
    )
    check(
        len(well_matrix) == 96 and well_matrix["well_id"].nunique() == 96,
        "Well matrix must contain 96 unique rows",
        errors,
    )
    check(
        len(wells) == 96 * N_SPLITS,
        "Well role table must contain 96 rows per outer fold",
        errors,
    )
    check(
        len(nested) == len(source) * N_SPLITS,
        "Nested assignment must contain one source copy per outer fold",
        errors,
    )

    outer_test_well_sets: list[set[str]] = []
    fold_details: list[dict[str, object]] = []
    for fold in range(1, N_SPLITS + 1):
        fold_data = nested.loc[nested["outer_fold"] == fold]
        fold_wells = wells.loc[wells["outer_fold"] == fold]
        check(
            len(fold_data) == len(source),
            f"Outer fold {fold} does not contain all source patches",
            errors,
        )
        check(
            fold_data["patch_id"].nunique() == len(source),
            f"Outer fold {fold} contains duplicate or missing patch_id values",
            errors,
        )
        check(
            set(fold_data["dl_role"]) == EXPECTED_ROLES,
            f"Outer fold {fold} has incomplete DL roles",
            errors,
        )
        check(
            set(fold_data["ml_role"]) == {"train", "test"},
            f"Outer fold {fold} has invalid ML roles",
            errors,
        )
        check(
            fold_data.groupby("well_id")["dl_role"].nunique().max() == 1,
            f"Outer fold {fold} leaks a well_id across DL roles",
            errors,
        )
        check(
            fold_data.groupby("well_id")["ml_role"].nunique().max() == 1,
            f"Outer fold {fold} leaks a well_id across ML roles",
            errors,
        )
        check(
            ((fold_data["dl_role"] == "test") == (fold_data["ml_role"] == "test")).all(),
            f"Outer fold {fold} has inconsistent ML and DL test rows",
            errors,
        )
        check(
            ((fold_data["dl_role"] != "test") == (fold_data["ml_role"] == "train")).all(),
            f"Outer fold {fold} has inconsistent development rows",
            errors,
        )
        check(
            set(fold_wells["inner_fold"]) == set(range(0, N_SPLITS + 1)),
            f"Outer fold {fold} must contain inner_fold 0 through 5",
            errors,
        )
        check(
            ((fold_wells["inner_fold"] == 0) == (fold_wells["dl_role"] == "test")).all(),
            f"Outer fold {fold} has inconsistent test inner_fold values",
            errors,
        )
        check(
            ((fold_wells["inner_fold"] == 1) == (fold_wells["dl_role"] == "validation")).all(),
            f"Outer fold {fold} has inconsistent validation inner_fold values",
            errors,
        )
        check(
            ((fold_wells["inner_fold"] >= 2) == (fold_wells["dl_role"] == "train")).all(),
            f"Outer fold {fold} has inconsistent train inner_fold values",
            errors,
        )

        test_wells = set(fold_wells.loc[fold_wells["dl_role"] == "test", "well_id"])
        validation_wells = set(
            fold_wells.loc[fold_wells["dl_role"] == "validation", "well_id"]
        )
        train_wells = set(fold_wells.loc[fold_wells["dl_role"] == "train", "well_id"])
        check(not (test_wells & validation_wells), f"Fold {fold}: test/validation leakage", errors)
        check(not (test_wells & train_wells), f"Fold {fold}: test/train leakage", errors)
        check(not (validation_wells & train_wells), f"Fold {fold}: validation/train leakage", errors)
        check(
            len(test_wells | validation_wells | train_wells) == 96,
            f"Fold {fold}: well roles do not cover all positions",
            errors,
        )
        outer_test_well_sets.append(test_wells)

        for analyte, analyte_data in fold_data.groupby("analyte"):
            expected_levels = source.loc[
                source["analyte"] == analyte, "concentration_order"
            ].nunique()
            for role in EXPECTED_ROLES:
                selected = analyte_data.loc[analyte_data["dl_role"] == role]
                counts = selected.groupby("concentration_order").size()
                check(
                    len(counts) == expected_levels,
                    f"Fold {fold} {analyte} {role}: missing concentration level",
                    errors,
                )
                check(
                    counts.nunique() == 1,
                    f"Fold {fold} {analyte} {role}: concentration imbalance",
                    errors,
                )

        fold_details.append(
            {
                "outer_fold": fold,
                "train_wells": len(train_wells),
                "validation_wells": len(validation_wells),
                "test_wells": len(test_wells),
                "inner_fold_wells": {
                    str(int(inner_fold)): int(count)
                    for inner_fold, count in (
                        fold_wells.loc[fold_wells["inner_fold"] > 0]
                        .groupby("inner_fold")["well_id"]
                        .nunique()
                        .items()
                    )
                },
            }
        )

    combined_test_wells = set().union(*outer_test_well_sets)
    pairwise_overlap = sum(
        len(outer_test_well_sets[left] & outer_test_well_sets[right])
        for left in range(N_SPLITS)
        for right in range(left + 1, N_SPLITS)
    )
    check(pairwise_overlap == 0, "Outer test well sets overlap", errors)
    check(len(combined_test_wells) == 96, "Outer test folds do not cover all 96 wells", errors)
    edge_counts = (
        well_matrix.groupby("outer_test_fold")["is_edge"].sum().astype(int)
    )
    check(
        int(edge_counts.max() - edge_counts.min()) <= 1,
        "Outer test folds are imbalanced for edge/interior position type",
        errors,
    )

    check(len(summary) == 50, "Expected 50 split summary rows", errors)
    check(len(balance) == 95, "Expected 95 concentration balance rows", errors)
    check(
        (balance["ml_train_count"] + balance["ml_test_count"] == 96).all(),
        "ML concentration counts do not sum to 96",
        errors,
    )
    check(
        (
            balance["dl_train_count"]
            + balance["dl_validation_count"]
            + balance["dl_test_count"]
            == 96
        ).all(),
        "DL concentration counts do not sum to 96",
        errors,
    )
    check(config.get("random_seed") == 240920, "Unexpected random seed", errors)
    check(config.get("n_splits") == N_SPLITS, "Unexpected split count", errors)

    report = {
        "source_patches": len(source),
        "unique_wells": source["well_id"].nunique(),
        "outer_folds": N_SPLITS,
        "fold_details": fold_details,
        "outer_test_pairwise_well_overlap": pairwise_overlap,
        "outer_test_edge_wells": {
            str(int(fold)): int(count) for fold, count in edge_counts.items()
        },
        "errors": errors,
        "warnings": warnings,
    }
    (OUTPUT_DIR / "verification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

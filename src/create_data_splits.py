"""Create reproducible group-aware cross-validation assignments.

The 96 wells are technical positions shared across every concentration image.
To prevent the same spatial position from appearing on both sides of a split,
``well_id`` is the grouping variable for both the outer test fold and the
inner validation split.  Concentration level is used only as a discrete
stratification label; the regression target remains the measured concentration.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


RANDOM_SEED = 240920
N_SPLITS = 5
REQUIRED_COLUMNS = [
    "patch_id",
    "image_id",
    "analyte",
    "concentration_order",
    "concentration_mg_ml",
    "well_id",
    "grid_row",
    "grid_col",
    "crop_file",
]
ROLE_ORDER = ["train", "validation", "test"]
ROLE_COLORS = {
    "train": "#4C78A8",
    "validation": "#F2CF5B",
    "test": "#E45756",
}
FOLD_COLORS = ["#4C78A8", "#F2CF5B", "#59A14F", "#E45756", "#B279A2"]


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
        default=Path("outputs/data_splits"),
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


def validate_source(df: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    if df["patch_id"].duplicated().any():
        raise ValueError("patch_id must be unique")
    if df["well_id"].nunique() != 96:
        raise ValueError(f"Expected 96 well_id values, found {df['well_id'].nunique()}")
    if set(df["grid_row"].unique()) != set(range(1, 9)):
        raise ValueError("grid_row must cover 1 through 8")
    if set(df["grid_col"].unique()) != set(range(1, 13)):
        raise ValueError("grid_col must cover 1 through 12")

    class_counts = df.groupby(["analyte", "concentration_order"]).size()
    if not (class_counts == 96).all():
        raise ValueError(
            "Every analyte-concentration level must contain one patch from each of 96 wells"
        )

    within_class_wells = df.groupby(["analyte", "concentration_order"])[
        "well_id"
    ].nunique()
    if not (within_class_wells == 96).all():
        raise ValueError("Every concentration level must contain all 96 unique well_id values")


def make_strata(df: pd.DataFrame) -> pd.Series:
    position_type = np.where(
        df["grid_row"].isin([1, 8]) | df["grid_col"].isin([1, 12]),
        "edge",
        "interior",
    )
    return (
        df["analyte"].astype(str)
        + "_level_"
        + df["concentration_order"].astype(int).astype(str)
        + "_"
        + pd.Series(position_type, index=df.index)
    )


def create_outer_assignment(df: pd.DataFrame) -> pd.Series:
    strata = make_strata(df)
    groups = df["well_id"].astype(str)
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    assignment = pd.Series(0, index=df.index, dtype="int64")
    seen_test_wells: set[str] = set()

    for fold, (train_index, test_index) in enumerate(
        splitter.split(df, y=strata, groups=groups), start=1
    ):
        train_wells = set(groups.iloc[train_index])
        test_wells = set(groups.iloc[test_index])
        if train_wells & test_wells:
            raise RuntimeError(f"Outer fold {fold} contains well_id leakage")
        if seen_test_wells & test_wells:
            raise RuntimeError("A well_id was assigned to more than one outer test fold")
        seen_test_wells.update(test_wells)
        assignment.loc[df.index[test_index]] = fold

    if (assignment == 0).any() or seen_test_wells != set(groups):
        raise RuntimeError("Outer fold assignment is incomplete")
    if assignment.groupby(groups).nunique().max() != 1:
        raise RuntimeError("A well_id has multiple outer test-fold assignments")
    return assignment


def create_inner_fold_map(
    df: pd.DataFrame,
    outer_assignment: pd.Series,
    outer_fold: int,
) -> dict[str, int]:
    development = df.loc[outer_assignment != outer_fold].copy()
    strata = make_strata(development)
    groups = development["well_id"].astype(str)
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED + outer_fold,
    )
    assignment = pd.Series(0, index=development.index, dtype="int64")
    for inner_fold, (_, validation_index) in enumerate(
        splitter.split(development, y=strata, groups=groups), start=1
    ):
        assignment.loc[development.index[validation_index]] = inner_fold
    if (assignment == 0).any():
        raise RuntimeError(f"Outer fold {outer_fold} has incomplete inner-fold assignment")
    if assignment.groupby(groups).nunique().max() != 1:
        raise RuntimeError(f"Outer fold {outer_fold} splits a well_id across inner folds")
    mapping = (
        pd.DataFrame({"well_id": groups, "inner_fold": assignment})
        .drop_duplicates()
        .set_index("well_id")["inner_fold"]
        .astype(int)
        .to_dict()
    )
    if set(mapping.values()) != set(range(1, N_SPLITS + 1)):
        raise RuntimeError(f"Outer fold {outer_fold} does not contain all inner folds")
    return mapping


def build_assignments(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = df[REQUIRED_COLUMNS].copy()
    outer_assignment = create_outer_assignment(df)
    outer = source.copy()
    outer["outer_test_fold"] = outer_assignment

    nested_frames: list[pd.DataFrame] = []
    well_frames: list[pd.DataFrame] = []
    all_wells = set(df["well_id"].astype(str))
    well_coordinates = (
        df[["well_id", "grid_row", "grid_col"]]
        .drop_duplicates()
        .sort_values(["grid_row", "grid_col"])
    )
    well_coordinates["is_edge"] = (
        well_coordinates["grid_row"].isin([1, 8])
        | well_coordinates["grid_col"].isin([1, 12])
    ).astype(int)

    for outer_fold in range(1, N_SPLITS + 1):
        test_wells = set(
            df.loc[outer_assignment == outer_fold, "well_id"].astype(str)
        )
        inner_fold_map = create_inner_fold_map(
            df, outer_assignment, outer_fold
        )
        validation_wells = {
            well_id for well_id, inner_fold in inner_fold_map.items() if inner_fold == 1
        }
        train_wells = all_wells - test_wells - validation_wells
        if (
            train_wells & validation_wells
            or train_wells & test_wells
            or validation_wells & test_wells
        ):
            raise RuntimeError(f"Outer fold {outer_fold} contains role leakage")
        if train_wells | validation_wells | test_wells != all_wells:
            raise RuntimeError(f"Outer fold {outer_fold} has incomplete well roles")

        nested = source.copy()
        nested.insert(0, "outer_fold", outer_fold)
        nested["inner_fold"] = (
            nested["well_id"].map(inner_fold_map).fillna(0).astype(int)
        )
        nested["ml_role"] = np.where(
            nested["well_id"].isin(test_wells), "test", "train"
        )
        nested["dl_role"] = np.select(
            [
                nested["well_id"].isin(test_wells),
                nested["well_id"].isin(validation_wells),
            ],
            ["test", "validation"],
            default="train",
        )
        nested_frames.append(nested)

        well_roles = well_coordinates.copy()
        well_roles.insert(0, "outer_fold", outer_fold)
        well_roles["inner_fold"] = (
            well_roles["well_id"].map(inner_fold_map).fillna(0).astype(int)
        )
        well_roles["ml_role"] = np.where(
            well_roles["well_id"].isin(test_wells), "test", "train"
        )
        well_roles["dl_role"] = np.select(
            [
                well_roles["well_id"].isin(test_wells),
                well_roles["well_id"].isin(validation_wells),
            ],
            ["test", "validation"],
            default="train",
        )
        well_frames.append(well_roles)

    nested_all = pd.concat(nested_frames, ignore_index=True)
    well_all = pd.concat(well_frames, ignore_index=True)
    return outer, nested_all, well_all


def build_well_matrix(well_assignments: pd.DataFrame) -> pd.DataFrame:
    coordinates = (
        well_assignments[["well_id", "grid_row", "grid_col", "is_edge"]]
        .drop_duplicates()
        .sort_values(["grid_row", "grid_col"])
        .reset_index(drop=True)
    )
    outer_map = (
        well_assignments.loc[well_assignments["dl_role"] == "test"]
        .set_index("well_id")["outer_fold"]
        .to_dict()
    )
    coordinates["outer_test_fold"] = coordinates["well_id"].map(outer_map).astype(int)
    for fold in range(1, N_SPLITS + 1):
        role_map = (
            well_assignments.loc[well_assignments["outer_fold"] == fold]
            .set_index("well_id")["dl_role"]
            .to_dict()
        )
        coordinates[f"dl_role_outer_{fold}"] = coordinates["well_id"].map(role_map)
    for fold in range(1, N_SPLITS + 1):
        inner_map = (
            well_assignments.loc[well_assignments["outer_fold"] == fold]
            .set_index("well_id")["inner_fold"]
            .to_dict()
        )
        coordinates[f"inner_fold_outer_{fold}"] = coordinates["well_id"].map(inner_map)
    return coordinates


def build_split_summary(nested: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    analyte_totals = nested.loc[nested["outer_fold"] == 1].groupby("analyte").size()
    for outer_fold in range(1, N_SPLITS + 1):
        fold_data = nested.loc[nested["outer_fold"] == outer_fold]
        for analyte, analyte_data in fold_data.groupby("analyte", sort=True):
            for pipeline, role_column, roles in [
                ("ML", "ml_role", ["train", "test"]),
                ("DL", "dl_role", ROLE_ORDER),
            ]:
                for role in roles:
                    selected = analyte_data.loc[analyte_data[role_column] == role]
                    per_level = selected.groupby("concentration_order").size()
                    rows.append(
                        {
                            "outer_fold": outer_fold,
                            "analyte": analyte,
                            "pipeline": pipeline,
                            "role": role,
                            "patch_count": len(selected),
                            "well_count": selected["well_id"].nunique(),
                            "percentage_of_analyte": len(selected)
                            / int(analyte_totals.loc[analyte]),
                            "concentration_levels": selected[
                                "concentration_order"
                            ].nunique(),
                            "min_patches_per_level": int(per_level.min()),
                            "max_patches_per_level": int(per_level.max()),
                        }
                    )
    return pd.DataFrame(rows)


def build_concentration_balance(nested: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (outer_fold, analyte, order, concentration), subset in nested.groupby(
        [
            "outer_fold",
            "analyte",
            "concentration_order",
            "concentration_mg_ml",
        ],
        sort=True,
    ):
        rows.append(
            {
                "outer_fold": outer_fold,
                "analyte": analyte,
                "concentration_order": order,
                "concentration_mg_ml": concentration,
                "ml_train_count": int((subset["ml_role"] == "train").sum()),
                "ml_test_count": int((subset["ml_role"] == "test").sum()),
                "dl_train_count": int((subset["dl_role"] == "train").sum()),
                "dl_validation_count": int(
                    (subset["dl_role"] == "validation").sum()
                ),
                "dl_test_count": int((subset["dl_role"] == "test").sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_split_design(
    well_matrix: pd.DataFrame,
    well_assignments: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Malgun Gothic",
            "axes.unicode_minus": False,
            "font.size": 10,
        }
    )
    fig = plt.figure(figsize=(14, 6.8), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.45, 1])
    ax_map = fig.add_subplot(grid[0, 0])
    ax_bar = fig.add_subplot(grid[0, 1])

    fold_grid = np.full((8, 12), np.nan)
    for row in well_matrix.itertuples(index=False):
        fold_grid[int(row.grid_row) - 1, int(row.grid_col) - 1] = int(
            row.outer_test_fold
        )
    cmap = ListedColormap(FOLD_COLORS)
    norm = BoundaryNorm(np.arange(0.5, N_SPLITS + 1.5), cmap.N)
    ax_map.imshow(fold_grid, cmap=cmap, norm=norm, aspect="equal")
    ax_map.set_xticks(range(12), [f"{column:02d}" for column in range(1, 13)])
    ax_map.set_yticks(range(8), list("ABCDEFGH"))
    ax_map.set_xlabel("Plate column")
    ax_map.set_ylabel("Plate row")
    ax_map.set_title("Outer test fold assigned to each well_id", fontweight="bold")
    ax_map.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    ax_map.set_yticks(np.arange(-0.5, 8, 1), minor=True)
    ax_map.grid(which="minor", color="white", linewidth=1.5)
    ax_map.tick_params(which="minor", bottom=False, left=False)
    for row in range(8):
        for column in range(12):
            ax_map.text(
                column,
                row,
                str(int(fold_grid[row, column])),
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold",
            )
    fold_handles = [
        Patch(color=FOLD_COLORS[index - 1], label=f"Fold {index}")
        for index in range(1, N_SPLITS + 1)
    ]
    ax_map.legend(
        handles=fold_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=5,
        frameon=False,
    )

    group_counts = (
        well_assignments.groupby(["outer_fold", "dl_role"])["well_id"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(columns=ROLE_ORDER, fill_value=0)
    )
    left = np.zeros(N_SPLITS)
    y = np.arange(1, N_SPLITS + 1)
    for role in ROLE_ORDER:
        values = group_counts[role].to_numpy()
        ax_bar.barh(
            y,
            values,
            left=left,
            color=ROLE_COLORS[role],
            label=role.title(),
            height=0.62,
        )
        for index, value in enumerate(values):
            if value:
                ax_bar.text(
                    left[index] + value / 2,
                    y[index],
                    str(int(value)),
                    ha="center",
                    va="center",
                    color="white" if role != "validation" else "#333333",
                    fontsize=9,
                    fontweight="bold",
                )
        left += values
    ax_bar.set_yticks(y, [f"Outer fold {fold}" for fold in y])
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 96)
    ax_bar.set_xlabel("Number of well_id groups")
    ax_bar.set_title("Nested train / validation / test roles", fontweight="bold")
    ax_bar.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    ax_bar.spines[["top", "right", "left"]].set_visible(False)
    ax_bar.grid(axis="x", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax_bar.set_axisbelow(True)

    fig.suptitle(
        "Group-aware stratified 5-fold cross-validation design",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.015,
        "All 19 analyte-concentration patches from the same well_id share one role; seed = 240920.",
        ha="center",
        fontsize=10,
        color="#4D4D4D",
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    prepare_output_dir(args.output_dir, args.overwrite)
    features = pd.read_csv(args.features)
    validate_source(features)

    outer, nested, well_assignments = build_assignments(features)
    well_matrix = build_well_matrix(well_assignments)
    split_summary = build_split_summary(nested)
    concentration_balance = build_concentration_balance(nested)

    outer.to_csv(args.output_dir / "outer_fold_assignments.csv", index=False)
    nested.to_csv(args.output_dir / "nested_split_assignments.csv", index=False)
    well_assignments.to_csv(args.output_dir / "well_split_assignments.csv", index=False)
    well_matrix.to_csv(args.output_dir / "well_fold_matrix.csv", index=False)
    split_summary.to_csv(args.output_dir / "split_summary.csv", index=False)
    concentration_balance.to_csv(
        args.output_dir / "concentration_balance.csv", index=False
    )

    figure_path = args.output_dir / "figures" / "cross_validation_split_design.png"
    plot_split_design(well_matrix, well_assignments, figure_path)

    config = {
        "source_features": args.features.as_posix(),
        "random_seed": RANDOM_SEED,
        "n_splits": N_SPLITS,
        "outer_splitter": "StratifiedGroupKFold",
        "outer_stratification": "analyte + concentration_order + edge/interior",
        "outer_group": "well_id",
        "inner_splitter": "StratifiedGroupKFold with all five inner folds",
        "inner_random_seed_rule": "random_seed + outer_fold",
        "inner_stratification": "analyte + concentration_order + edge/interior",
        "inner_group": "well_id",
        "ml_usage": "all five inner folds for tuning; outer test for final fold evaluation",
        "dl_usage": "inner_fold 1 as validation; inner_fold 2-5 as train; outer test for evaluation",
        "interpretation": "internal patch-level validation grouped by spatial well_id",
        "known_limitation": (
            "Different well positions from the same source image remain across roles "
            "because only one source image exists per concentration."
        ),
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    test_well_counts = (
        well_assignments.loc[well_assignments["dl_role"] == "test"]
        .groupby("outer_fold")["well_id"]
        .nunique()
        .to_dict()
    )
    print(f"Created group-aware splits for {len(features):,} patches")
    print(f"Outer test well counts: {test_well_counts}")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()

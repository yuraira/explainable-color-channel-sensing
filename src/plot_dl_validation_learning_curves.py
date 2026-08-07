"""Plot fold-wise CNN validation MAE histories from the completed training run."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANALYTES = ["glucose", "ketone"]
INPUT_MODES = ["roi_masked", "full_patch"]
ANALYTE_LABELS = {"glucose": "Glucose", "ketone": "Ketone"}
INPUT_LABELS = {"roi_masked": "Central ROI", "full_patch": "Full patch"}


def main() -> None:
    history_path = Path("outputs/modeling/dl/training_history.csv")
    output_path = Path(
        "outputs/modeling/dl/figures/dl_validation_learning_curves.png"
    )
    history = pd.read_csv(history_path)
    selected = history.loc[history["selected_target"] == 1].copy()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, 5))
    for row, analyte in enumerate(ANALYTES):
        for column, input_mode in enumerate(INPUT_MODES):
            axis = axes[row, column]
            subset = selected.loc[
                (selected["analyte"] == analyte)
                & (selected["input_mode"] == input_mode)
            ]
            for color, outer_fold in zip(colors, range(1, 6), strict=True):
                fold = subset.loc[subset["outer_fold"] == outer_fold].sort_values(
                    "epoch"
                )
                axis.plot(
                    fold["epoch"],
                    fold["validation_mae"],
                    color=color,
                    linewidth=1.5,
                    label=f"Fold {outer_fold}",
                )
                best = fold.loc[fold["validation_mae"].idxmin()]
                axis.scatter(
                    [best["epoch"]],
                    [best["validation_mae"]],
                    color=color,
                    edgecolor="white",
                    linewidth=0.7,
                    s=38,
                    zorder=3,
                )
            axis.set_title(
                f"{ANALYTE_LABELS[analyte]} — {INPUT_LABELS[input_mode]}",
                fontweight="bold",
            )
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Validation MAE (mg/mL)")
            axis.grid(color="#D9D9D9", linewidth=0.8)
            axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle(
        "TinyColorCNN validation histories and selected epochs",
        fontsize=16,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    best = (
        selected.sort_values("validation_mae")
        .groupby(["analyte", "input_mode", "outer_fold"], as_index=False)
        .first()
    )
    summary = (
        best.groupby(["analyte", "input_mode"], as_index=False)
        .agg(
            folds=("outer_fold", "nunique"),
            best_epoch_mean=("epoch", "mean"),
            best_epoch_std=("epoch", "std"),
            validation_mae_mean=("validation_mae", "mean"),
            validation_mae_std=("validation_mae", "std"),
        )
    )
    summary.to_csv(
        "outputs/modeling/dl/dl_validation_learning_curve_summary.csv", index=False
    )


if __name__ == "__main__":
    main()

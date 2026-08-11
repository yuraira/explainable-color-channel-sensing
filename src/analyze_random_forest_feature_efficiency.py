"""Measure Random Forest performance and deployment cost after feature reduction."""

from analyze_extra_trees_feature_efficiency import run_analysis


def main() -> None:
    run_analysis("RandomForest", "random_forest", "Random Forest")


if __name__ == "__main__":
    main()

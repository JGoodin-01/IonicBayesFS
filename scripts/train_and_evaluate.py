import pandas as pd
from src.training.ExperimentRunner import ExperimentRunner
from src.EXPERIMENT_CONFIGS import EXPERIMENT_CONFIGS


if __name__ == "__main__":
    data = pd.read_csv("./data/processed_with_pca.csv").dropna(subset=["η / mPa s"])
    X, y = data.drop("η / mPa s", axis=1), data["η / mPa s"].values
    runner = ExperimentRunner(EXPERIMENT_CONFIGS)

    feature_selection_strategies = [
        {"name": "Base"},
        {"name": "SelectKBest"},
        # {"name": "RFE"},
        # {"name": "BFS"},
    ]

    runner.run_cross_experiment(X, y, feature_selection_strategies)

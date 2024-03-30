import pandas as pd
from src.EXPERIMENT_CONFIGS import EXPERIMENT_CONFIGS
from src.training.ExperimentRunner import ExperimentRunner


# Function to check CUDA availability
def cuda_available():
    try:
        import cuml

        return True
    except ImportError:
        return False


if cuda_available():
    print("Using RAPIDS cuML for GPU acceleration.")
else:
    print("CUDA not available. Falling back to scikit-learn.")


if __name__ == "__main__":
    models = ["lr", "rf"]
    data = pd.read_csv("./data/processed.csv").dropna(subset=["η / mPa s"])
    X, y = data.drop("η / mPa s", axis=1), data["η / mPa s"].values
    runner = ExperimentRunner()

    feature_selection_strategies = [
        {"name": "Base"},
        {"name": "SelectKBest"},
        {"name": "RFE"},
        {"name": "BFS"},
    ]
    for model_key in models:
        model_config = EXPERIMENT_CONFIGS[model_key]
        model_class = model_config["model"]
        model_params = model_config["param_grid"]
        runner.run_cross_experiment(
            X, y, model_class, model_params, feature_selection_strategies
        )

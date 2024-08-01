import os
import subprocess
import pandas as pd
from src.training.ExperimentRunner import ExperimentRunner
from src.EXPERIMENT_CONFIGS import EXPERIMENT_CONFIGS


def ensure_directory_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def run_script(script_path):
    try:
        with subprocess.Popen(
            ["python", script_path],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ) as proc:
            # Read line by line as it's being output
            while True:
                output = proc.stdout.readline()
                if output == "" and proc.poll() is not None:
                    break
                if output:
                    print(output.strip())
            rc = proc.poll()
            return rc

    except subprocess.CalledProcessError as e:
        print(f"Error running script {script_path}: {e}")


def run_training(
    source_csv="./data/processed.csv",
    feature_selection_strategies=["Base", "RFE"],
    save_folder=None,
):
    if save_folder:
        ensure_directory_exists(save_folder)
        runner = ExperimentRunner(EXPERIMENT_CONFIGS, save_folder)
    else:
        runner = ExperimentRunner(EXPERIMENT_CONFIGS)

    data = pd.read_csv(source_csv).dropna(subset=["η / mPa s"])
    X, y = data.drop("η / mPa s", axis=1), data["η / mPa s"].values
    runner.run_cross_experiment(X, y, feature_selection_strategies, n_splits=10)


if __name__ == "__main__":
    # Run pre-training scripts
    # run_script("./src/preprocessing/preprocess.py")
    # run_script("./src/preprocessing/PCAs.py")
    run_script("./src/evaluation/dataset_plots.py")

    # # Run training & evaluation
    # feature_selection_strategies = [
    #     {"name": "Base"},
    #     # {"name": "SelectKBest"},
    #     # {"name": "RFE"},
    #     # {"name": "BFS"},
    # ]

    # # Pre feature engineering & selection
    # save_folder = "./pre_fe_results/"
    # run_training("./data/processed.csv", feature_selection_strategies, save_folder)

    # # Post feature engineering & selection
    # save_folder = "./post_fe_results/"
    # run_training("./data/processed_with_pca.csv", feature_selection_strategies, save_folder)
    
    # run_script("./src/evaluation/model_plots.py")

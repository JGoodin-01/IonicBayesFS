import subprocess
import os


def run_preprocessing():
    """Run the data preprocessing script."""
    subprocess.run(["python", "./scripts/run_preprocessing.py"], check=True)


def train_and_evaluate():
    """Run the training and evaluation script for a given model."""
    subprocess.run(["python", "-m", "scripts.train_and_evaluate"], check=True)


def run_dataset_plotting():
    """Run the data plotting script."""
    subprocess.run(["python", "./scripts/run_dataset_plotting.py"], check=True)


def run_model_plotting():
    """Run the data plotting script."""
    subprocess.run(["python", "./scripts/run_model_plotting.py"], check=True)


if __name__ == "__main__":
    run_preprocessing()
    # run_dataset_plotting()
    train_and_evaluate()
    run_model_plotting()

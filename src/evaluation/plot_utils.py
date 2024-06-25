# Standard Library Imports
import os
import re
import signal
from contextlib import contextmanager
from functools import wraps

# Third-Party Library Imports
import numpy as np
import matplotlib.pyplot as plt


## Decorator for plotting functions
def plot_wrapper(
    figsize=(3.375, 3.375),
    xlabel="",
    ylabel="",
    scale=None,
    filename="image.svg",
    dynamic_params_func=None,
    get_image_directory=lambda: "./",
):
    def decorator(plot_func):
        @wraps(plot_func)
        def wrapper(*args, **kwargs):
            nonlocal filename  # Ensures filename can be modified by dynamic_params_func
            image_directory = get_image_directory()

            # Dynamic parameter processing
            if dynamic_params_func is not None:
                dynamic_params = dynamic_params_func(*args, **kwargs)
                dynamic_filename = dynamic_params.get("filename", filename)
            else:
                dynamic_filename = filename

            if figsize is not None:
                plt.figure(figsize=figsize)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if scale is not None:
                plt.yscale(scale)
                plt.xscale(scale)

            plot_func(*args, **kwargs)

            if not os.path.exists(image_directory):
                os.makedirs(image_directory)
            plt.savefig(os.path.join(image_directory, dynamic_filename), format="svg")
            plt.close()

        return wrapper

    return decorator


## Utility functions for Dataset Plots
class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def sanitize_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', "_", filename)


## Utility functions for Model Plots
def average_folds_predictions(data, techniques, fold_numbers):
    """
    For each technique, calculate the average prediction across all folds and store it in a new column.

    Parameters:
    data (DataFrame): The pandas DataFrame containing the prediction data.
    techniques (list): A list of the feature selection techniques used.
    fold_numbers (list): A list of fold numbers to be averaged.

    Returns:
    DataFrame: The original DataFrame with new columns for the average predictions of each technique.
    """
    for technique in techniques:
        fold_columns = [f"{technique}_Predicted_{fold}" for fold in fold_numbers]
        data[f"{technique}_Predicted_Avg"] = data[fold_columns].mean(axis=1)
        data[f"{technique}_Predicted_Std"] = data[fold_columns].std(axis=1)

    return data


def determine_cluster(values, num_clusters=5):
    quantiles = np.linspace(0, 1, num_clusters + 1)
    bin_edges = np.quantile(values, quantiles)

    clusters = np.digitize(values, bin_edges, right=False)
    clusters = clusters - 1

    return clusters


def calculate_errors(data, techniques, fold_numbers):
    """Calculate absolute errors for each feature selection technique."""
    for technique in techniques:
        fold_columns = [f"{technique}_Predicted_{fold}" for fold in fold_numbers]
        predicted_column = f"{technique}_Predicted_Avg"
        data[f"{technique}_Error"] = np.abs(data["Actual"] - data[predicted_column])
        data[f"{technique}_Predicted_SEM"] = data[fold_columns].std(axis=1) / np.sqrt(
            len(fold_numbers)
        )
    return data

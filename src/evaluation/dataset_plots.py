import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from functools import wraps
from tqdm import tqdm

IMAGE_DIRECTORY = "./dataset_images"

import signal
from contextlib import contextmanager

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


def plot_wrapper(
    figsize=(8, 6),
    xlabel="",
    ylabel="",
    scale=None,
    filename="image.svg",
    dynamic_params_func=None,
):
    def decorator(plot_func):
        @wraps(plot_func)
        def wrapper(*args, **kwargs):
            global IMAGE_DIRECTORY

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

            if not os.path.exists(IMAGE_DIRECTORY):
                os.makedirs(IMAGE_DIRECTORY)
            plt.savefig(os.path.join(IMAGE_DIRECTORY, dynamic_filename), format="svg")
            plt.close()

        return wrapper

    return decorator


@plot_wrapper(
    figsize=(12, 6),
    ylabel="Variance",
    filename="variance_plot.svg",
    dynamic_params_func=lambda data, column: {
        "filename": sanitize_filename(f"{column}_variance.svg")
    },
)
def plot_variance(data, column, **kwargs):
    # Plot a histogram for the descriptor
    sns.histplot(data[column], kde=True)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.xlabel(column)
    plt.tight_layout()


def main():
    global IMAGE_DIRECTORY
    IMAGE_DIRECTORY = f"./dataset_images/variances/"
    data = pd.read_csv("./data/processed.csv")

    data.drop(columns=["SMILES"], inplace=True)
    for column in tqdm(data.columns, desc="Plotting variances"):
        try:
            # Set a time limit for each plotting operation (e.g., 30 seconds)
            with time_limit(30):
                plot_variance(data, column)
        except TimeoutException as e:
            print(f"Timed out on column {column}")
        except Exception as e:
            print(f"An error occurred while plotting column {column}: {e}")



if __name__ == "__main__":
    main()

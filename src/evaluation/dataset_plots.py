import os
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from functools import wraps
from tqdm import tqdm
import signal
from contextlib import contextmanager
import numpy as np

IMAGE_DIRECTORY = "./dataset_images"


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


def define_variance(data):
    targetless_data = data.drop(columns=["η / mPa s"])

    for col in data.select_dtypes(include=["object"]).columns:
        targetless_data[col] = LabelEncoder().fit_transform(data[col])

    imputer = SimpleImputer(strategy="mean")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(imputer.fit_transform(targetless_data))

    pca = PCA()
    pca.fit(scaled_data)

    return pca.explained_variance_ratio_


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


@plot_wrapper(
    figsize=(12, 6),
    ylabel="Variance Explained",
    xlabel="Principal Component",
    filename="pca_plot.svg",
)
def plot_PCA_ratio(data):
    explained_variance = define_variance(data)
    PCAS_COUNT = 30
    
    plt.plot(explained_variance[:PCAS_COUNT], "o-")
    plt.xticks(
        range(PCAS_COUNT),
        [f"PC{i+1}" for i in range(PCAS_COUNT)],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    plt.tight_layout()


@plot_wrapper(
    figsize=(12, 6),
    ylabel="Variance Explained",
    xlabel="Principal Component",
    filename="pca_95_plot.svg",
)
def plot_95_variance(data):
    explained_variance = define_variance(data)
    cumulative_variance = np.cumsum(explained_variance)
    num_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(explained_variance, "o-", label="Individual Explained Variance")
    plt.plot(cumulative_variance, "o-", label="Cumulative Explained Variance")

    plt.axhline(y=0.95, color="r", linestyle="--")
    plt.axvline(x=num_components_95 - 1, color="g", linestyle="--")

    plt.legend(loc="best")
    plt.text(
        num_components_95,
        0.95,
        f" 95% cut-off\n {num_components_95} components",
        color="g",
    )

    print(f"Number of components that explain 95% variance: {num_components_95}")


def main():
    global IMAGE_DIRECTORY
    IMAGE_DIRECTORY = f"./dataset_images/variances/"
    data = pd.read_csv("./data/processed.csv")

    data.drop(columns=["SMILES"], inplace=True)
    # for column in tqdm(data.columns, desc="Plotting variances"):
    #     try:
    #         with time_limit(30):
    #             plot_variance(data, column)
    #     except TimeoutException as e:
    #         print(f"Timed out on column {column}")
    #     except Exception as e:
    #         print(f"An error occurred while plotting column {column}: {e}")

    IMAGE_DIRECTORY = f"./dataset_images/"
    plot_PCA_ratio(data)
    plot_95_variance(data)


if __name__ == "__main__":
    main()

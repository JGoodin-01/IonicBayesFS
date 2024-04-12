# Standard Library Imports
import os
import sys

# Third-Party Library Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from tqdm import tqdm

# Local Application Imports
from plot_utils import plot_wrapper, TimeoutException, time_limit, sanitize_filename

IMAGE_DIRECTORY = f"./dataset_images/"


def get_current_image_directory():
    return IMAGE_DIRECTORY


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


@plot_wrapper(
    figsize=(12, 6),
    ylabel="Variance",
    dynamic_params_func=lambda data, column: {
        "filename": sanitize_filename(f"{column}_variance.svg")
    },
    get_image_directory=get_current_image_directory,
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
    get_image_directory=get_current_image_directory,
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
    dynamic_params_func=lambda data, variance_threshold, max_components: {
        "filename": sanitize_filename(f"pca_{variance_threshold}_plot.svg")
    },
    get_image_directory=get_current_image_directory,
)
def plot_PCA_variance_capture(data, variance_threshold, max_components=40, **kwargs):
    explained_variance = define_variance(data)
    cumulative_variance = np.cumsum(explained_variance)
    num_components = np.argmax(cumulative_variance >= variance_threshold) + 1

    explained_variance_limited = explained_variance[:max_components]
    cumulative_variance_limited = cumulative_variance[:max_components]

    plt.figure(figsize=(12, 6))
    plt.plot(explained_variance_limited, "o-", label="Individual Explained Variance")
    plt.plot(cumulative_variance_limited, "o-", label="Cumulative Explained Variance")

    plt.axhline(y=variance_threshold, color="r", linestyle="--")

    if num_components <= max_components:
        plt.axvline(x=num_components - 1, color="g", linestyle="--")
        plt.text(
            num_components,
            variance_threshold,
            f" {variance_threshold} cut-off\n {num_components} components",
            color="g",
        )

    plt.xticks(
        range(max_components),
        [f"PC{i+1}" for i in range(max_components)],
        rotation=45,
        ha="right",
        fontsize=8,
    )

    plt.xlim([0, max_components - 1])
    plt.legend(loc="best")
    plt.tight_layout()

    print(f"PCAs that explain {variance_threshold} variance: {num_components}")


def main():
    global IMAGE_DIRECTORY

    data = pd.read_csv("./data/processed.csv")
    data.drop(columns=["SMILES"], inplace=True)

    IMAGE_DIRECTORY = f"./dataset_images/variances/"
    for column in tqdm(data.columns, desc="Plotting variances"):
        try:
            with time_limit(30):
                plot_variance(data, column)
        except TimeoutException as e:
            print(f"Timed out on column {column}")
        except Exception as e:
            print(f"An error occurred while plotting column {column}: {e}")

    IMAGE_DIRECTORY = f"./dataset_images/PCAs/"
    plot_PCA_ratio(data)
    plot_PCA_variance_capture(data, variance_threshold=0.95, max_components=40)
    plot_PCA_variance_capture(data, variance_threshold=0.75, max_components=40)


if __name__ == "__main__":
    main()

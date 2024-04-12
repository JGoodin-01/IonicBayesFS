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


def define_variance(data, n_components=None):
    for col in data.select_dtypes(include=["object"]).columns:
        data[col] = LabelEncoder().fit_transform(data[col])

    imputer = SimpleImputer(strategy="mean")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(imputer.fit_transform(data))

    if n_components is not None:
        pca = PCA(n_components=n_components)
    else:
        pca = PCA()
    pca.fit(scaled_data)

    return pca


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
    explained_variance = define_variance(data).explained_variance_ratio_
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
    ylabel="Variance",
    xlabel="Principal Components",
    dynamic_params_func=lambda data, variance_threshold, max_components: {
        "filename": sanitize_filename(f"pca_{variance_threshold}_plot.svg")
    },
    get_image_directory=get_current_image_directory,
)
def plot_PCA_variance_capture(data, variance_threshold, max_components=40, **kwargs):
    explained_variance = define_variance(data).explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    num_components = np.argmax(cumulative_variance >= variance_threshold) + 1

    explained_variance_limited = explained_variance[:max_components]
    cumulative_variance_limited = cumulative_variance[:max_components]

    plt.plot(explained_variance_limited, "o-", label="Individual Explained Variance")
    plt.plot(cumulative_variance_limited, "o-", label="Cumulative Explained Variance")

    plt.axhline(y=variance_threshold, color="r", linestyle="--")

    if num_components <= max_components:
        plt.axvline(x=num_components - 1, color="g", linestyle="--")
        plt.text(
            num_components,  # Slightly offset from the crossing point
            variance_threshold - 0.1,  # Slightly higher above the crossing point
            f" {variance_threshold} cut-off\n {num_components} components",
            color="g",
            fontsize=10,
            ha="left",  # Horizontal alignment to the left of the point
            va="bottom",  # Vertical alignment below the point
        )

    plt.xticks(
        range(max_components),
        [f"PC{i+1}" for i in range(max_components)],
        rotation=45,
        ha="right",
    )

    plt.xlim([0, max_components - 1])
    plt.legend(loc="best")
    plt.tight_layout()


@plot_wrapper(
    figsize=(15, 6),
    xlabel="Features",
    ylabel="Sum of Squared Loadings",
    filename="feature_variances.svg",
    dynamic_params_func=lambda data, n_components: {
        "filename": sanitize_filename(f"feature_variances_{n_components}.svg")
    },
    get_image_directory=get_current_image_directory,
)
def plot_feature_variances(data, n_components=10):
    pca = define_variance(data, n_components=n_components)

    # Sum the squared loadings for each feature
    loadings = pca.components_
    feature_variances = np.sum(loadings**2, axis=0)

    # Create a DataFrame of the feature variances
    feature_variance_df = pd.DataFrame(
        feature_variances, index=data.columns, columns=["Variance"]
    )
    feature_variance_df.sort_values(by="Variance", ascending=False, inplace=True)

    # Plot
    sns.barplot(x=feature_variance_df.index, y=feature_variance_df["Variance"])
    plt.xticks(rotation=90)
    plt.tight_layout()


def main():
    global IMAGE_DIRECTORY

    data = pd.read_csv("./data/processed.csv")
    data.drop(columns=["SMILES"], inplace=True)

    # IMAGE_DIRECTORY = f"./dataset_images/variances/"
    # for column in tqdm(data.columns, desc="Plotting variances"):
    #     try:
    #         with time_limit(30):
    #             plot_variance(data, column)
    #     except TimeoutException as e:
    #         print(f"Timed out on column {column}")
    #     except Exception as e:
    #         print(f"An error occurred while plotting column {column}: {e}")

    IMAGE_DIRECTORY = f"./dataset_images/PCAs/"
    continuous_data = data.drop(columns=["η / mPa s"])
    unique_threshold = 10  # or some other number that makes sense for your data
    continuous_columns = [col for col in continuous_data.columns if 
                          (continuous_data[col].dtype in ['float64', 'int64']) and
                          (continuous_data[col].nunique() > unique_threshold)]
    continuous_data = continuous_data[continuous_columns]
    
    plot_PCA_ratio(continuous_data)
    plot_PCA_variance_capture(
        continuous_data, variance_threshold=0.95, max_components=30
    )
    plot_PCA_variance_capture(
        continuous_data, variance_threshold=0.75, max_components=30
    )
    plot_feature_variances(continuous_data, n_components=6)
    plot_feature_variances(continuous_data, n_components=25)


if __name__ == "__main__":
    main()

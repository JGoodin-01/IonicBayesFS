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
    dynamic_params_func=lambda data, max_components: {
        "filename": sanitize_filename(f"pca_plot.svg")
    },
    get_image_directory=get_current_image_directory,
)
def plot_PCA_variance_capture(data, max_components=40, **kwargs):
    explained_variance = define_variance(data).explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # Find the number of components for 0.75 & 0.95 variance capture
    num_components_075 = np.argmax(cumulative_variance >= 0.75) + 1
    num_components_095 = np.argmax(cumulative_variance >= 0.95) + 1

    explained_variance_limited = explained_variance[:max_components]
    cumulative_variance_limited = cumulative_variance[:max_components]

    plt.plot(explained_variance_limited, "o-", label="Individual Explained Variance")
    plt.plot(cumulative_variance_limited, "o-", label="Cumulative Explained Variance")

    plt.axhline(y=0.75, color="r", linestyle="--", label="0.75 Variance Threshold")
    plt.axhline(y=0.95, color="b", linestyle="--", label="0.95 Variance Threshold")

    if num_components_075 <= max_components:
        plt.axvline(x=num_components_075 - 1, color="g", linestyle="--")
        plt.text(
            num_components_075,  # Slightly offset from the crossing point
            0.75 - 0.1,  # Slightly higher above the crossing point
            f" 0.75 cut-off\n {num_components_075} components",
            color="g",
            fontsize=10,
            ha="left",  # Horizontal alignment to the left of the point
            va="bottom",  # Vertical alignment below the point
        )

    if num_components_095 <= max_components:
        plt.axvline(x=num_components_095 - 1, color="m", linestyle="--")
        plt.text(
            num_components_095,  # Slightly offset from the crossing point
            0.95 - 0.1,  # Slightly higher above the crossing point
            f" 0.95 cut-off\n {num_components_095} components",
            color="m",
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


@plot_wrapper(
    figsize=(15, 15),
    xlabel="Features",
    ylabel="Features",
    dynamic_params_func=lambda data: {
        "filename": sanitize_filename(f"{len(data.columns)}_correlation_matrix.svg")
    },
    get_image_directory=get_current_image_directory,
)
def plot_correlation_matrix(data, **kwargs):
    # Ensure data has only numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    correlation_matrix = numeric_data.corr()

    # Generate a heatmap using seaborn
    sns.heatmap(
        correlation_matrix,
        annot=False,  # Turn off annotations for large matrices
        fmt=".2f",
        cmap="coolwarm",
        xticklabels=numeric_data.columns,
        yticklabels=numeric_data.columns,
        cbar=False,  # Consider enabling the color bar for scale reference
    )
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()


def main():
    global IMAGE_DIRECTORY

    data = pd.read_csv("./data/processed.csv")
    data.drop(columns=["SMILES"], inplace=True)

    IMAGE_DIRECTORY = f"./dataset_images/variances/"
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
    continuous_columns = [
        col
        for col in continuous_data.columns
        if (continuous_data[col].dtype in ["float64", "int64"])
        and (continuous_data[col].nunique() > unique_threshold)
    ]
    continuous_data = continuous_data[continuous_columns]

    plot_PCA_ratio(continuous_data)
    # plot_PCA_variance_capture(continuous_data, max_components=30)
    plot_feature_variances(continuous_data, n_components=6)
    plot_feature_variances(continuous_data, n_components=25)

    IMAGE_DIRECTORY = f"./dataset_images/"
    plot_correlation_matrix(data)

    pca_data = pd.read_csv("./data/processed_with_pca.csv")
    plot_correlation_matrix(pca_data)
    plot_PCA_variance_capture(pca_data, max_components=30)


if __name__ == "__main__":
    main()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    r2_score,
)
from functools import wraps

# Global variable for directory
IMAGE_DIRECTORY = "./images"


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


def confusion_matrices(data, techniques):
    data["True Cluster"] = determine_cluster(data["Actual"]).astype(int)

    for technique in techniques:
        plot_confusion_matrix(data, technique)


def determine_cluster(values, num_clusters=5):
    # Determine the quantile-based bins for the clusters
    quantiles = np.linspace(0, 1, num_clusters + 1)
    bin_edges = np.quantile(values, quantiles)

    # Assign each value to a cluster based on the bin edges
    clusters = np.digitize(
        values, bin_edges, right=False
    )  # This assigns bins from 1 to 5

    # Offset clusters to be within the range of 0 to num_clusters - 1
    clusters = clusters - 1

    return clusters


@plot_wrapper(
    figsize=None,
    xlabel="Predicted O(η)",
    ylabel="True O(η)",
    dynamic_params_func=lambda data, technique: {
        "filename": f"{technique}_confusion_matrix.svg"
    },
)
def plot_confusion_matrix(data, technique, **kwargs):
    data[f"{technique} Predicted Cluster"] = determine_cluster(
        data[f"{technique}_Predicted_Avg"]
    ).astype(int)

    true_clusters = data["True Cluster"]
    predicted_clusters = data[f"{technique} Predicted Cluster"]

    # Compute confusion matrix and related metrics
    cm = confusion_matrix(true_clusters, predicted_clusters, labels=[0, 1, 2, 3, 4])
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    precision = precision_score(
        true_clusters,
        predicted_clusters,
        average=None,
        labels=[0, 1, 2, 3, 4],
        zero_division=0,
    )
    recall = recall_score(
        true_clusters,
        predicted_clusters,
        average=None,
        labels=[0, 1, 2, 3, 4],
        zero_division=0,
    )
    overall_accuracy = np.trace(cm) / cm.sum()

    fig, ax = plt.subplots(figsize=(9, 7))

    mask_diagonal = np.eye(cm.shape[0], dtype=bool)
    mask_zero_elements = cm == 0
    combined_mask = mask_diagonal | mask_zero_elements
    mask_non_diagonal = ~mask_diagonal

    sns.heatmap(
        cm_normalized,
        mask=mask_non_diagonal,
        annot=False,
        fmt=".2%",
        cmap=sns.light_palette("grey", as_cmap=True),
        cbar=False,
        ax=ax,
    )
    sns.heatmap(
        cm_normalized,
        mask=combined_mask,
        annot=False,
        fmt=".2%",
        cmap=sns.light_palette("red", as_cmap=True),
        cbar=False,
        ax=ax,
    )

    # Add precision and recall to the plot
    for i in range(len(precision)):
        off_precision = 1 - precision[i]
        ax.text(
            i + 0.5,
            len(recall) + 0.5,
            f"{precision[i]:.2%}",
            ha="center",
            va="center",
            color="blue",
        )
        ax.text(
            i + 0.5,
            len(recall) + 0.7,
            f"{off_precision:.2%}",
            ha="center",
            va="center",
            color="red",
        )
        off_recall = 1 - recall[i]
        ax.text(
            len(precision) + 0.5,
            i + 0.5,
            f"{recall[i]:.2%}",
            ha="center",
            va="center",
            color="blue",
        )
        ax.text(
            len(precision) + 0.5,
            i + 0.7,
            f"{off_recall:.2%}",
            ha="center",
            va="center",
            color="red",
        )

    # Add overall accuracy to the bottom right
    off_accuracy = 1 - overall_accuracy
    ax.text(
        len(precision) + 0.5,
        len(recall) + 0.5,
        f"{overall_accuracy:.2%}",
        ha="center",
        va="center",
        color="blue",
    )
    ax.text(
        len(precision) + 0.5,
        len(recall) + 0.7,
        f"{off_accuracy:.2%}",
        ha="center",
        va="center",
        color="red",
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i < cm.shape[0] and j < cm.shape[1]:  # Inside the confusion matrix
                percentage = cm[i, j] / cm.sum()
                annotation = f"{int(cm[i, j])}\n({percentage:.2%})"

            ax.text(
                j + 0.5, i + 0.5, annotation, ha="center", va="center", color="black"
            )

    # Just use the range for the number of classes in the confusion matrix
    ax.set_xticks(np.arange(cm.shape[1]) - 0.5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0]) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", size=0)

    plt.tight_layout()
    plt.subplots_adjust(left=0.15)


@plot_wrapper(ylabel="Mean Absolute Error", filename="MAE_Comparison_Techniques.svg")
def plot_mae(data, techniques):
    """Compare the mean absolute error of each technique with a bar chart."""

    mae_values = [data[f"{technique}_Error"].mean() for technique in techniques]
    plt.bar(techniques, mae_values)


@plot_wrapper(
    xlabel="Actual",
    ylabel="Predicted",
    scale="log",
    filename="Avg_Actual_vs_Predicted.svg",
)
def plot_scatter(data, techniques):
    """Generate a scatter plot for actual vs. average predicted values for each technique with shaded confidence intervals representing the standard deviation across folds."""

    # Plot the identity line
    plt.plot(
        [min(data["Actual"]), max(data["Actual"])],
        [min(data["Actual"]), max(data["Actual"])],
        "k--",
        lw=1,
    )

    for technique in techniques:
        # Sort data for correct CI visualization
        data_sorted = data.sort_values(by="Actual")
        avg_pred_column = f"{technique}_Predicted_Avg"

        # Then plot the scatter points on top
        r2 = r2_score(data_sorted["Actual"], data_sorted[avg_pred_column])
        plt.scatter(
            data_sorted["Actual"],
            data_sorted[avg_pred_column],
            label=f"{technique} (R² = {r2:.2f})",
        )

    plt.legend(loc="best")


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


@plot_wrapper(
    xlabel="Average Feature Importance",
    filename="Feature_Importance.svg",
)
def plot_feature_importances(feature_data):
    # Extract technique names
    feature_data["Technique"] = feature_data[feature_data.columns[0]].str.extract(r"(.*)_")[0]

    # Group by technique and calculate average and standard error
    grouped = feature_data.groupby("Technique").mean(numeric_only=True)
    grouped_sem = feature_data.groupby("Technique").sem(numeric_only=True)

    techniques = [tech for tech in grouped.index if "Base" not in tech]
    
    if len(techniques) == 1:
        technique = techniques[0]
        top_features = grouped.loc[technique].nsmallest(20)
        top_sems = grouped_sem.loc[technique][top_features.index]
        top_features.plot(kind="barh", xerr=top_sems, color="skyblue")
        plt.gca().invert_yaxis()
        plt.title(f"{technique}")
        plt.tight_layout()
    else:
        nrows = int(np.ceil(np.sqrt(len(techniques))))
        ncols = int(np.ceil(len(techniques) / nrows))

        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows)
        )
        fig.subplots_adjust(hspace=0.4, wspace=0.4)

        for i, technique in enumerate(techniques):
            ax = axes.flatten()[i]
            top_features = grouped.loc[technique].nsmallest(20)
            top_sems = grouped_sem.loc[technique][top_features.index]
            top_features.plot(kind="barh", xerr=top_sems, color="skyblue", ax=ax)
            ax.invert_yaxis()
            ax.set_title(f"{technique}")
            ax.set_xlabel("Average Feature Importance")

        for j in range(i + 1, nrows * ncols):
            fig.delaxes(axes.flatten()[j])

        plt.tight_layout()


def main():
    global IMAGE_DIRECTORY
    for file in os.listdir("./"):
        if file.endswith("_results.xlsx"):
            file_path = os.path.join("./", file)
            technique_prefix = file_path.split("/")[-1].split("_")[0]
            IMAGE_DIRECTORY = f"./images/{technique_prefix}"

            pred_data = pd.read_excel(file_path, sheet_name="Predictions")

            prediction_columns = [
                col for col in pred_data.columns if "_Predicted" in col
            ]
            techniques = list(
                set(col.split("_Predicted_")[0] for col in prediction_columns)
            )
            fold_numbers = list(
                set(col.split("_Predicted_")[-1] for col in prediction_columns)
            )

            pred_data = average_folds_predictions(pred_data, techniques, fold_numbers)
            pred_data = calculate_errors(pred_data, techniques, fold_numbers)

            plot_scatter(pred_data, techniques)
            plot_mae(pred_data, techniques)
            confusion_matrices(pred_data, techniques)

            feature_data = pd.read_excel(file_path, sheet_name="Features")
            plot_feature_importances(feature_data)


if __name__ == "__main__":
    main()

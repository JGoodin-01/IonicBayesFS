# Standard library imports
import os
import sys

# Third-party library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, r2_score

# Local application imports
from plot_utils import (
    plot_wrapper,
    determine_cluster,
    calculate_errors,
    average_folds_predictions,
)

IMAGE_DIRECTORY = "./model_images"


def get_current_image_directory():
    return IMAGE_DIRECTORY


def confusion_matrices(data, techniques):
    data["True Cluster"] = determine_cluster(data["Actual"]).astype(int)

    for technique in techniques:
        plot_confusion_matrix(data, technique)


@plot_wrapper(
    figsize=None,
    xlabel="Predicted O(η)",
    ylabel="True O(η)",
    dynamic_params_func=lambda data, technique: {
        "filename": f"{technique}_confusion_matrix.svg"
    },
    get_image_directory=get_current_image_directory,
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


@plot_wrapper(
    ylabel="Mean Absolute Error",
    filename="MAE_Comparison_Techniques.svg",
    get_image_directory=get_current_image_directory,
)
def plot_mae(data, techniques):
    """Compare the mean absolute error of each technique with a bar chart."""

    mae_values = [data[f"{technique}_Error"].mean() for technique in techniques]
    plt.bar(techniques, mae_values)


@plot_wrapper(
    figsize=(3.375, 3.375),
    xlabel="Actual",
    ylabel="Predicted",
    scale="log",
    filename="Avg_Actual_vs_Predicted.svg",
    get_image_directory=get_current_image_directory,
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
    plt.tight_layout()


@plot_wrapper(
    xlabel="Average Feature Importance",
    filename="Feature_Importance.svg",
    get_image_directory=get_current_image_directory,
)
def plot_feature_importances(feature_data):
    # Extract technique names
    feature_data["Technique"] = feature_data[feature_data.columns[0]].str.extract(
        r"(.*)_"
    )[0]

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


@plot_wrapper(
    figsize=(3.375, 3.375),
    xlabel="Predicted",
    ylabel="Residuals",
    filename="Residuals_Plot.svg",
    get_image_directory=get_current_image_directory,
)
def plot_residuals(data, technique):
    """
    Plot the residuals for a given prediction technique.

    Parameters:
    data (DataFrame): The pandas DataFrame containing the prediction data.
    technique (str): The name of the prediction technique to plot.
    """
    predicted_column = f"{technique}_Predicted_Avg"
    residuals = data["Actual"] - data[predicted_column]

    # Plot the identity line for reference (y=0)
    plt.axhline(y=0, color="r", linestyle="--", linewidth=1)

    plt.scatter(data[predicted_column], residuals, color="blue", alpha=0.5)
    plt.tight_layout()


@plot_wrapper(
    ylabel="R² Score",
    xlabel="Fold",
    filename="R2_Scores_By_Phase.svg",
    get_image_directory=get_current_image_directory,
)
def plot_r2_scores(df):
    """
    Plot the R² scores for different feature selection methods across folds.
    Assumes the first column of df contains the compound method-phase-fold information.
    """
    # Extract columns
    df = df.rename(columns={df.columns[0]: "Method_Phase_Fold"})
    df[["Method", "Phase", "Fold"]] = df["Method_Phase_Fold"].str.rsplit(
        "_", n=2, expand=True
    )
    df["Fold"] = df["Fold"].astype(int)
    
    # Create a single plot for both training and testing scores
    fig, ax = plt.subplots(figsize=(3.375, 3.375))

    # Calculate range and add a buffer
    r2_range = df['R2'].max() - df['R2'].min()
    buffer = r2_range * 0.1  # 10% buffer on each side
    y_min = max(0, df['R2'].min() - buffer)  # Ensure y_min is not below 0
    y_max = min(1, df['R2'].max() + buffer)  # Ensure y_max is not above 1
    ax.set_ylim([y_min, y_max])


    # Training data plot
    training_data = df[df['Phase'] == 'Training']
    sns.lineplot(x='Fold', y='R2', hue='Method', data=training_data, marker='o', ax=ax, linestyle='-')

    # Testing data plot
    testing_data = df[df['Phase'] == 'Testing']
    sns.lineplot(x='Fold', y='R2', hue='Method', data=testing_data, marker='s', ax=ax, linestyle='--')

    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    if len(unique_labels) > 1:
        ax.legend(unique_labels.values(), unique_labels.keys(), title='Method')
    else:
        ax.legend().remove()

    plt.tight_layout()


def main():
    global IMAGE_DIRECTORY
    for root, dirs, files in os.walk("."):
        for dir in dirs:
            if "results" in dir:
                # Construct the path to the directory
                results_directory = os.path.join(root, dir)
                print(f"Processing directory: {results_directory}")

                # List Excel files in the identified directory
                for file in os.listdir(results_directory):
                    if file.endswith("_results.xlsx"):
                        file_path = os.path.join(results_directory, file)
                        technique_prefix = file.split("_")[0]
                        IMAGE_DIRECTORY = os.path.join(
                            results_directory, "model_images", technique_prefix
                        )
                        if not os.path.exists(IMAGE_DIRECTORY):
                            os.makedirs(IMAGE_DIRECTORY)

                        try:
                            pred_data = pd.read_excel(
                                file_path, sheet_name="Predictions"
                            )
                            prediction_columns = [
                                col for col in pred_data.columns if "_Predicted" in col
                            ]
                            techniques = list(
                                set(
                                    col.split("_Predicted_")[0]
                                    for col in prediction_columns
                                )
                            )
                            fold_numbers = list(
                                set(
                                    col.split("_Predicted_")[-1]
                                    for col in prediction_columns
                                )
                            )

                            pred_data = average_folds_predictions(
                                pred_data, techniques, fold_numbers
                            )
                            pred_data = calculate_errors(
                                pred_data, techniques, fold_numbers
                            )

                            plot_scatter(pred_data, techniques)
                            plot_mae(pred_data, techniques)
                            confusion_matrices(pred_data, techniques)

                            for technique in techniques:
                                plot_residuals(pred_data, technique)

                            metric_data = pd.read_excel(file_path, sheet_name="Metrics")
                            plot_r2_scores(metric_data)

                            # Extract the feature importance data if it exists
                            if "post_fe" in results_directory:
                                feature_data = pd.read_excel(
                                    file_path, sheet_name="Features"
                                )
                                plot_feature_importances(feature_data)
                        except Exception as e:
                            print(f"Failed to process {file_path}: {str(e)}")


if __name__ == "__main__":
    main()

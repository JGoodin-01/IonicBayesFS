import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2, venn3
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    r2_score,
)
from functools import wraps


def plot_wrapper(figsize=(8, 6), xlabel="", ylabel="", scale=None, directory="images", filename="image.svg", dynamic_params_func=None):
    def decorator(plot_func):
        @wraps(plot_func)
        def wrapper(*args, **kwargs):
            # Dynamic parameter processing
            if dynamic_params_func is not None:
                dynamic_params = dynamic_params_func(*args, **kwargs)
                dynamic_filename = dynamic_params.get("filename", filename)  # Use a different variable
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
            
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(os.path.join(directory, dynamic_filename), format='svg')  # Use the dynamically determined filename
            plt.close()

        return wrapper
    return decorator


def calculate_errors(data, techniques):
    """Calculate absolute errors for each feature selection technique."""
    for technique in techniques:
        predicted_column = f"{technique}_Predicted"
        data[f"{technique}_Error"] = np.abs(data["Actual"] - data[predicted_column])
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
    clusters = np.digitize(values, bin_edges, right=False)  # This assigns bins from 1 to 5
    
    # Offset clusters to be within the range of 0 to num_clusters - 1
    clusters = clusters - 1
    
    return clusters


@plot_wrapper(figsize=None, xlabel="Predicted O(η)", ylabel="True O(η)", dynamic_params_func=lambda data, technique: {"filename": f"{technique}_confusion_matrix.svg"})
def plot_confusion_matrix(data, technique, **kwargs):
    data[f"{technique} Predicted Cluster"] = determine_cluster(
                data[f"{technique}_Predicted"]
            ).astype(int)
        
    true_clusters = data["True Cluster"]
    predicted_clusters = data[f"{technique} Predicted Cluster"]
    
    # Compute confusion matrix and related metrics
    cm = confusion_matrix(true_clusters, predicted_clusters, labels=[0, 1, 2, 3, 4])
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    precision = precision_score(true_clusters, predicted_clusters, average=None, labels=[0, 1, 2, 3, 4], zero_division=0)
    recall = recall_score(true_clusters, predicted_clusters, average=None, labels=[0, 1, 2, 3, 4], zero_division=0)
    overall_accuracy = np.trace(cm) / cm.sum()

    fig, ax = plt.subplots(figsize=(9, 7))

    mask_diagonal = np.eye(cm.shape[0], dtype=bool)
    mask_zero_elements = cm == 0
    combined_mask = mask_diagonal | mask_zero_elements
    mask_non_diagonal = ~mask_diagonal

    sns.heatmap(cm_normalized, mask=mask_non_diagonal, annot=False, fmt=".2%", cmap=sns.light_palette("grey", as_cmap=True), cbar=False, ax=ax)
    sns.heatmap(cm_normalized, mask=combined_mask, annot=False, fmt=".2%", cmap=sns.light_palette("red", as_cmap=True), cbar=False, ax=ax)

    # Add precision and recall to the plot
    for i in range(len(precision)):
        off_precision = 1 - precision[i]
        ax.text(i + 0.5, len(recall) + 0.5, f"{precision[i]:.2%}", ha="center", va="center", color="blue")
        ax.text(i + 0.5, len(recall) + 0.7, f"{off_precision:.2%}", ha="center", va="center", color="red")
        off_recall = 1 - recall[i]
        ax.text(len(precision) + 0.5, i + 0.5, f"{recall[i]:.2%}", ha="center", va="center", color="blue")
        ax.text(len(precision) + 0.5, i + 0.7, f"{off_recall:.2%}", ha="center", va="center", color="red")

    # Add overall accuracy to the bottom right
    off_accuracy = 1 - overall_accuracy
    ax.text(len(precision) + 0.5, len(recall) + 0.5, f"{overall_accuracy:.2%}", ha="center", va="center", color="blue")
    ax.text(len(precision) + 0.5, len(recall) + 0.7, f"{off_accuracy:.2%}", ha="center", va="center", color="red")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i < cm.shape[0] and j < cm.shape[1]:  # Inside the confusion matrix
                percentage = cm[i, j] / cm.sum()
                annotation = f"{int(cm[i, j])}\n({percentage:.2%})"
            
            ax.text(
                j + 0.5,
                i + 0.5,
                annotation,
                ha="center",
                va="center",
                color="black"
            )
    
    # Just use the range for the number of classes in the confusion matrix
    ax.set_xticks(np.arange(cm.shape[1]) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0]) - .5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", size=0)

    plt.tight_layout()
    plt.subplots_adjust(left=0.15)


@plot_wrapper(filename="venn.svg")
def plot_feature_correspondance(data, techniques):
    data = data == True

    assert 1 < len(techniques) <= 3, "This function supports 2 or 3 techniques for Venn diagrams."
    
    # Initialize selections and subsets for Venn diagram
    selections = {tech: data[tech].sum() for tech in techniques}
    subsets = [selections[tech] for tech in techniques]
    intersections = {}

    # Calculate intersections
    for i, tech1 in enumerate(techniques):
        for j, tech2 in enumerate(techniques[i+1:], i+1):
            intersection_key = f'{tech1}_{tech2}'
            intersections[intersection_key] = data[data[tech1] & data[tech2]].shape[0]
            
            if len(techniques) == 3 and j < len(techniques) - 1:
                for k, tech3 in enumerate(techniques[j+1:], j+1):
                    all_intersection_key = f'{tech1}_{tech2}_{tech3}'
                    intersections[all_intersection_key] = data[data[tech1] & data[tech2] & data[tech3]].shape[0]

    # For 2 techniques, adjust subsets list directly
    if len(techniques) == 2:
        subsets.append(intersections[next(iter(intersections))])  # Only intersection for 2 techniques
        venn_diagram = venn2(subsets=subsets, set_labels=techniques)
    # For 3 techniques, create subsets list based on the order required by venn3
    elif len(techniques) == 3:
        tech1, tech2, tech3 = techniques
        subsets = [
            selections[tech1], selections[tech2], intersections[f'{tech1}_{tech2}'],
            selections[tech3], intersections[f'{tech1}_{tech3}'], intersections[f'{tech2}_{tech3}'],
            intersections[f'{tech1}_{tech2}_{tech3}']
        ]
        venn_diagram = venn3(subsets=subsets, set_labels=techniques)

    # Adjust font sizes for readability
    for text in venn_diagram.set_labels:
        if text: text.set_fontsize(14)
    for text in venn_diagram.subset_labels:
        if text: text.set_fontsize(12)


@plot_wrapper(ylabel="Mean Absolute Error", filename="MAE_Comparison_Techniques.svg")
def plot_mae(data, techniques):
    """Compare the mean absolute error of each technique with a bar chart."""
    
    mae_values = [data[f"{technique}_Error"].mean() for technique in techniques]
    plt.bar(techniques, mae_values)


@plot_wrapper(xlabel="Actual", ylabel="Predicted", scale="log", filename="Combined_Actual_vs_Predicted_R2.svg")
def plot_scatter(data, techniques):
    """Generate a combined scatter plot for actual vs. predicted values for each technique, including R^2 annotations."""
    
    for technique in techniques:
        predicted_column = f"{technique}_Predicted"
        r2 = r2_score(data["Actual"], data[predicted_column])  # Compute R^2
        plt.scatter(data["Actual"], data[predicted_column], alpha=0.5, label=f'{technique} (R² = {r2:.2f})')
    
    # Plotting the identity line in a more compact form
    plt.plot([min(data["Actual"]), max(data["Actual"])], [min(data["Actual"]), max(data["Actual"])], "k--", lw=2)
    plt.legend(loc="best")  # Show legend to identify each technique


def main():
    file_path = "./results.xlsx"
    pred_data = pd.read_excel(file_path, sheet_name="Results")
    feature_data = pd.read_excel(file_path, sheet_name="Selected_Features")

    # Identify the feature selection techniques based on the column names
    prediction_columns = [col for col in pred_data.columns if "_Predicted" in col]
    techniques = [col.replace("_Predicted", "") for col in prediction_columns]

    # Calculate errors for each technique
    pred_data = calculate_errors(pred_data, techniques)

    # Generate plots
    plot_scatter(pred_data, techniques)
    plot_mae(pred_data, techniques)
    plot_feature_correspondance(feature_data, techniques)

    # Plot the confusion matricess
    confusion_matrices(pred_data, techniques)


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import r2_score, confusion_matrix, precision_score, recall_score


def plot_scatter(data, entries):
    """Generate a combined scatter plot for actual vs. predicted values for each technique, including R^2 annotations."""
    plt.figure(figsize=(8, 8))
    
    for i, technique in enumerate(entries):
        predicted_column = f"{technique}_Predicted"
        r2 = r2_score(data["Actual"], data[predicted_column])  # Compute R^2

        plt.scatter(data["Actual"], data[predicted_column], alpha=0.5, label=f'{technique} (R^2 = {r2:.2f})')
    
    # Plotting the identity line for reference
    plt.plot(
        [data["Actual"].min(), data["Actual"].max()],
        [data["Actual"].min(), data["Actual"].max()],
        "k--",
        lw=2,
    )
    
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.legend(loc="best")  # Show legend to identify each technique
    
    # Save the combined plot
    plt.savefig("images/Combined_Actual_vs_Predicted_R2.svg", format="svg")
    plt.show()
    plt.close()


def plot_error_distribution(data, entries):
    """Plot the error distribution for each technique."""
    plt.figure(figsize=(14, 6))

    for technique in entries:
        error_column = f"{technique}_Error"
        plt.plot(data[error_column], label=f"{technique} Error", alpha=0.7)

    plt.title("Error Distribution: Techniques Comparison")
    plt.xlabel("Sample Index")
    plt.ylabel("Absolute Error")
    plt.legend()

    # Save the plot
    plt.savefig("images/Error_Distribution_Techniques_Comparison.svg", format="svg")
    plt.close()


def compare_performance(data, entries):
    """Compare the mean absolute error of each technique with a bar chart."""
    mae_values = [data[f"{technique}_Error"].mean() for technique in entries]

    plt.bar(entries, mae_values)
    plt.title("Mean Absolute Error: Techniques Comparison")
    plt.ylabel("Mean Absolute Error")

    # Save the plot
    plt.savefig("images/MAE_Comparison_Techniques.svg", format="svg")
    plt.close()


def calculate_errors(data, entries):
    """Calculate absolute errors for each feature selection technique."""
    for technique in entries:
        predicted_column = f"{technique}_Predicted"
        data[f"{technique}_Error"] = np.abs(data["Actual"] - data[predicted_column])
    return data


def ensure_directory(directory_path):
    """Ensure that the specified directory exists; create it if it doesn't."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def determine_cluster(values, num_clusters=5):
    # Determine the quantile-based bins for the clusters
    quantiles = np.linspace(0, 1, num_clusters + 1)
    bin_edges = np.quantile(values, quantiles)
    
    # Assign each value to a cluster based on the bin edges
    clusters = np.digitize(values, bin_edges, right=False)  # This assigns bins from 1 to 5
    
    # Offset clusters to be within the range of 0 to num_clusters - 1
    clusters = clusters - 1
    
    return clusters


def plot_confusion_matrix(data, entries):
    data["True Cluster"] = determine_cluster(data["Actual"])
    data["True Cluster"] = data["True Cluster"].astype(int)
    
    for technique in entries:
        data[f"{technique} Predicted Cluster"] = determine_cluster(
                data[f"{technique}_Predicted"]
            )
        data[f"{technique} Predicted Cluster"] = data[
            f"{technique} Predicted Cluster"
        ].astype(int)
        
        true_clusters = data["True Cluster"]
        predicted_clusters = data[f"{technique} Predicted Cluster"]
        
        # Calculate the confusion matrix for the entire dataset
        cm = confusion_matrix(true_clusters, predicted_clusters, labels=[0, 1, 2, 3, 4])

        # Normalize the confusion matrix
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        total = np.sum(cm)  # Total number of instances

        # Calculate precision and recall for each class
        precision = precision_score(true_clusters, predicted_clusters, average=None, labels=[0, 1, 2, 3, 4], zero_division=0)
        recall = recall_score(true_clusters, predicted_clusters, average=None, labels=[0, 1, 2, 3, 4], zero_division=0)

        # Calculate overall accuracy
        overall_accuracy = np.trace(cm) / total

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
                    percentage = cm[i, j] / total
                    annotation = f"{int(cm[i, j])}\n({percentage:.2%})"
                
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    annotation,
                    ha="center",
                    va="center",
                    color="black"
                )

        # Set the labels for the axes
        ax.set_xlabel("Predicted O(η)")
        ax.set_ylabel("True O(η)")
        
        # Just use the range for the number of classes in the confusion matrix
        ax.set_xticks(np.arange(cm.shape[1]) - .5, minor=True)
        ax.set_yticks(np.arange(cm.shape[0]) - .5, minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
        ax.tick_params(which="minor", size=0)

        plt.tight_layout()
        plt.subplots_adjust(left=0.15)
        plt.savefig(f"images/{technique}_accuracy_matrix.svg", format="svg")


def main():
    file_path = "./results.xlsx"
    data = pd.read_excel(file_path)

    # Identify the feature selection techniques based on the column names
    prediction_columns = [col for col in data.columns if "_Predicted" in col]
    techniques = [col.replace("_Predicted", "") for col in prediction_columns]

    # Calculate errors for each technique
    data = calculate_errors(data, techniques)

    # Ensure the images directory exists
    images_dir = "images"
    ensure_directory(images_dir)

    # Generate plots
    plot_scatter(data, techniques)
    plot_error_distribution(data, techniques)
    compare_performance(data, techniques)

    # Plot the confusion matrix
    plot_confusion_matrix(data, techniques)


if __name__ == "__main__":
    main()

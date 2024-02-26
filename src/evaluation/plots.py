import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score

def plot_scatter(data, entries):
    """Generate scatter plots for actual vs. predicted values for each technique, including R^2."""
    num_techniques = len(entries)
    plt.figure(figsize=(14, 6 * num_techniques))
    
    for i, technique in enumerate(entries, 1):
        predicted_column = f'{technique}_Predicted'
        r2 = r2_score(data['Actual'], data[predicted_column])  # Compute R^2
        
        plt.subplot(num_techniques, 1, i)
        plt.scatter(data['Actual'], data[predicted_column], alpha=0.5)
        plt.plot([data['Actual'].min(), data['Actual'].max()], [data['Actual'].min(), data['Actual'].max()], 'k--', lw=2)
        plt.title(f'{technique}: Actual vs. Predicted (R^2 = {r2:.2f})')  # Include R^2 in title
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        
        # Save the plot
        plt.savefig(f'images/{technique}_Actual_vs_Predicted_R2.svg', format='svg')
    plt.close()

def plot_error_distribution(data, entries):
    """Plot the error distribution for each technique."""
    plt.figure(figsize=(14, 6))
    
    for technique in entries:
        error_column = f'{technique}_Error'
        plt.plot(data[error_column], label=f'{technique} Error', alpha=0.7)
    
    plt.title('Error Distribution: Techniques Comparison')
    plt.xlabel('Sample Index')
    plt.ylabel('Absolute Error')
    plt.legend()
    
    # Save the plot
    plt.savefig('images/Error_Distribution_Techniques_Comparison.svg', format='svg')
    plt.close()

def compare_performance(data, entries):
    """Compare the mean absolute error of each technique with a bar chart."""
    mae_values = [data[f'{technique}_Error'].mean() for technique in entries]
    
    plt.bar(entries, mae_values, color=np.random.rand(len(entries),3))
    plt.title('Mean Absolute Error: Techniques Comparison')
    plt.ylabel('Mean Absolute Error')
    
    # Save the plot
    plt.savefig('images/MAE_Comparison_Techniques.svg', format='svg')
    plt.close()

def calculate_errors(data, entries):
    """Calculate absolute errors for each feature selection technique."""
    for technique in entries:
        predicted_column = f'{technique}_Predicted'
        data[f'{technique}_Error'] = np.abs(data['Actual'] - data[predicted_column])
    return data

def ensure_directory(directory_path):
    """Ensure that the specified directory exists; create it if it doesn't."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def main():
    file_path = './results.xlsx'
    data = pd.read_excel(file_path)
    
    # Identify the feature selection techniques based on the column names
    prediction_columns = [col for col in data.columns if '_Predicted' in col]
    techniques = [col.replace('_Predicted', '') for col in prediction_columns]
    
    # Calculate errors for each technique
    data = calculate_errors(data, techniques)
    
    # Ensure the images directory exists
    images_dir = 'images'
    ensure_directory(images_dir)
    
    # Generate plots
    plot_scatter(data, techniques)
    plot_error_distribution(data, techniques)
    compare_performance(data, techniques)

if __name__ == "__main__":
    main()

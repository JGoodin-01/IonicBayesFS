import subprocess
import os

def run_preprocessing():
    """Run the data preprocessing script."""
    subprocess.run(['python', './scripts/run_preprocessing.py'], check=True)

def train_and_evaluate(model_name):
    """Run the training and evaluation script for a given model."""
    subprocess.run(['python', './scripts/train_and_evaluate.py', '--model', model_name], check=True)

if __name__ == '__main__':
    run_preprocessing()
    
    # models = ['knn']
    # for model in models:
    #     train_and_evaluate(model)

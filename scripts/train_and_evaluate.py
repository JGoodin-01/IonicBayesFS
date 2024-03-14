import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor as skKNeighborsRegressor
from src.preprocessing.BFS import BFS
import subprocess
import sys


# Function to check CUDA availability
def cuda_available():
    try:
        import cuml

        return True
    except ImportError:
        return False


# Conditional imports based on CUDA availability
if cuda_available():
    from cuml.neighbors import KNeighborsRegressor as cuKNeighborsRegressor
    
    KNeighborsRegressor = cuKNeighborsRegressor
    
    print("Using RAPIDS cuML for GPU acceleration.")
else:
    KNeighborsRegressor = skKNeighborsRegressor
    print("CUDA not available. Falling back to scikit-learn.")


def apply_feature_selection(fs_strategy, X, y):
    if fs_strategy["name"] == "Base":
        return X, None

    if fs_strategy["name"] == "SelectKBest":
        selector = SelectKBest(mutual_info_regression, k=fs_strategy.get("k", 10))
    elif fs_strategy["name"] == "RFE":
        selector = RFE(
            RandomForestRegressor(n_estimators=5),
            n_features_to_select=fs_strategy.get("k", 10),
        )
    elif fs_strategy["name"] == "BFS":
        selector = BFS()
    else:
        raise ValueError("Invalid feature selection strategy")

    selector.fit(X, y)
    if fs_strategy["name"] == "BFS":
        selector.traceplot()
    return selector.transform(X), selector


def run_experiment(X, y, model, model_params, feature_selection_strategies):
    def preprocess_features(X):
        """Encode categorical variables and scale numerical features."""
        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col])
        imputer = SimpleImputer(strategy="mean")
        scaler = MinMaxScaler()
        return scaler.fit_transform(imputer.fit_transform(X))

    def evaluate_model(X_train_fs, X_test_fs, y_train, y_test):
        """Fit the model and evaluate it on the test set."""
        model_instance = model(**model_params)
        model_instance.fit(X_train_fs, y_train)
        predictions = model_instance.predict(X_test_fs)
        r2 = r2_score(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        return predictions, r2, mse

    # Main experiment logic starts here
    smiles = X["SMILES"].values
    X = X.drop("SMILES", axis=1)
    X_scaled = preprocess_features(X)
    X_train, X_test, y_train, y_test, _, smiles_test = train_test_split(
        X_scaled, y, smiles, test_size=0.2, random_state=10
    )

    results_log = pd.DataFrame({"SMILES": smiles_test, "Actual": y_test})
    features_log = pd.DataFrame(index=X.columns)
    metrics_log = pd.DataFrame()
    with pd.ExcelWriter("results.xlsx") as writer:
        for fs_strategy in feature_selection_strategies:
            X_train_fs, selector = apply_feature_selection(
                fs_strategy, X_train, y_train
            )
            X_test_fs = selector.transform(X_test) if selector else X_test

            predictions, r2, mse = evaluate_model(
                X_train_fs, X_test_fs, y_train, y_test
            )
            print(f"{fs_strategy['name']}: R2={r2}, MSE={mse}")

            results_log[f"{fs_strategy['name']}_Predicted"] = predictions
            features_log[fs_strategy["name"]] = pd.Series(
                (
                    selector.get_support()
                    if selector and hasattr(selector, "get_support")
                    else [True] * X_train_fs.shape[1]
                ),
                index=X.columns,
            )
            metrics_log[fs_strategy["name"]] = {"R2": r2, "MSE": mse}

        # Save logs to Excel
        results_log.to_excel(writer, sheet_name="Predictions", index=False)
        features_log.to_excel(writer, sheet_name="Selected_Features")
        metrics_log.T.to_excel(writer, sheet_name="Metrics")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model evaluation with feature selection."
    )
    parser.add_argument("--model", type=str, choices=["knn", "rf"], help="Model name")
    args = parser.parse_args()

    data = pd.read_csv("./data/processed.csv").dropna(subset=["η / mPa s"])
    X, y = data.drop("η / mPa s", axis=1), data["η / mPa s"].values

    feature_selection_strategies = [
        {"name": "Base"},
        {"name": "SelectKBest", "k": 8},
        {"name": "RFE", "k": 10},
        {"name": "BFS"},
    ]
    if args.model == "knn":
        model = KNeighborsRegressor
        model_params = {"n_neighbors": 3}
    elif args.model == "rf":
        model = RandomForestRegressor
        model_params = {
            "n_estimators": 100,
            "max_depth": None,  # None means nodes are expanded until all leaves are pure or contain < min_samples_split samples
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        }

    run_experiment(X, y, model, model_params, feature_selection_strategies)

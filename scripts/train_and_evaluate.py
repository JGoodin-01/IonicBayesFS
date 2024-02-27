import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor as RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor as skKNeighborsRegressor

from src.preprocessing.BFS import BFS

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


def apply_feature_selection(fs_strategy, X_train, y_train):
    if fs_strategy["name"] == "SelectKBest":
        selector = SelectKBest(mutual_info_regression, k=fs_strategy["k"])
    elif fs_strategy["name"] == "RFE":
        estimator = RandomForestRegressor(n_estimators=5)
        selector = RFE(estimator, n_features_to_select=fs_strategy["k"])
    elif fs_strategy["name"] == "BFS":
        selector = BFS()

    selector.fit(X_train, y_train)
    if fs_strategy["name"] == "BFS":
        selector.traceplot()
    
    X_train_selected = selector.transform(X_train)
    return selector, X_train_selected


def run_experiment(X, y, model, model_params, feature_selection_strategies):
    smiles = X["SMILES"].values
    X = X.drop("SMILES", axis=1)

    label_encoder = LabelEncoder()
    categorical_columns = X.select_dtypes(include=["object"]).columns
    for col in categorical_columns:
        X[col] = label_encoder.fit_transform(X[col])

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(
        X_scaled, y, smiles, test_size=0.2, random_state=10
    )

    all_results_df = pd.DataFrame({"SMILES": smiles_test, "Actual": y_test})

    # Initialize a DataFrame to hold all the selected features
    feature_log = pd.DataFrame(index=X.columns)

    with pd.ExcelWriter("results.xlsx") as writer:
        for fs_strategy in feature_selection_strategies:
            selector, X_train_selected = apply_feature_selection(
                fs_strategy, X_train, y_train
            )
            X_test_selected = selector.transform(X_test)
            model_instance = model(**model_params)
            model_instance.fit(X_train_selected, y_train)
            predictions = model_instance.predict(X_test_selected)

            # Add predictions to the results DataFrame
            all_results_df[f"{fs_strategy['name']}_Predicted"] = predictions

            # Collect selected features
            if hasattr(selector, "get_support"):
                feature_log[fs_strategy["name"]] = pd.Series(
                    selector.get_support(), index=X.columns
                )

            r2 = r2_score(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            print(
                f"Feature Selection Strategy: {fs_strategy['name']}, R2: {r2}, MSE: {mse}"
            )

        all_results_df.to_excel(writer, sheet_name="Results", index=False)
        feature_log.to_excel(writer, sheet_name="Selected_Features")


if __name__ == "__main__":
    source_data = pd.read_csv("./data/processed.csv", low_memory=False)
    source_data = source_data.dropna(subset=["η / mPa s"])

    y = source_data["η / mPa s"].values
    X = source_data.drop("η / mPa s", axis=1)

    parser = argparse.ArgumentParser(
        description="Train and evaluate a model with feature selection."
    )
    parser.add_argument("--model", type=str, help="Model name")

    args = parser.parse_args()

    feature_selection_strategies = [
        {"name": "SelectKBest", "k": [2, 114]},
        {"name": "RFE", "k": 10},
        {"name": "BFS"}
    ]

    if args.model == "knn":
        model_params = {"n_neighbors": 3}
        run_experiment(X, y, KNeighborsRegressor, model_params, feature_selection_strategies)

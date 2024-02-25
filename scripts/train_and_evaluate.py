import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from src.models.KNNModel import KNNModel


def run_fs(X_train, X_test, y_train, y_test, fs_strategy, model, model_params):
    if fs_strategy["name"] == "SelectKBest":
        selector = SelectKBest(mutual_info_regression, k=fs_strategy["k"])
    elif fs_strategy["name"] == "RFE":
        estimator = RandomForestRegressor(n_estimators=100)
        selector = RFE(estimator, n_features_to_select=fs_strategy["k"])

    selector.fit(X_train, y_train)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    model_instance = model(**model_params)
    model_instance.train(X_train_selected, y_train)
    predictions = model_instance.predict(X_test_selected)

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    print(f"Feature Selection Strategy: {fs_strategy['name']}, R2: {r2}, MSE: {mse}")


def run_experiment(X, y, model_params, feature_selection_strategies):
    smiles = X["SMILES"]
    X = X.drop("SMILES", axis=1)

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    for fs_strategy in feature_selection_strategies:
        run_fs(X_train, X_test, y_train, y_test, fs_strategy, KNNModel, model_params)


if __name__ == "__main__":
    source_data = pd.read_csv("./data/processed.csv", low_memory=False)
    source_data = source_data.dropna(subset=["η / mPa s"])

    label_encoder = LabelEncoder()
    categorical_columns = source_data.select_dtypes(include=["object"]).columns
    for col in categorical_columns:
        source_data[col] = label_encoder.fit_transform(source_data[col])

    y = source_data["η / mPa s"].values
    X = source_data.drop("η / mPa s", axis=1)

    parser = argparse.ArgumentParser(
        description="Train and evaluate a model with feature selection."
    )
    parser.add_argument("--model", type=str, help="Model name")

    args = parser.parse_args()

    feature_selection_strategies = [
        {"name": "SelectKBest", "k": 10},
        {"name": "RFE", "k": 10},
    ]

    if args.model == "knn":
        model_params = {"n_neighbors": 3}
        run_experiment(X, y, model_params, feature_selection_strategies)

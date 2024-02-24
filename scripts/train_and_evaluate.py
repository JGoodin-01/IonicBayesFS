import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from src.feature_selection.SelectKBest import SelectKBest
from src.models.KNNModel import KNNModel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def run_experiment(X, y, model_params):
    smiles = X['SMILES']  # SMILES strings
    X = X.drop("SMILES", axis=1)

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    # Split data and SMILES strings
    X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(
        X_imputed, y, smiles, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    best_r2 = 0
    best_k = 0
    for k in range(1, 112):
        feature_selector = SelectKBest(k=k)
        feature_selector.fit(X_train, y_train)
        X_train_selected = feature_selector.transform(X_train)
        X_test_selected = feature_selector.transform(X_test)

        model = KNNModel(**model_params)
        model.train(X_train_selected, y_train)
        predictions = model.predict(X_test_selected)
        r2 = r2_score(y_test, predictions)

        if r2 > best_r2:
            best_r2 = r2
            best_k = k

    print(best_k)
    feature_selector = SelectKBest(k=best_k)
    feature_selector.fit(X_train, y_train)
    X_train_selected = feature_selector.transform(X_train)
    X_test_selected = feature_selector.transform(X_test)

    selected_feature_indices = feature_selector.get_selected_indices()
    selected_feature_names = X.columns[selected_feature_indices]

    print("Selected Features:")
    for feature_name in selected_feature_names:
        print(feature_name)

    model = KNNModel(**model_params)
    model.train(X_train_selected, y_train)
    predictions = model.predict(X_test_selected)

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    print(f"R2: {r2}")
    print(f"Mean Squared Error: {mse}")

    output = {"SMILES": smiles_test, "Prediction": predictions, "True": y_test}
    output = pd.DataFrame(output)
    output.to_csv("./results.csv", index=False)


def classify_viscosity(eta, O_min=0, O_max=4):
    log_eta = np.floor(np.log(eta))
    O_eta = np.minimum(np.maximum(O_min, log_eta), O_max)
    return O_eta


if __name__ == "__main__":
    source_data = pd.read_csv("./data/processed.csv", low_memory=False)
    source_data = source_data.dropna(subset=["η / mPa s"])

    label_encoder = LabelEncoder()
    categorical_columns = source_data.select_dtypes(include=["object"]).columns
    for col in categorical_columns:
        source_data[col] = label_encoder.fit_transform(source_data[col])

    y = source_data["η / mPa s"]
    X = source_data.drop("η / mPa s", axis=1)

    parser = argparse.ArgumentParser(
        description="Train and evaluate a model with feature selection."
    )
    parser.add_argument("--model", type=str, help="Model name")

    args = parser.parse_args()

    if args.model == "knn":
        model_params = {"n_neighbors": 5}  # Example parameters for kNN

        run_experiment(X, y, model_params)  # Ensure model_params is passed

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

try:
    from cuml.neighbors import KNeighborsRegressor as cuKNeighborsRegressor
    KNeighborsRegressor = cuKNeighborsRegressor
    print("Using RAPIDS cuML for GPU acceleration.")
except ImportError:
    KNeighborsRegressor = skKNeighborsRegressor
    print("CUDA not available. Falling back to scikit-learn.")

def apply_feature_selection(fs_strategy, X, y):
    if fs_strategy["name"] == "SelectKBest":
        selector = SelectKBest(mutual_info_regression, k=fs_strategy.get("k", 10))
    elif fs_strategy["name"] == "RFE":
        selector = RFE(RandomForestRegressor(n_estimators=5), n_features_to_select=fs_strategy.get("k", 10))
    elif fs_strategy["name"] == "BFS":
        selector = BFS()
    else:
        raise ValueError("Invalid feature selection strategy")

    selector.fit(X, y)
    if fs_strategy["name"] == "BFS":
        selector.traceplot()
    return selector.transform(X), selector

def preprocess_data(X):
    # Encode categorical features
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col])
    
    imputer = SimpleImputer(strategy="mean")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(imputer.fit_transform(X))
    
    return X_scaled


def run_experiment(X, y, model, model_params, feature_selection_strategies):
    X = preprocess_data(X.drop("SMILES", axis=1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    results = []
    for fs_strategy in feature_selection_strategies:
        X_train_fs, selector = apply_feature_selection(fs_strategy, X_train, y_train)
        X_test_fs = selector.transform(X_test)

        model_instance = model(**model_params)
        model_instance.fit(X_train_fs, y_train)
        predictions = model_instance.predict(X_test_fs)

        r2, mse = r2_score(y_test, predictions), mean_squared_error(y_test, predictions)
        results.append((fs_strategy["name"], r2, mse))
        print(f"{fs_strategy['name']}: R2={r2}, MSE={mse}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model evaluation with feature selection.")
    parser.add_argument("--model", type=str, choices=['knn'], help="Model name")
    args = parser.parse_args()

    data = pd.read_csv("./data/processed.csv").dropna(subset=["η / mPa s"])
    X, y = data.drop("η / mPa s", axis=1), data["η / mPa s"].values

    feature_selection_strategies = [{"name": "SelectKBest", "k": 8}, {"name": "RFE", "k": 10}, {"name": "BFS"}]
    model_params = {"n_neighbors": 3} if args.model == "knn" else {}
    run_experiment(X, y, KNeighborsRegressor, model_params, feature_selection_strategies)

import argparse
import pandas as pd
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from src.preprocessing.BFS import BFS


# Function to check CUDA availability
def cuda_available():
    try:
        import cuml
        return True
    except ImportError:
        return False


if cuda_available():
    print("Using RAPIDS cuML for GPU acceleration.")
else:
    print("CUDA not available. Falling back to scikit-learn.")


EXPERIMENT_CONFIGS = {
    "rf": {
        "model": RandomForestRegressor,
        "param_grid": {
            "n_estimators": [10, 50, 100],
            "max_depth": [None, 10, 20, 30],
        },
    }
}


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
    # Main experiment logic starts here
    smiles = X["SMILES"].values
    X = X.drop("SMILES", axis=1)
    for col in X.select_dtypes(include=["object"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col])
    imputer = SimpleImputer(strategy="mean")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(imputer.fit_transform(X))
    X_train, X_test, y_train, y_test, _, smiles_test = train_test_split(
        X_scaled, y, smiles, test_size=0.2, random_state=10
    )

    results_log = pd.DataFrame({"SMILES": smiles_test, "Actual": y_test})
    features_log = pd.DataFrame(index=X.columns)
    metrics_log = pd.DataFrame()
    
    # Convert param_grid to a format compatible with BayesSearchCV
    skopt_space = {}
    for param, values in model_params.items():
        if isinstance(values[0], int):
            skopt_space[param] = Integer(low=min(values), high=max(values), prior='uniform')
        elif isinstance(values[0], float):
            skopt_space[param] = Real(low=min(values), high=max(values), prior='uniform')
        elif isinstance(values[0], str) or isinstance(values[0], bool):
            skopt_space[param] = Categorical(categories=values)
    
    with pd.ExcelWriter("results.xlsx") as writer:
        for fs_strategy in feature_selection_strategies:
            X_train_fs, selector = apply_feature_selection(
                fs_strategy, X_train, y_train
            )
            X_test_fs = selector.transform(X_test) if selector else X_test

            # Bayesian Optimization
            opt = BayesSearchCV(
                estimator=model(),
                search_spaces=skopt_space,
                n_iter=3,  # Number of iterations, increase for better results but longer runtime
                cv=5,       # Cross-validation folds
                n_jobs=-1,  # Use all available cores
                verbose=1,
                random_state=42
            )
            opt.fit(X_train_fs, y_train)

            # Best model after tuning
            model_instance = opt.best_estimator_
            predictions = model_instance.predict(X_test_fs)
            r2 = r2_score(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            print(f"{fs_strategy['name']} - Best Params: {opt.best_params_}: R2={r2}, MSE={mse}")

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
    if args.model in EXPERIMENT_CONFIGS:
        model = EXPERIMENT_CONFIGS[args.model]['model']
        model_params = EXPERIMENT_CONFIGS[args.model]['param_grid']
    else:
        raise Exception()

    run_experiment(X, y, model, model_params, feature_selection_strategies)

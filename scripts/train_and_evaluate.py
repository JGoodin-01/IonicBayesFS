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
from sklearn.linear_model import LinearRegression
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
    "lr": {
        "model": LinearRegression,
        "param_grid": {},
    },
    "rf": {
        "model": RandomForestRegressor,
        "param_grid": {
            "n_estimators": [10, 50, 100, 200],
            "max_depth": [None, 10, 20, 30, 40],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
        },
    },
}


def apply_feature_selection(fs_strategy, X, y):
    rankings = None

    if fs_strategy["name"] == "SelectKBest":
        selector = SelectKBest(mutual_info_regression, k="all").fit(X, y)
        rankings = selector.scores_.argsort()[::-1]  # Descending order of importance
    elif fs_strategy["name"] == "RFE":
        estimator = RandomForestRegressor(
            n_estimators=5, random_state=42
        )  # More estimators for stability
        selector = RFE(estimator, n_features_to_select=1).fit(X, y)
        rankings = selector.ranking_.argsort()
    elif fs_strategy["name"] == "BFS":
        selector = BFS()
        selector.fit(X, y)
        rankings = selector.get_feature_rankings()
    else:
        raise ValueError("Invalid feature selection strategy")

    return rankings


def run_experiment(X, y, model, model_params, feature_selection_strategies):
    model_name = model().__class__.__name__
    print(f"{model_name}:")

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
    if model_params:
        skopt_space = {}
        for param, values in model_params.items():
            if isinstance(values[0], int):
                skopt_space[param] = Integer(
                    low=min(values), high=max(values), prior="uniform"
                )
            elif isinstance(values[0], float):
                skopt_space[param] = Real(
                    low=min(values), high=max(values), prior="uniform"
                )
            elif isinstance(values[0], str) or isinstance(values[0], bool):
                skopt_space[param] = Categorical(categories=values)
    else:
        print(f"No hyperparameters for {model_name}, using default model parameters.")

    with pd.ExcelWriter(f"{model_name}_results.xlsx") as writer:
        for fs_strategy in feature_selection_strategies:
            if fs_strategy["name"] != "Base":
                ranking = apply_feature_selection(fs_strategy, X_train, y_train)
                best_score = -1

                # Iterate over a range of top N features, for example, 1 to the total number of features
                for N in range(1, len(X_train[0]) + 1):
                    top_features = ranking[:N]
                    X_train_sub = X_train[:, top_features]
                    X_test_sub = X_test[:, top_features]

                    if model_name == "RandomForestRegressor":
                        model_instance = model(n_estimators=5)
                    else:
                        model_instance = model()

                    model_instance.fit(X_train_sub, y_train)
                    y_pred = model_instance.predict(X_test_sub)
                    score = r2_score(y_test, y_pred)

                    if score > best_score:
                        best_score = score
                        fs_strategy["N"] = N

                top_features = ranking[: fs_strategy["N"]]
                X_train_opt = X_train[:, top_features].reshape(-1, fs_strategy["N"])
                X_test_opt = X_test[:, top_features].reshape(-1, fs_strategy["N"])
            else:
                ranking = range(0, len(X_train[0]))
                X_train_opt = X_train
                X_test_opt = X_test
                fs_strategy["N"] = len(ranking)

            if model_params:
                print("Performing Bayesian Optimization for Tuning")
                # Bayesian Optimization
                opt = BayesSearchCV(
                    estimator=model(),
                    search_spaces=skopt_space,
                    n_iter=10,  # Number of iterations, increase for better results but longer runtime
                    cv=3,  # Cross-validation folds
                    n_jobs=-1,  # Use all available cores
                    random_state=42,
                )
                opt.fit(X_train_opt, y_train)

                # Best model after tuning
                model_instance = opt.best_estimator_
                print(f"{fs_strategy['name']} - Best Params: {opt.best_params_}")
                predictions = model_instance.predict(X_test_opt)
            else:
                model_instance = model()
                model_instance.fit(X_train, y_train)
                predictions = model_instance.predict(X_test)

            r2 = r2_score(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            print(
                f"{fs_strategy['name']} - R2={r2}, MSE={mse} - Using {fs_strategy['N']} features."
            )

            results_log[f"{fs_strategy['name']}_Predicted"] = predictions
            selected_features_mask = [False] * len(X.columns)
            for i in range(0, fs_strategy["N"]):
                if ranking[i] < len(selected_features_mask):
                    selected_features_mask[ranking[i]] = True
                else:
                    print(
                        f"Warning: Attempted to access out-of-range index {ranking[i]}"
                    )

            features_log[fs_strategy["name"]] = pd.Series(
                selected_features_mask, index=X.columns
            )
            metrics_log[fs_strategy["name"]] = {"R2": r2, "MSE": mse}

        results_log.to_excel(writer, sheet_name="Predictions", index=False)
        features_log.to_excel(writer, sheet_name="Features")
        metrics_log.T.to_excel(writer, sheet_name="Metrics")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model evaluation with feature selection."
    )
    parser.add_argument("--model", type=str, choices=["lr", "rf"], help="Model name")
    args = parser.parse_args()

    data = pd.read_csv("./data/processed.csv").dropna(subset=["η / mPa s"])
    X, y = data.drop("η / mPa s", axis=1), data["η / mPa s"].values

    feature_selection_strategies = [
        {"name": "Base"},
        {"name": "SelectKBest"},
        {"name": "RFE"},
        {"name": "BFS"},
    ]
    if args.model in EXPERIMENT_CONFIGS:
        model = EXPERIMENT_CONFIGS[args.model]["model"]
        model_params = EXPERIMENT_CONFIGS[args.model]["param_grid"]
    else:
        raise Exception()

    run_experiment(X, y, model, model_params, feature_selection_strategies)

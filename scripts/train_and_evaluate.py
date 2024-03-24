import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from src.preprocessing.BFS import BFS
from src.preprocessing.DataPrepperMixin import DataPrepperMixin
from src.evaluation.LoggerMixin import LoggerMixin
from src.training.ModelOptimizationMixin import ModelOptimizationMixin
from src.EXPERIMENT_CONFIGS import EXPERIMENT_CONFIGS

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

    prepper = DataPrepperMixin()
    X_train, X_test, y_train, y_test, smiles_test = prepper.preprocess_data(X, y)

    logger = LoggerMixin()
    opt = ModelOptimizationMixin()
    opt.configure_search_space(model_params)
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

        opt.perform_bayesian_optimization(model(), X_train_opt, y_train)
        best_est = opt.best_estimator
        if opt.best_params:
            print(f"{fs_strategy['name']} - Best Params: {opt.best_params}")
        else:
            print(f"No hyperparameters for {model_name}, using defaults.")
            best_est.fit(X_train_opt, y_train)

        predictions = best_est.predict(X_test_opt)

        r2 = r2_score(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        print(
            f"{fs_strategy['name']} - R2={r2}, MSE={mse} - Using {fs_strategy['N']} features."
        )

        logger.log_results(smiles_test, y_test, predictions, fs_strategy["name"])
        logger.log_features(X, ranking, fs_strategy)
        logger.log_metrics(r2, mse, fs_strategy["name"])

    logger.save_logs(f"{model_name}_results.xlsx")


if __name__ == "__main__":
    models = ["lr", "rf"]
    data = pd.read_csv("./data/processed.csv").dropna(subset=["η / mPa s"])
    X, y = data.drop("η / mPa s", axis=1), data["η / mPa s"].values

    feature_selection_strategies = [
        {"name": "Base"},
        {"name": "SelectKBest"},
        {"name": "RFE"},
        {"name": "BFS"},
    ]
    for model_key in models:
        model_config = EXPERIMENT_CONFIGS[model_key]
        model_class = model_config["model"]
        model_params = model_config["param_grid"]
        run_experiment(X, y, model_class, model_params, feature_selection_strategies)

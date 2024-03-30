import numpy as np
from src.preprocessing.DataPrepperMixin import DataPrepperMixin
from src.evaluation.LoggerMixin import LoggerMixin
from src.training.ModelOptimizationMixin import ModelOptimizationMixin
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from src.preprocessing.BFS import BFS
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


class ExperimentRunner:
    def __init__(self):
        self.model = None

    @staticmethod
    def apply_feature_selection(fs_strategy, X, y):
        rankings = None

        if fs_strategy["name"] == "SelectKBest":
            selector = SelectKBest(mutual_info_regression, k="all").fit(X, y)
            rankings = selector.scores_.argsort()[
                ::-1
            ]  # Descending order of importance
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

    def run_experiment(self, X, y, model, model_params, feature_selection_strategies):
        model_name = model().__class__.__name__
        print(f"{model_name}:")

        prepper = DataPrepperMixin()
        X_train, X_test, y_train, y_test, smiles_test = prepper.preprocess_data(X, y)

        logger = LoggerMixin()
        opt = ModelOptimizationMixin()
        opt.configure_search_space(model_params)
        for fs_strategy in feature_selection_strategies:
            if fs_strategy["name"] != "Base":
                ranking = self.apply_feature_selection(fs_strategy, X_train, y_train)
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

    def run_cross_experiment(
        self, X, y, model, model_params, feature_selection_strategies, n_splits=2
    ):
        model_name = model().__class__.__name__
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        X = X.drop("SMILES", axis=1)
        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col])
        imputer = SimpleImputer(strategy="mean")
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(imputer.fit_transform(X))

        # Logging and Optimization instances
        logger = LoggerMixin()
        opt = ModelOptimizationMixin()
        opt.configure_search_space(model_params)
        for fs_strategy in feature_selection_strategies:
            r2_scores = []
            mse_scores = []

            for train_index, test_index in kf.split(X):
                X_train, X_test = X_scaled[train_index], X_scaled[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # Apply feature selection if not 'Base'
                if fs_strategy["name"] != "Base":
                    ranking = self.apply_feature_selection(
                        fs_strategy, X_train, y_train
                    )
                    best_score = -1

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
                    X_train_opt = X_train[:, top_features]
                    X_test_opt = X_test[:, top_features]
                else:
                    ranking = range(0, len(X_train[0]))
                    X_train_opt = X_train
                    X_test_opt = X_test
                    fs_strategy["N"] = len(ranking)

                # Hyperparameter optimization and model evaluation
                opt.perform_bayesian_optimization(model(), X_train_opt, y_train)
                best_est = opt.best_estimator
                if opt.best_params:
                    print(f"{fs_strategy['name']} - Best Params: {opt.best_params}")
                else:
                    best_est.fit(X_train_opt, y_train)

                predictions = best_est.predict(X_test_opt)
                r2 = r2_score(y_test, predictions)
                mse = mean_squared_error(y_test, predictions)

                r2_scores.append(r2)
                mse_scores.append(mse)

                logger.log_features(X, ranking, fs_strategy)
                logger.log_metrics(r2, mse, fs_strategy["name"])

            # Calculate average scores after all folds
            avg_r2 = np.mean(r2_scores)
            avg_mse = np.mean(mse_scores)
            print(
                f"{fs_strategy['name']} - Average R2: {avg_r2}, Average MSE: {avg_mse}"
            )

        # Save logs after each strategy (Adjust according to your needs)
        logger.save_logs(f"{model().__class__.__name__}_cross_val_results.xlsx")

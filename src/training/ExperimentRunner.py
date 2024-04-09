import numpy as np
from src.preprocessing.DataPrepperMixin import DataPrepperMixin
from src.evaluation.ExcelLogger import ExcelLogger
from src.training.OptimizationManager import OptimizationManager
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from src.preprocessing.FeatureSelectionMixin import FeatureSelectionMixin


class ExperimentRunner(DataPrepperMixin, FeatureSelectionMixin):
    def __init__(self, config):
        self.model = None
        self.config = config
        self.logger = ExcelLogger()
        self.opt = OptimizationManager()

    def record_predictions(self, model, X, y, strategy_name, phase, fold_index):
        predictions = model.predict(X)
        r2 = r2_score(y, predictions)
        mse = mean_squared_error(y, predictions)
        self.logger.log_metrics(r2, mse, strategy_name, phase, fold_index)

        if phase == "Testing":
            self.logger.log_predictions(predictions, strategy_name, fold_index)

        return r2, mse

    def reset_experiment(self):
        self.logger.clear_logs()
        self.opt.reset_space()

    def run_cross_experiment(self, X, y, feature_selection_strategies, n_splits=5):
        X_train_full_scaled, X_test_scaled, y_train_full, y_test = (
            self.split_and_scale_data(X, y)
        )
        self.logger.set_actual_test_values(y_test)

        for model_details in self.config:
            model = model_details["model"]
            model_params = model_details["param_grid"]
            model_name = model().__class__.__name__
            print(f"Running experiment for {model_name}...")

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            self.opt.configure_search_space(model_params)
            for fs_strategy in feature_selection_strategies:
                val_r2_scores, test_r2_scores = [], []
                val_mse_scores, test_mse_scores = [], []

                for fold_index, (train_index, val_index) in enumerate(
                    kf.split(X_train_full_scaled)
                ):
                    X_train, X_val = (
                        X_train_full_scaled[train_index],
                        X_train_full_scaled[val_index],
                    )
                    y_train, y_val = y_train_full[train_index], y_train_full[val_index]

                    # Apply feature selection if not 'Base'
                    if fs_strategy["name"] != "Base":
                        ranking = self.apply_feature_selection(
                            fs_strategy, X_train, y_train
                        )
                        best_score = -1

                        for N in range(1, len(X_train[0]) + 1):
                            top_features = ranking[:N]
                            X_train_sub = X_train[:, top_features]
                            X_val_sub = X_val[:, top_features]

                            if model_name == "RandomForestRegressor":
                                model_instance = model(n_estimators=5)
                            else:
                                model_instance = model()
                            model_instance.fit(X_train_sub, y_train)
                            y_pred = model_instance.predict(X_val_sub)
                            score = r2_score(y_val, y_pred)

                            if score > best_score:
                                best_score = score
                                fs_strategy["N"] = N

                        top_features = ranking[: fs_strategy["N"]]
                        X_train_opt = X_train[:, top_features]
                        X_val_opt = X_val[:, top_features]
                        X_test_opt = X_test_scaled[:, top_features]
                    else:
                        ranking = range(0, len(X_train[0]))
                        X_train_opt = X_train
                        X_val_opt = X_val
                        X_test_opt = X_test_scaled
                        fs_strategy["N"] = len(ranking)

                    # Hyperparameter optimization and model evaluation
                    self.opt.perform_bayesian_optimization(
                        model(), X_train_opt, y_train
                    )
                    best_est = self.opt.best_estimator
                    if self.opt.best_params:
                        print(
                            f"{fs_strategy['name']} - Best Params: {self.opt.best_params}"
                        )
                        self.logger.log_params(fs_strategy["name"], fold_index, self.opt.best_params)
                    else:
                        best_est.fit(X_train_opt, y_train)

                    val_r2, val_mse = self.record_predictions(
                        best_est,
                        X_val_opt,
                        y_val,
                        fs_strategy["name"],
                        "Validation",
                        fold_index,
                    )
                    test_r2, test_mse = self.record_predictions(
                        best_est,
                        X_test_opt,
                        y_test,
                        fs_strategy["name"],
                        "Testing",
                        fold_index,
                    )

                    val_r2_scores.append(val_r2)
                    val_mse_scores.append(val_mse)

                    test_r2_scores.append(test_r2)
                    test_mse_scores.append(test_mse)

                    self.logger.log_features(X, ranking, fs_strategy, fold_index)

                # Calculate and log average scores after all folds for both validation and testing
                val_avg_r2 = np.mean(val_r2_scores)
                val_avg_mse = np.mean(val_mse_scores)
                test_avg_r2 = np.mean(test_r2_scores)
                test_avg_mse = np.mean(test_mse_scores)

                print(
                    f"VALIDATION: {fs_strategy['name']} - Average R2: {val_avg_r2}, Average MSE: {val_avg_mse}"
                )
                print(
                    f"TESTING: {fs_strategy['name']} - Average R2: {test_avg_r2}, Average MSE: {test_avg_mse}"
                )

            self.logger.save_logs(f"{model().__class__.__name__}_results.xlsx")
            self.reset_experiment()

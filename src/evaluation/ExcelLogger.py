import pandas as pd


class ExcelLogger:
    def __init__(self):
        self.predictions_log = pd.DataFrame()  # Renamed for clarity
        self.features_log = pd.DataFrame()
        self.metrics_log = pd.DataFrame()
        self.params_log = pd.DataFrame()
        self.actual_test_values = None

    def set_actual_test_values(self, y_test):
        # Store the actual test values only once
        if self.actual_test_values is None:
            self.actual_test_values = y_test
            self.predictions_log["Actual"] = self.actual_test_values

    def log_predictions(self, predictions, strategy_name, fold_index=0):
        # Log only the predictions, ensuring the actual values are stored separately
        col_name = f"{strategy_name}_Predicted_{fold_index}"
        self.predictions_log[col_name] = predictions

    def log_features(self, X, ranking, fs_strategy, fold_index=0):
        if fs_strategy["name"] == "Base":
            feature_rankings = {feature: 1 for feature in X.columns}
        else:
            feature_rankings = {feature: None for feature in X.columns}
            for rank, feature_index in enumerate(ranking[:fs_strategy["N"]], start=1):
                if feature_index < len(X.columns):
                    feature_name = X.columns[feature_index]
                    feature_rankings[feature_name] = rank
                else:
                    print(f"Warning: Accessed out-of-range index {feature_index}")

        features_df = pd.DataFrame(feature_rankings, index=[f"{fs_strategy['name']}_{fold_index}"])
        if self.features_log.empty:
            self.features_log = features_df
        else:
            self.features_log = pd.concat([self.features_log, features_df])

    def log_metrics(self, r2, mse, strategy_name, phase, fold_index=0):
        name = f"{strategy_name}_{phase}_{fold_index}"
        new_entry = pd.DataFrame({name: {"R2": r2, "MSE": mse}})
        if name in self.metrics_log:
            self.metrics_log[name].update(new_entry)
        else:
            self.metrics_log = pd.concat([self.metrics_log, new_entry], axis=1)

    def log_params(self, strategy_name, fold_index, params):
        experiment_id = f"{strategy_name}_{fold_index}"

        params_row = pd.DataFrame({"Strategy_Fold": [experiment_id]})
        for param, value in params.items():
            params_row[param] = value

        self.params_log = pd.concat(
            [self.params_log, params_row], ignore_index=True, sort=False
        )
        self.params_log = self.params_log[
            ["Strategy_Fold"]
            + [col for col in self.params_log.columns if col != "Strategy_Fold"]
        ]

    def save_logs(self, filename):
        with pd.ExcelWriter(filename) as writer:
            if not self.predictions_log.empty:
                self.predictions_log.to_excel(
                    writer, sheet_name="Predictions", index=False
                )
            if not self.features_log.empty:
                self.features_log.to_excel(writer, sheet_name="Features")
            if not self.metrics_log.empty:
                self.metrics_log.T.to_excel(writer, sheet_name="Metrics")
            if not self.params_log.empty:
                self.params_log.to_excel(writer, sheet_name="Parameters", index=False)

    def clear_logs(self):
        """Reset the logs to their initial empty state."""
        self.predictions_log = pd.DataFrame()
        self.features_log = pd.DataFrame()
        self.metrics_log = pd.DataFrame()
        self.predictions_log["Actual"] = self.actual_test_values

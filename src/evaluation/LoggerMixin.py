import pandas as pd


class LoggerMixin:
    def __init__(self):
        self.predictions_log = pd.DataFrame()  # Renamed for clarity
        self.features_log = pd.DataFrame()
        self.metrics_log = pd.DataFrame()
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
        selected_features_mask = [False] * len(X.columns)
        for i in range(0, fs_strategy["N"]):
            if ranking[i] < len(selected_features_mask):
                selected_features_mask[ranking[i]] = True
            else:
                print(f"Warning: Accessed out-of-range index {ranking[i]}")

        if self.features_log.empty:
            self.features_log = pd.DataFrame(index=X.columns)
        name = f"{fs_strategy['name']}_{fold_index}"
        self.features_log[name] = pd.Series(selected_features_mask, index=X.columns)

    def log_metrics(self, r2, mse, strategy_name, phase, fold_index=0):
        # Adjusted to include phase (Validation/Testing) in the logging
        name = f"{strategy_name}_{phase}_{fold_index}"
        new_entry = pd.DataFrame({name: {"R2": r2, "MSE": mse}})
        if name in self.metrics_log:
            self.metrics_log[name].update(new_entry)
        else:
            self.metrics_log = pd.concat([self.metrics_log, new_entry], axis=1)

    def save_logs(self, filename):
        with pd.ExcelWriter(filename) as writer:
            if not self.predictions_log.empty:
                self.predictions_log.to_excel(writer, sheet_name="Predictions")
            if not self.features_log.empty:
                self.features_log.to_excel(writer, sheet_name="Features")
            if not self.metrics_log.empty:
                self.metrics_log.T.to_excel(writer, sheet_name="Metrics")

    def clear_logs(self):
        """Reset the logs to their initial empty state."""
        self.predictions_log = pd.DataFrame()
        self.features_log = pd.DataFrame()
        self.metrics_log = pd.DataFrame()
        self.predictions_log["Actual"] = self.actual_test_values

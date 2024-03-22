import pandas as pd

class LoggerMixin:
    def __init__(self):
        self.results_log = pd.DataFrame()
        self.features_log = pd.DataFrame()
        self.metrics_log = pd.DataFrame()

    def log_results(self, smiles_test, y_test, predictions, strategy_name):
        if self.results_log.empty:
            self.results_log = pd.DataFrame({"SMILES": smiles_test, "Actual": y_test})
        self.results_log[f"{strategy_name}_Predicted"] = predictions

    def log_features(self, X, ranking, fs_strategy):
        selected_features_mask = [False] * len(X.columns)
        for i in range(0, fs_strategy["N"]):
            if ranking[i] < len(selected_features_mask):
                selected_features_mask[ranking[i]] = True
            else:
                print(
                    f"Warning: Attempted to access out-of-range index {ranking[i]}"
                )
        
        if self.features_log.empty:
            self.features_log = pd.DataFrame(index=X.columns)
        self.features_log[fs_strategy["name"]] = pd.Series(selected_features_mask, index=X.columns)

    def log_metrics(self, r2, mse, strategy_name):
        self.metrics_log[strategy_name] = {"R2": r2, "MSE": mse}

    def save_logs(self, filename):
        with pd.ExcelWriter(filename) as writer:
            self.results_log.to_excel(writer, sheet_name="Predictions", index=False)
            self.features_log.to_excel(writer, sheet_name="Features")
            pd.DataFrame(self.metrics_log).T.to_excel(writer, sheet_name="Metrics")

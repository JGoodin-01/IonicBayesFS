from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

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

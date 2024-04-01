from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso

EXPERIMENT_CONFIGS = [
    {
        "model": LinearRegression,
        "param_grid": {},
    },
    {
        "model": Ridge,
        "param_grid": {
            "alpha": [0.1, 1.0, 10.0, 100.0],  # Regularization strength for Ridge
            "tol": [0.0001, 0.001, 0.01],  # Optimization tolerance
            "max_iter": [None, 1000, 5000, 10000],  # Maximum number of iterations
            "random_state": [None, 42]  # Reproducibility of results
        },
    },
    {
        "model": Lasso,
        "param_grid": {
            "alpha": [0.1, 1.0, 10.0, 100.0],
            "tol": [0.0001, 0.001, 0.01],
            "max_iter": [None, 1000, 5000, 10000],
            "random_state": [None, 42]  # Reproducibility of results
        },
    },
    {
        "model": RandomForestRegressor,
        "param_grid": {
            "n_estimators": [10, 50, 100, 200],
            "max_depth": [None, 10, 20, 30, 40],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
        },
    },
]

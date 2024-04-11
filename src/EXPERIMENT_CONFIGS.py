from sklearn.linear_model import LinearRegression, Ridge, Lasso
from skopt.space import Real, Integer, Categorical

EXPERIMENT_CONFIGS = [
    {
        "model": LinearRegression,
        "param_grid": [],
    },
    {
        "model": Ridge,
        "param_grid": [
            Real(0.1, 100.0, prior="log-uniform", name="alpha"),
            Real(0.0001, 0.01, prior="log-uniform", name="tol"),
            Categorical([None, 1000, 5000, 10000], name="max_iter"),
            Categorical([None, 10, 42], name="random_state"),
            Categorical([True, False], name="positive"),
        ],
    },
    {
        "model": Lasso,
        "param_grid": [
            Real(0.1, 100.0, prior="log-uniform", name="alpha"),
            Real(0.0001, 0.01, prior="log-uniform", name="tol"),
            Integer(10000, 15000, name="max_iter"),
            Categorical([None, 10, 42], name="random_state"),
            Categorical(["cyclic", "random"], name="selection"),
        ],
    },
]


def cuda_available():
    try:
        import cuml

        return True
    except ImportError:
        return False


if cuda_available():
    print("Using RAPIDS cuML for GPU acceleration.")
    from cuml.ensemble import RandomForestRegressor

    EXPERIMENT_CONFIGS.append(
        {
            "model": RandomForestRegressor,
            "cuML_used": True,
            "param_grid": [
                Categorical([2, 4, 5, 6], name="split_criterion"),
                Integer(10, 200, name="n_estimators"),
                Integer(1, 40, name="max_depth"),
                Integer(2, 10, name="min_samples_split"),
                Integer(100, 512, name="n_bins"),
                Integer(1, 3, name="n_streams"),
                Integer(1, 2, name="min_samples_leaf"),
                Categorical(["sqrt", "log2", "auto"], name="max_features"),
            ],
        },
    )

else:
    print("CUDA not available. Falling back to scikit-learn.")
    from sklearn.ensemble import RandomForestRegressor

    EXPERIMENT_CONFIGS.append(
        {
            "model": RandomForestRegressor,
            "param_grid": [
                Categorical(
                    ["squared_error", "absolute_error", "friedman_mse", "poisson"],
                    name="criterion",
                ),
                Integer(10, 200, name="n_estimators"),
                Categorical([None, 10, 20, 30, 40], name="max_depth"),
                Integer(2, 10, name="min_samples_split"),
                Integer(1, 4, name="min_samples_leaf"),
                Categorical(["sqrt", "log2", None], name="max_features"),
            ],
        },
    )

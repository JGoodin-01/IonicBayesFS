from sklearn.linear_model import LinearRegression, Ridge, Lasso


EXPERIMENT_CONFIGS = [
    # {
    #     "model": LinearRegression,
    #     "param_grid": {},
    # },
    {
        "model": Ridge,
        "param_grid": {
            "alpha": [0.1, 1.0, 10.0, 100.0],  # Regularization strength for Ridge
            "tol": [0.0001, 0.001, 0.01],  # Optimization tolerance
            "max_iter": [None, 1000, 5000, 10000],  # Maximum number of iterations
            "random_state": [None, 42]  # Reproducibility of results
        },
    },
    # {
    #     "model": Lasso,
    #     "param_grid": {
    #         "alpha": [0.1, 1.0, 10.0, 100.0],
    #         "tol": [0.0001, 0.001, 0.01],
    #         "max_iter": [None, 1000, 5000, 10000],
    #         "random_state": [None, 42]  # Reproducibility of results
    #     },
    # },
]


# # Function to check CUDA availability
# def cuda_available():
#     try:
#         import cuml

#         return True
#     except ImportError:
#         return False


# if cuda_available():
#     print("Using RAPIDS cuML for GPU acceleration.")
#     from cuml.ensemble import RandomForestRegressor

#     EXPERIMENT_CONFIGS.append(
#         {
#             "model": RandomForestRegressor,
#             "param_grid": {
#                 "split_criterion": [4, 5],
#                 "n_estimators": [10, 50, 100, 200],
#                 "max_depth": [10, 20, 30, 40],
#                 "min_samples_split": [2, 5, 10],
#                 "min_samples_leaf": [1, 2, 4],
#                 "max_features": ["sqrt", "log2", "auto"],
#             },
#         },
#     )

# else:
#     print("CUDA not available. Falling back to scikit-learn.")
#     from sklearn.ensemble import RandomForestRegressor

#     EXPERIMENT_CONFIGS.append(
#         {
#             "model": RandomForestRegressor,
#             "param_grid": {
#                 "criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"],
#                 "n_estimators": [10, 50, 100, 200],
#                 "max_depth": [None, 10, 20, 30, 40],
#                 "min_samples_split": [2, 5, 10],
#                 "min_samples_leaf": [1, 2, 4],
#                 "max_features": ["sqrt", "log2", None],
#             },
#         },
#     )

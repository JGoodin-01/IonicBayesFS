from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer


class ModelOptimizationMixin:
    def __init__(self):
        self.skopt_space = {}
        self.best_estimator = None
        self.best_params = {}

    def configure_search_space(self, model_params):
        """
        Convert model parameters to a search space compatible with BayesSearchCV.
        """
        if model_params:
            for param, values in model_params.items():
                if isinstance(values[0], int):
                    self.skopt_space[param] = Integer(
                        low=min(values), high=max(values), prior="uniform"
                    )
                elif isinstance(values[0], float):
                    self.skopt_space[param] = Real(
                        low=min(values), high=max(values), prior="uniform"
                    )
                elif isinstance(values[0], str) or isinstance(values[0], bool):
                    self.skopt_space[param] = Categorical(categories=values)

    def perform_bayesian_optimization(
        self, estimator, X, y, n_iter=25, cv=3, n_jobs=-1, random_state=42
    ):
        """
        Perform Bayesian Optimization to find the best model hyperparameters.
        """
        if self.skopt_space:
            print("Performing Bayesian Optimization for Tuning")
            opt = BayesSearchCV(
                estimator=estimator,
                search_spaces=self.skopt_space,
                n_iter=10,
                cv=3,
                n_jobs=-1,
                random_state=42,
            )
            opt.fit(X, y)

            # Best model after tuning
            self.best_estimator = opt.best_estimator_
            self.best_params = opt.best_params_
        else:
            self.best_estimator = estimator

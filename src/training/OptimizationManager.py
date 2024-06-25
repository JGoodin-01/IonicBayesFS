from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import warnings


class OptimizationManager:
    def __init__(self):
        self.skopt_space = {}
        self.best_estimator = None
        self.best_params = {}
        self.cuML = False

    def configure_search_space(self, model_params):
        """
        Convert model parameters to a search space compatible with BayesSearchCV.
        """
        if model_params["param_grid"]:
            self.skopt_space.update({param.name: param for param in model_params["param_grid"]})
            if "cuML_used" in model_params:
                self.cuML = True

    def perform_bayesian_optimization(
        self, estimator, X, y, n_iter=25, cv=3, n_jobs=-1, random_state=42
    ):
        """
        Perform Bayesian Optimization to find the best model hyperparameters.
        """
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="The objective has been evaluated at point*",
        )

        if self.skopt_space:
            if self.cuML:
                n_jobs = 1

            opt = BayesSearchCV(
                estimator=estimator,
                search_spaces=self.skopt_space,
                n_iter=n_iter,
                cv=cv,
                n_jobs=n_jobs,
                random_state=random_state,
            )
            opt.fit(X, y)

            # Best model after tuning
            self.best_estimator = opt.best_estimator_
            self.best_params = opt.best_params_
        else:
            self.best_estimator = estimator

    def reset_space(self):
        self.skopt_space = {}
        self.best_estimator = None
        self.best_params = {}
        self.cuML = False

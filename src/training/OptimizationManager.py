from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

class OptimizationManager:
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
                if None in values:
                    self.skopt_space[param] = Categorical(categories=values)
                elif all(isinstance(value, int) for value in values):
                    self.skopt_space[param] = Integer(low=min(values), high=max(values), prior="uniform")
                elif all(isinstance(value, (int, float)) for value in values):
                    self.skopt_space[param] = Real(low=min(values), high=max(values), prior="uniform")

    def perform_bayesian_optimization(
        self, estimator, X, y, n_iter=25, cv=3, n_jobs=6, random_state=42
    ):
        """
        Perform Bayesian Optimization to find the best model hyperparameters.
        """
        if self.skopt_space:
            print("Performing Bayesian Optimization for Tuning")
            opt = BayesSearchCV(
                estimator=estimator,
                search_spaces=self.skopt_space,
                n_iter=n_iter,  # Reflect the n_iter argument
                cv=cv,
                n_jobs=n_jobs,
                random_state=random_state,
            )
            opt.fit(X, y)

            # Ensure integer parameters are integers
            best_params_corrected = {param: int(value) if isinstance(value, float) and param in self.skopt_space and isinstance(self.skopt_space[param], Integer) else value for param, value in opt.best_params_.items()}

            # Best model after tuning
            self.best_estimator = opt.best_estimator_
            self.best_params = best_params_corrected
        else:
            print("No search space configured. Using default estimator parameters.")
            self.best_estimator = estimator

    def reset_space(self):
        self.skopt_space = {}
        self.best_estimator = None
        self.best_params = {}

import numpy as np
import pymc as pm


class BFS:
    def __init__(self, priors_alpha=1.0, priors_beta=1.0, error_beta=10, mu=0, sigma=1):
        self.priors_alpha = priors_alpha
        self.priors_beta = priors_beta
        self.error_beta = error_beta
        self.mu = mu
        self.sigma = sigma
        self.selected_features = None

    def model_define(self, X, y):
        # Step 2: Model Definition
        with pm.Model() as feature_selection_model:
            # Priors on the inclusion of features
            inclusion_probs = pm.Beta(
                "inclusion_probs", alpha=1.0, beta=1.0, shape=X.shape[1]
            )
            included_features = pm.Bernoulli(
                "included_features", p=inclusion_probs, shape=X.shape[1]
            )

            # Model error
            sigma = pm.HalfCauchy("sigma", beta=10)

            # Expected value of outcome, considering only included features
            weights = pm.Normal("weights", mu=0, sigma=1, shape=X.shape[1])
            y_hat = pm.math.dot(X * included_features, weights)

            # Likelihood (sampling distribution) of observations
            y_obs = pm.Normal("y_obs", mu=y_hat, sigma=sigma, observed=y)

        with feature_selection_model:
            self.trace = pm.sample(1000, tune=1000, return_inferencedata=True)

    def fit(self, X_train, y_train):
        # Normalize features for better convergence
        X_normalized = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

        self.model_define(X_normalized, y_train)

        posterior_inclusion_probs = np.mean(
            self.trace.posterior["included_features"].values, axis=(0, 1)
        )

        # Select features with a high probability of inclusion
        self.selected_features = np.where(posterior_inclusion_probs > 0.5)[0]

    def transform(self, X):
        if self.selected_features is None:
            raise "BFS selector has not been fitted before transformation performed."

        return X[:, self.selected_features]

    def get_support(self):
        if self.selected_features is None:
            raise RuntimeError("BFS selector has not been fitted before calling get_support.")
        
        # Create a boolean mask with the same length as the number of features
        support_mask = np.zeros(self.trace.posterior["included_features"].shape[-1], dtype=bool)
        support_mask[self.selected_features] = True
        return support_mask


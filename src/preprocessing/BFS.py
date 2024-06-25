import os
import numpy as np
import matplotlib as plt
import pymc as pm
import arviz as az

class BFS:
    def __init__(self, priors_alpha=1.0, priors_beta=1.0, error_beta=10, mu=0, sigma=1):
        self.priors_alpha = priors_alpha
        self.priors_beta = priors_beta
        self.error_beta = error_beta
        self.mu = mu
        self.sigma = sigma
        self.selected_features = None

    def model_define(self, X, y):
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
            self.trace = pm.sample(1000, tune=1000)

    def get_feature_rankings(self):
        if self.selected_features is None:
            raise RuntimeError("BFS selector has not been fitted.")

        # Calculate the mean posterior inclusion probability for each feature
        posterior_inclusion_probs = np.mean(
            self.trace.posterior["included_features"].values, axis=(0, 1)
        )
        
        # Rank features based on these probabilities (higher is better)
        feature_rankings = np.argsort(posterior_inclusion_probs)[::-1]  # Descending order
        
        return feature_rankings

    def fit(self, X_train, y_train):
        # Normalize features for better convergence
        X_normalized = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
        X_normalized = X_normalized.astype('float32')
        y_train = y_train.astype('float32')

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

    def traceplot(self):
        # Ensure the 'images' directory exists
        os.makedirs('images', exist_ok=True)

        # Set the ArviZ style
        az.style.use("arviz-darkgrid")
        
        # Plot trace for specified variables using ArviZ
        var_names = ["inclusion_probs"]
        az_trace_data = az.plot_trace(self.trace, var_names=var_names)
        
        # Check if plot_trace returns a tuple of (fig, ax) or just ax
        if isinstance(az_trace_data, tuple):
            _, ax = az_trace_data
        else:
            ax = az_trace_data
        
        # Save the ArviZ trace plot
        for i, var_name in enumerate(var_names):
            for j in range(2):
                fig = ax[i, j].get_figure()
                fig.savefig(f'images/traceplot.png', format='png')
                break
            break

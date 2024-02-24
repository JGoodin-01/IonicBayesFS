from sklearn.feature_selection import SelectKBest as SKB, f_classif


class SelectKBest:
    def __init__(self, k=10):
        self.k = k
        self.selector = None

    def fit(self, X, y):
        # Ensure 'k' is only passed once, as a keyword argument
        self.selector = SKB(score_func=f_classif, k=self.k)
        self.selector.fit(X, y)

    def transform(self, X):
        if not self.selector:
            raise RuntimeError("You must fit the selector before transforming.")
        return self.selector.transform(X)

    def get_selected_indices(self):
        # Return the indices of selected features
        return self.selector.get_support(indices=True)
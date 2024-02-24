class BaseModel:
    """
    Base class for the different ML models.
    """

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_test):
        raise NotImplementedError

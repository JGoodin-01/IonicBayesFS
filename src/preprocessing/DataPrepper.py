from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

class DataPrepper:
    def preprocess_data(self, X, y, test_size=0.2, random_state=10):
        smiles = X["SMILES"].values
        X = X.drop("SMILES", axis=1)
        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col])
        imputer = SimpleImputer(strategy="mean")
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(imputer.fit_transform(X))
        X_train, X_test, y_train, y_test, _, smiles_test = train_test_split(
            X_scaled, y, smiles, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test, smiles_test

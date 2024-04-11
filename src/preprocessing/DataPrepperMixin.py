from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


class DataPrepperMixin:
    @staticmethod
    def split_and_scale_data(X, y, test_portion=0.2, random_state=42):
        X = X.drop("SMILES", axis=1)
        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col])

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=test_portion, random_state=random_state
        )

        imputer = SimpleImputer(strategy="mean")
        scaler = MinMaxScaler()
        X_train_full_scaled = scaler.fit_transform(imputer.fit_transform(X_train_full))
        X_test_scaled = scaler.transform(imputer.transform(X_test))

        return (
            X_train_full_scaled.astype("float32"),
            X_test_scaled.astype("float32"),
            y_train_full.astype("float32"),
            y_test.astype("float32"),
        )

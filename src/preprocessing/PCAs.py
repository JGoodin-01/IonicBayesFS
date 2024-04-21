import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

# Suppress specific sklearn warnings
warnings.filterwarnings(
    "ignore", message="X has feature names, but PCA was fitted without feature names"
)

data = pd.read_csv("./data/processed.csv")

# Define the feature groups based on their chemical or functional properties
feature_groups = {
    "EState_Indexes": [
        "MaxAbsEStateIndex",
        "MaxEStateIndex",
        "MinAbsEStateIndex",
        "MinEStateIndex",
    ],
    "BCUT2D": [col for col in data.columns if "BCUT2D" in col],
    "Chi": [col for col in data.columns if "Chi" in col],
    "Kappa": [col for col in data.columns if "Kappa" in col],
    "PEOE_VSA": [col for col in data.columns if "PEOE_VSA" in col],
    "SMR_VSA": [col for col in data.columns if "SMR_VSA" in col],
    "SlogP_VSA": [col for col in data.columns if "SlogP_VSA" in col],
    "EState_VSA": [col for col in data.columns if "EState_VSA" in col],
    "VSA_EState": [col for col in data.columns if "VSA_EState" in col],
    "Functional_Groups": [col for col in data.columns if col.startswith("fr_")],
}


# Function to perform PCA on a group of features, capturing up to 95% of variance
def perform_pca(features):
    group_data = data[features]
    scaler = StandardScaler()
    # Explicit conversion to numpy array to avoid feature name warnings
    scaled_group_data = scaler.fit_transform(group_data.values)
    pca = PCA(n_components=0.95)
    pca.fit(scaled_group_data)
    return pca


# Function to extract top principal components and rename them for clarity
def extract_components(group_name, pca_model, features):
    n_components = (pca_model.explained_variance_ratio_.cumsum() <= 0.95).sum() + 1
    principal_components = pca_model.transform(data[features])[:, :n_components]
    component_names = [f"{group_name}_PC{i+1}" for i in range(n_components)]
    return pd.DataFrame(principal_components, columns=component_names), n_components


if __name__ == "__main__":
    # Applying PCA to each feature group and collecting all new PCA features
    pca_features = pd.DataFrame()
    input_features, output_features = 0, 0
    for group_name, features in feature_groups.items():
        pca_model = perform_pca(features)
        pca_components, n_components = extract_components(
            group_name, pca_model, features
        )
        pca_features = pd.concat([pca_features, pca_components], axis=1)
        input_features += len(features)
        output_features += n_components

    for group_name, features in feature_groups.items():
        data.drop(columns=features, inplace=True)
        data = pd.concat([data, pca_features.filter(regex=f"^{group_name}_PC")], axis=1)

    # Save the modified dataset and the PCA features
    data.to_csv("./data/processed_with_pca.csv", index=False)

    print(f"Total input features: {input_features}")
    print(f"Total output features w/ PCAs: {output_features}")

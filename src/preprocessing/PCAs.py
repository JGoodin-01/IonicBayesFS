import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
from tabulate import tabulate

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
    scaled_group_data = pd.DataFrame(scaler.fit_transform(group_data), columns=group_data.columns)
    pca = PCA(n_components=0.95)
    pca.fit(scaled_group_data)
    return pca, scaled_group_data, pca.components_

# Function to extract top principal components and rename them for clarity
def extract_components(group_name, pca_model, scaled_group_data, components):
    n_components = (pca_model.explained_variance_ratio_.cumsum() <= 0.95).sum() + 1
    principal_components = pca_model.transform(scaled_group_data)[:, :n_components]
    component_names = [f"{group_name}_PC{i+1}" for i in range(n_components)]
    loadings_df = pd.DataFrame(components[:n_components, :], index=component_names, columns=scaled_group_data.columns)
    return pd.DataFrame(principal_components, columns=component_names), loadings_df, n_components

if __name__ == "__main__":
    pca_features = pd.DataFrame()
    loadings_dfs = []
    input_features, output_features = 0, 0
    feature_details = []

    for group_name, features in feature_groups.items():
        pca_model, scaled_group_data, components = perform_pca(features)
        pca_components, loadings_df, n_components = extract_components(group_name, pca_model, scaled_group_data, components)
        pca_features = pd.concat([pca_features, pca_components], axis=1)
        loadings_dfs.append(loadings_df)
        feature_details.append([group_name, len(features), n_components])
        input_features += len(features)
        output_features += n_components

    # Concatenate all loadings into a single DataFrame and save
    all_loadings = pd.concat(loadings_dfs)
    all_loadings.to_csv("./data/pca_loadings.csv")

    for group_name, features in feature_groups.items():
        data.drop(columns=features, inplace=True)
        data = pd.concat([data, pca_features.filter(regex=f"^{group_name}_PC")], axis=1)

    print(tabulate(feature_details, headers=['Group Name', 'Original Features', 'PCA Features'], tablefmt='grid'))
    print(f"Total input features: {input_features}")
    print(f"Total output features w/ PCAs: {output_features}")

    # Save the modified dataset and the PCA features
    data.to_csv("./data/processed_with_pca.csv", index=False)
    print("Data and PCA loadings have been saved.")

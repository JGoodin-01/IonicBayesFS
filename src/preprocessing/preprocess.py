import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

RAW_FILE_PATH = "./data/raw.xlsx"
PROCESSED_FILE_PATH = "./data/processed.csv"


def load_workbook_sheets(file_path):
    # Load specified sheets from an Excel workbook at once
    xl = pd.ExcelFile(file_path)
    ions_sheet = xl.parse(sheet_name="S2 | Ions")
    database_sheet = xl.parse(
        sheet_name='S8 | Modeling vs "raw" database',
        usecols=[
            "Cation",
            "Anion",
            "Cationic family",
            "Anionic family",
            "Excluded IL",
            "Accepted dataset",
            "T / K",
            "η / mPa s",
        ],
    )
    return ions_sheet, database_sheet


def preprocess_ions_sheet(ions_sheet):
    # Create dictionaries for cations and anions using vectorized operations
    cation_smiles = (
        ions_sheet[ions_sheet["Ion type"] == "cation"]
        .set_index("Abbreviation")["SMILES"]
        .to_dict()
    )
    anion_smiles = (
        ions_sheet[ions_sheet["Ion type"] == "anion"]
        .set_index("Abbreviation")["SMILES"]
        .to_dict()
    )
    return cation_smiles, anion_smiles


def combine_smiles(database_sheet, cation_smiles, anion_smiles):
    """
    For each accepted ionic liquid in the database, combine the SMILES notations.
    """
    # Filter rows where 'Accepted' column is True
    accepted_liquids = database_sheet[database_sheet["Accepted dataset"] == True]
    accepted_liquids = accepted_liquids[accepted_liquids["Excluded IL"] == False]

    # Combine the SMILES notations
    accepted_liquids = accepted_liquids[
        accepted_liquids["Cation"].map(cation_smiles).notnull()
        & accepted_liquids["Anion"].map(anion_smiles).notnull()
    ]

    accepted_liquids["SMILES"] = accepted_liquids.apply(
        lambda row: str(cation_smiles[row["Cation"]])
        + "."
        + str(anion_smiles[row["Anion"]]),
        axis=1,
    )
    return accepted_liquids


def compute_descriptors(smiles):
    # Compute descriptors using vectorized operations
    mol = Chem.MolFromSmiles(smiles)
    return {desc[0]: desc[1](mol) if mol else None for desc in Descriptors._descList}


def add_descriptors(processed_df):
    # Add RDKit descriptors to each row of the DataFrame
    descriptors_df = processed_df["SMILES"].apply(compute_descriptors).apply(pd.Series)
    return pd.concat([processed_df, descriptors_df], axis=1)


def filter_dataframe(processed_df, threshold=10):
    # Simplify the DataFrame using vectorized filtering
    columns_to_drop = ["Cation", "Anion", "Excluded IL", "Accepted dataset"]
    processed_df.drop(columns=columns_to_drop, inplace=True)

    # Get boolean masks for nunique and value_counts
    nunique_mask = processed_df.nunique() > 1
    value_counts_mask = (
        processed_df.apply(lambda x: max(x.value_counts(normalize=True))) <= 0.85
    )

    # Use the boolean masks to filter columns
    processed_df = processed_df.loc[:, nunique_mask & value_counts_mask]

    # Z-score filtering for numerical columns
    for col in processed_df.select_dtypes(include="number"):
        if processed_df[col].std() > 0:  # Prevent division by zero
            z_scores = (processed_df[col] - processed_df[col].mean()) / processed_df[
                col
            ].std()
            processed_df = processed_df[(np.abs(z_scores) < threshold)]
    return processed_df


def main():
    try:
        ions_sheet, database_sheet = load_workbook_sheets(RAW_FILE_PATH)
        cation_smiles, anion_smiles = preprocess_ions_sheet(ions_sheet)
        print("Ions categories split successfully.")
        processed_data = combine_smiles(database_sheet, cation_smiles, anion_smiles)
        print("Added SMILES")
        processed_data = add_descriptors(processed_data)
        print("Descriptors successfully added.")
        processed_data = filter_dataframe(processed_data)
        print(
            f"Filtered DataFrame holds {processed_data.shape[1]} columns & {processed_data.shape[0]} rows."
        )
        processed_data.to_csv(PROCESSED_FILE_PATH, index=False)
        print("CSV successfully produced.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()

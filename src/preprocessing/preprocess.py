import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

RAW_FILE_PATH = "./data/raw.xlsx"
PROCESSED_FILE_PATH = "./data/processed.csv"


def load_workbook_sheets(file_path):
    """
    Load specified sheets from an Excel workbook.
    """
    # Load the ions sheet (S2 | Ions) and database sheet (S3 | Database)
    ions_sheet = pd.read_excel(file_path, sheet_name="S2 | Ions")
    database_sheet = pd.read_excel(file_path, sheet_name="S3 | Database")
    return ions_sheet, database_sheet


def preprocess_ions_sheet(ions_sheet):
    """
    Preprocess the ions sheet to create a mapping of ion names to SMILES notations,
    separating into cation and anion dictionaries.
    """
    # Separate the ions sheet into cations and anions based on 'Ion Type'
    cations = ions_sheet[ions_sheet["Ion type"] == "cation"]
    anions = ions_sheet[ions_sheet["Ion type"] == "anion"]

    # Create dictionaries mapping 'Abbreviation' to 'SMILES' for both cations and anions
    cation_smiles = cations.set_index("Abbreviation")["SMILES"].to_dict()
    anion_smiles = anions.set_index("Abbreviation")["SMILES"].to_dict()

    return cation_smiles, anion_smiles


def combine_smiles(database_sheet, cation_smiles, anion_smiles):
    """
    For each accepted ionic liquid in the database, combine the SMILES notations.
    """
    # Filter rows where 'Accepted' column is True
    accepted_liquids = database_sheet[database_sheet["Accepted"] == True]

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
    """
    Computes selected RDKit descriptors for a given SMILES string.
    """

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:  # Check if the SMILES string was valid
        return {descriptor: None for descriptor in Descriptors._descList}

    descriptors = {}
    for descriptor_name, descriptor_function in Descriptors._descList:
        try:
            descriptors[descriptor_name] = descriptor_function(molecule)
        except Exception as e:
            descriptors[descriptor_name] = (
                None  # Handle cases where a descriptor can't be computed
            )
    return descriptors


def add_descriptors(processed_df, smiles_column):
    """
    Aims to add RDKit descriptors to each row of accepted IL
    """
    descriptors_df = pd.DataFrame()

    # Compute descriptors for each SMILES in the DataFrame
    for index, row in processed_df.iterrows():
        smiles = row[smiles_column]
        descriptors = compute_descriptors(smiles)
        descriptors_df = descriptors_df._append(descriptors, ignore_index=True)

    enhanced_df = pd.concat([processed_df, descriptors_df], axis=1)

    return enhanced_df


def filter_dataframe(processed_df):
    """
    Simplify the DataFrame by removing columns that do not contribute to viscosity causation within the model.
    This includes dropping specific non-contributory columns, removing columns where all values are the same,
    and filtering out columns where more than 75% of the values are identical.
    """
    # Drop specific non-contributory columns
    columns_to_drop = [
        "Dataset ID",
        "IL ID",
        "Cation",
        "Anion",
        "Accepted",
        "Reference",
    ]
    processed_df.drop(columns=columns_to_drop, inplace=True)

    # Remove columns where >= 0.75 values are the same
    processed_df = processed_df.loc[
        :, processed_df.apply(lambda col: col.nunique() > 1)
    ]
    processed_df = processed_df.loc[
        :,
        processed_df.apply(
            lambda col: col.value_counts(normalize=True).iloc[0] <= 0.75
        ),
    ]

    return processed_df


def main():
    try:
        ions_sheet, database_sheet = load_workbook_sheets(RAW_FILE_PATH)
        cation_smiles, anion_smiles = preprocess_ions_sheet(ions_sheet)
        print("Ions Categories Split Successfully.")
        processed_data = combine_smiles(database_sheet, cation_smiles, anion_smiles)
        processed_data = add_descriptors(processed_data, "SMILES")
        print("Descriptors Successfully Added.")
        print(f"Adjoint DF holds {len(processed_data.columns)}")
        processed_data = filter_dataframe(processed_data)
        print(f"Filtered DF holds {len(processed_data.columns)}")
        processed_data.to_csv(
            PROCESSED_FILE_PATH, index=False
        )  # Assuming index_label is not needed or adjust accordingly
        print("CSV Successfully Produced.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

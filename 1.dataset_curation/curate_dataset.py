# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
from pathlib import Path
from IPython.display import display

# %%
# First download the data from
# s3://cellpainting-gallery/cpg0003-rosetta/broad/workspace/preprocessed_data
# and save it in the ./preprocessed_data folder

dataset_paths = {
    "LINCS-Pilot1": {
        "l1k": "./preprocessed_data/LINCS-Pilot1/L1000/replicate_level_l1k.csv.gz",
        "cp": "./preprocessed_data/LINCS-Pilot1/CellPainting/replicate_level_cp_augmented.csv.gz",
    },
    "CDRP-BBBC047-Bray": {
        "l1k": "./preprocessed_data/CDRP-BBBC047-Bray/L1000/replicate_level_l1k.csv.gz",
        "cp": "./preprocessed_data/CDRP-BBBC047-Bray/CellPainting/replicate_level_cp_augmented.csv.gz",
    },
    "TA-ORF-BBBC037-Rohban": {
        "l1k": "./preprocessed_data/TA-ORF-BBBC037-Rohban/L1000/replicate_level_l1k.csv.gz",
        "cp": "./preprocessed_data/TA-ORF-BBBC037-Rohban/CellPainting/replicate_level_cp_augmented.csv.gz",
    },
    "LUAD-BBBC041-Caicedo": {
        "l1k": "./preprocessed_data/LUAD-BBBC041-Caicedo/L1000/replicate_level_l1k.csv.gz",
        "cp": "./preprocessed_data/LUAD-BBBC041-Caicedo/CellPainting/replicate_level_cp_augmented.csv.gz",
    },
}

# Define column mappings for each dataset and data type
column_rename_mappings = {
    "CDRP-BBBC047-Bray": {
        "l1k": {
            "pert_id": "Metadata_pert_id",
            "pert_dose": "Metadata_pert_dose_micromolar",
            "det_plate": "Metadata_Plate",
            "CPD_NAME": "Metadata_pert_iname",
            "CPD_TYPE": "Metadata_cdrp_group",
            "CPD_SMILES": "Metadata_SMILES",
        },
        "cp": {
            "Metadata_broad_sample": "Metadata_pert_id",
            "Metadata_broad_sample_type": "Metadata_pert_type",
            "Metadata_mmoles_per_liter2": "Metadata_pert_dose_micromolar",
        },
    },
    "LINCS-Pilot1": {
        "l1k": {
            "pert_dose": "Metadata_pert_dose_micromolar",
            "det_plate": "Metadata_Plate",
            "cell_id": "Metadata_cell_id",
            "det_well": "Metadata_Well",
            "mfc_plate_name": "Metadata_ARP_ID",
            "pert_iname_x": "Metadata_pert_iname",
            "pert_time": "Metadata_pert_timepoint",
            "pert_mfc_id": "Metadata_pert_id",
            "pert_type_x": "Metadata_pert_type",
            "x_smiles": "Metadata_SMILES",
        },
        "cp": {
            "Metadata_broad_sample": "Metadata_pert_id",
            "Metadata_broad_sample_type": "Metadata_pert_type",
            "Metadata_mmoles_per_liter": "Metadata_pert_dose_micromolar",
            "pert_iname": "Metadata_pert_iname",
        },
    },
    "TA-ORF-BBBC037-Rohban": {
        "l1k": {
            "det_plate": "Metadata_Plate",
            "cell_id": "Metadata_cell_id",
            "det_well": "Metadata_Well",
            "mfc_plate_name": "Metadata_ARP_ID",
            "pert_time": "Metadata_pert_timepoint",
            "pert_mfc_id": "Metadata_pert_id",
            "pert_type": "Metadata_pert_type",
            "x_genesymbol_mutation": "Metadata_genesymbol_mutation",
        },
        "cp": {
            "Metadata_broad_sample": "Metadata_pert_id",
            "Metadata_broad_sample_type": "Metadata_pert_type",
            "Metadata_pert_name": "Metadata_genesymbol_mutation",
            "Metadata_gene_name": "Metadata_genesymbol",
        },
    },
    "LUAD-BBBC041-Caicedo": {
        "l1k": {
            "det_plate": "Metadata_Plate",
            "cell_id": "Metadata_cell_id",
            "det_well": "Metadata_Well",
            "mfc_plate_name": "Metadata_ARP_ID",
            "pert_time": "Metadata_pert_timepoint",
            "pert_mfc_id": "Metadata_pert_id",
            "pert_type": "Metadata_pert_type",
            "x_transcriptdb": "Metadata_transcriptdb",
        },
        "cp": {
            "Metadata_broad_sample": "Metadata_pert_id",
            "Metadata_broad_sample_type": "Metadata_pert_type",
            "x_mutation_status": "Metadata_genesymbol_mutation",
            "Symbol": "Metadata_genesymbol",
        },
    },
}

# Define the columns we want to keep for each dataset and data type
columns_to_keep = {
    "CDRP-BBBC047-Bray": {
        "l1k": [
            "Metadata_Plate",
            "Metadata_pert_id",
            "Metadata_pert_iname",
            "Metadata_pert_dose_micromolar",
            "Metadata_cdrp_group",
            "Metadata_SMILES",
        ],
        "cp": [
            "Metadata_Plate_Map_Name",
            "Metadata_Plate",
            "Metadata_Well",
            "Metadata_pert_id",
            "Metadata_pert_dose_micromolar",
            "Metadata_pert_type",
            "Metadata_cell_id",
        ],
    },
    "LINCS-Pilot1": {
        "l1k": [
            "Metadata_Plate",
            "Metadata_Well",
            "Metadata_pert_id",
            "Metadata_pert_type",
            "Metadata_pert_dose_micromolar",
            "Metadata_cell_id",
            "Metadata_pert_iname",
            "Metadata_ARP_ID",
            "Metadata_pert_timepoint",
            "Metadata_SMILES",
        ],
        "cp": [
            "Metadata_Plate_Map_Name",
            "Metadata_Plate",
            "Metadata_Well",
            "Metadata_pert_id",
            "Metadata_pert_type",
            "Metadata_pert_dose_micromolar",
            "Metadata_cell_id",
            "Metadata_pert_iname",
        ],
    },
    "TA-ORF-BBBC037-Rohban": {
        "l1k": [
            "Metadata_Plate",
            "Metadata_Well",
            "Metadata_pert_id",
            "Metadata_pert_type",
            "Metadata_cell_id",
            "Metadata_ARP_ID",
            "Metadata_pert_timepoint",
            "Metadata_genesymbol_mutation",
        ],
        "cp": [
            "Metadata_Plate_Map_Name",
            "Metadata_Plate",
            "Metadata_Well",
            "Metadata_pert_id",
            "Metadata_pert_type",
            "Metadata_cell_id",
            "Metadata_genesymbol_mutation",
            "Metadata_genesymbol",
        ],
    },
    "LUAD-BBBC041-Caicedo": {
        "l1k": [
            "Metadata_Plate",
            "Metadata_Well",
            "Metadata_pert_id",
            "Metadata_pert_type",
            "Metadata_cell_id",
            "Metadata_ARP_ID",
            "Metadata_pert_timepoint",
            "Metadata_transcriptdb",
        ],
        "cp": [
            "Metadata_Plate_Map_Name",
            "Metadata_Plate",
            "Metadata_Well",
            "Metadata_pert_id",
            "Metadata_pert_type",
            "Metadata_cell_id",
            "Metadata_genesymbol_mutation",
            "Metadata_genesymbol",
        ],
    },
}

# First load the data
dataset_data = {}
for dataset_name, paths in dataset_paths.items():
    dataset_data[dataset_name] = {}
    for data_type, dataset_path in paths.items():
        parquet_path = dataset_path.replace(".csv.gz", ".parquet")
        if not Path(parquet_path).exists():
            data = pd.read_csv(dataset_path, low_memory=False)
            data.to_parquet(parquet_path)
            dataset_data[dataset_name][data_type] = data
        else:
            data = pd.read_parquet(parquet_path)
            dataset_data[dataset_name][data_type] = data


# %%

# Then apply the column renaming
for dataset_name, data_types in dataset_data.items():
    for data_type, data in data_types.items():
        if (
            dataset_name in column_rename_mappings
            and data_type in column_rename_mappings[dataset_name]
        ):
            # First, identify feature columns we want to preserve
            if data_type == "l1k":
                feature_mask = data.columns.str.endswith("_at")
            else:  # cp
                feature_mask = (
                    data.columns.str.startswith("Cells_")
                    | data.columns.str.startswith("Cytoplasm_")
                    | data.columns.str.startswith("Nuclei_")
                )
            feature_cols = data.columns[feature_mask]
            metadata_cols = data.columns[~feature_mask]

            # Apply renaming only to metadata columns
            rename_mapping = {
                k: v
                for k, v in column_rename_mappings[dataset_name][data_type].items()
                if k in metadata_cols
            }

            # Check if new name already exists and drop it if so
            for old, new in rename_mapping.items():
                if new in data.columns and new != old:
                    data.drop(columns=[new], inplace=True)
            # Rename metadata columns
            data = data.rename(columns=rename_mapping)

            # Keep only desired metadata columns plus all feature columns
            keep_metadata = columns_to_keep[dataset_name][data_type]
            dataset_data[dataset_name][data_type] = data[
                keep_metadata + feature_cols.tolist()
            ]

# %%

# Make "Metadata_Well" uppercase
for dataset_name, data_types in dataset_data.items():
    for data_type, data in data_types.items():
        if "Metadata_Well" in data.columns:
            data["Metadata_Well"] = data["Metadata_Well"].str.upper()


# Make "Metadata_cell_id" = U2OS for CDRP-BBBC047-Bray cp
dataset_data["CDRP-BBBC047-Bray"]["l1k"]["Metadata_cell_id"] = "U2OS"

# Set timepoints
dataset_data["LINCS-Pilot1"]["cp"]["Metadata_pert_timepoint"] = 48
dataset_data["LINCS-Pilot1"]["l1k"]["Metadata_pert_timepoint"] = 24

dataset_data["CDRP-BBBC047-Bray"]["cp"]["Metadata_pert_timepoint"] = 48
dataset_data["CDRP-BBBC047-Bray"]["l1k"]["Metadata_pert_timepoint"] = 6

dataset_data["TA-ORF-BBBC037-Rohban"]["cp"]["Metadata_pert_timepoint"] = 72
dataset_data["TA-ORF-BBBC037-Rohban"]["l1k"]["Metadata_pert_timepoint"] = 72

dataset_data["LUAD-BBBC041-Caicedo"]["cp"]["Metadata_pert_timepoint"] = 96
dataset_data["LUAD-BBBC041-Caicedo"]["l1k"]["Metadata_pert_timepoint"] = 96

# %%

# Display the datasets
for dataset_name, data_types in dataset_data.items():
    for data_type, data in data_types.items():
        display(f"Dataset: {dataset_name}, Data Type: {data_type}")
        display(data.sample(5)[data.columns[data.columns.str.startswith("Metadata")]])

# %%

# %%
for dataset_name, data_types in dataset_data.items():
    for data_type, data in data_types.items():
        if "Metadata_pert_type" in data.columns:
            data["Metadata_pert_type"] = data["Metadata_pert_type"].replace(
                {"ctl_vehicle": "control", "trt_cp": "trt"}
            )


#  TA-ORF-BBBC037-Rohban cp does not correctly identify Metadata_pert_type, because it marks all as trt.

# %%

# Print columns for each dataset and data type
print("\nColumns in each dataset:")
for dataset_name, data_types in dataset_data.items():
    print(f"\n{dataset_name}:")
    for data_type, data in data_types.items():
        metadata_cols = [col for col in data.columns if col.startswith("Metadata")]
        print(f"  {data_type}: {sorted(metadata_cols)}")

# Find common columns between l1k datasets
l1k_common = set.intersection(
    *[set(data_types["l1k"].columns) for data_types in dataset_data.values()]
)
l1k_metadata_common = sorted([col for col in l1k_common if col.startswith("Metadata")])

# Find common columns between cp datasets
cp_common = set.intersection(
    *[set(data_types["cp"].columns) for data_types in dataset_data.values()]
)
cp_metadata_common = sorted([col for col in cp_common if col.startswith("Metadata")])

# Find common columns across all datasets
all_common = set.intersection(l1k_common, cp_common)
all_metadata_common = sorted([col for col in all_common if col.startswith("Metadata")])

print("\nCommon Metadata columns across L1K datasets:")
print(l1k_metadata_common)
print("\nCommon Metadata columns across CP datasets:")
print(cp_metadata_common)
print("\nCommon Metadata columns across ALL datasets:")
print(all_metadata_common)

# %%

# Check for duplicate columns within each dataset
for dataset_name, data_types in dataset_data.items():
    for data_type, data in data_types.items():
        duplicate_cols = data.columns.duplicated()
        if any(duplicate_cols):
            print(f"Duplicate columns found in {dataset_name} {data_type}:")
            print(data.columns[duplicate_cols])

# %%
# Create markdown output for datasets
markdown_output = "# Dataset Samples\n\n"

for dataset_name, data_types in dataset_data.items():
    markdown_output += f"## {dataset_name}\n\n"
    for data_type, data in data_types.items():
        markdown_output += f"### {data_type.upper()} Data\n\n"
        # Convert sample to markdown table
        sample_df = data.sample(50)[
            [col for col in data.columns if col.startswith("Metadata")]
        ]
        markdown_output += sample_df.to_markdown(index=False) + "\n\n"
        display(sample_df.head())

# Write to file
with open("dataset_samples.md", "w") as f:
    f.write(markdown_output)

print("Dataset samples have been written to dataset_samples.md")

# %%

# Save processed datasets using same structure as input
for dataset_name, data_types in dataset_data.items():
    for data_type, data in data_types.items():
        # Mirror the input path structure but with processed data
        input_path = Path(dataset_paths[dataset_name][data_type])
        output_path = (
            Path("curated")
            / input_path.parent
            / input_path.name.replace(".csv.gz", ".parquet")
        )
        # Create the processed subdirectory if it doesn't exist
        output_path.parent.mkdir(exist_ok=True, parents=True)

        # # Save the data
        data.to_parquet(output_path, index=False)
        print(f"Saved {dataset_name} {data_type} data to {output_path}")

# %%

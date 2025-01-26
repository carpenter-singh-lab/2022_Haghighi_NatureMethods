# Dataset Curation

## Structure

Available at:
`s3://cellpainting-gallery/cpg0003-rosetta/broad/workspace/curated_ preprocessed_data`

```
curated_preprocessed_data
├── CDRP-BBBC047-Bray
│   ├── CellPainting
│   │   └── replicate_level_cp_augmented.parquet
│   └── L1000
│       └── replicate_level_l1k.parquet
├── LINCS-Pilot1
│   ├── CellPainting
│   │   └── replicate_level_cp_augmented.parquet
│   └── L1000
│       └── replicate_level_l1k.parquet
├── LUAD-BBBC041-Caicedo
│   ├── CellPainting
│   │   └── replicate_level_cp_augmented.parquet
│   └── L1000
│       └── replicate_level_l1k.parquet
└── TA-ORF-BBBC037-Rohban
    ├── CellPainting
    │   └── replicate_level_cp_augmented.parquet
    └── L1000
        └── replicate_level_l1k.parquet
```

## Curated columns

- `Metadata_Plate` [All]: Identifier of the multi‐well plate (e.g., SQ00015156, PAC053_U2OS_6H_X2_B1_UNI4445R, TA.OE005_U2OS_72H_X1_B15).
- `Metadata_Plate_Map_Name` [All CP]: Plate‐map identifier (e.g., C-7161-01-LM6-003).
- `Metadata_ARP_ID` [LINCS-Pilot1 L1K, TA-ORF-BBBC037-Rohban L1K, LUAD-BBBC041-Caicedo L1K]: Internal plate identifier (e.g., AB00016187).
- `Metadata_Well` [All except CDRP-BBBC047-Bray L1K]: Specific well position within the plate (e.g., A01, H11).
- `Metadata_pert_id` [All]: Unique perturbation identifier (e.g., BRD-K50691590-001-02-2, TRCN0000471252, EMPTY).
- `Metadata_pert_type` [All except CDRP-BBBC047-Bray L1K]: Perturbation type (e.g., trt_cp, ctl_vehicle, trt, control).
- `Metadata_cell_id` [All]: Cell line used (e.g., A549, U2OS).
- `Metadata_pert_timepoint` [All]: Time (in hours) from perturbation to measurement (e.g., 24, 48, 72, 96).
- `Metadata_pert_dose_micromolar` [LINCS-Pilot1, CDRP-BBBC047-Bray]: Final compound concentration (µM) (e.g., 0.0411523, 10).
- `Metadata_pert_iname` [LINCS-Pilot1, CDRP-BBBC047-Bray]: Common name of the compound or control (e.g., bortezomib, DMSO).
- `Metadata_SMILES` [LINCS-Pilot1 L1K, CDRP-BBBC047-Bray L1K]: SMILES string for the compound structure.
- `Metadata_cdrp_group` [CDRP-BBBC047-Bray L1K]: Subset/group label in the CDRP compound library (e.g., DOS, BIO).
- `Metadata_genesymbol_mutation` [TA-ORF-BBBC037-Rohban L1K+CP, LUAD-BBBC041-Caicedo CP]: Gene plus mutation notation (e.g., TP53_p.R248Q).
- `Metadata_genesymbol` [TA-ORF-BBBC037-Rohban CP, LUAD-BBBC041-Caicedo CP]: Gene symbol alone (e.g., TP53, MAPK8).
- `Metadata_transcriptdb` [LUAD-BBBC041-Caicedo L1K]: Reference to specific transcript/isoform (e.g., NM_001126112.2:c.796G>C).


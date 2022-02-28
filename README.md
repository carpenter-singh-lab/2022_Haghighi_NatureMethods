<!-- #### 2021_Haghighi_NeurIPS_Dataset_submitted -->
# High-Dimensional Gene Expression and Morphology Profiles of Cells across 28,000 Genetic and Chemical Perturbations
Populations of cells can be perturbed by various chemical and genetic treatments and the impact on the cells’ gene expression (transcription, i.e. mRNA levels) and morphology (in an image-based assay) can be measured in high dimensions. The patterns observed in this data can be used for more than a dozen applications in drug discovery and basic biology research. We provide a collection of four datasets where both gene expression and morphological data are available; roughly a thousand features are measured for each data type, across more than 28,000 thousand chemical and genetic perturbations. We have defined a set of biological problems that can be investigated using these two data modalities and provided baseline analysis and evaluation metrics for addressing each. This data resource is available at s3 bucket s3://cellpainting-datasets/Rosetta-GE-CP.

 [Link to Paper](https://www.biorxiv.org/content/10.1101/2021.09.08.459417v1)


# Data Modalities:
<details>
<summary>Click to expand</summary>
  
### Gene expression (GE) profiles
Each cell has DNA in the nucleus which is transcribed into various mRNA molecules which are then translated into proteins that carry out functions in the cell. The levels of mRNA in the cell are often biologically meaningful - collectively, mRNA levels for a cell are known as its transcriptional state; each individual mRNA level is referred to as the corresponding gene's "expression".
The L1000 assay \cite{subramanian2017next} was used to measure the transcriptional state of cells in the datasets here. The assay reports a sample's mRNA levels for 978 genes at high-throughput, from the bulk population of cells treated with a given perturbation. These 978 "landmark" genes capture approximately 80\% of the transcriptional variance for the entire genome \cite{subramanian2017next}. The data processing tools and workflows to produce these profiles are available at https://clue.io/.


### Cell Painting morphological (CP) profiles
We used the Cell Painting assay \cite{bray2016cell} to measure the morphological state of cells treated with a given perturbation. The assay captures fluorescence images of cells colored by six well-characterized fluorescent dyes to stain the nucleus, nucleoli, cytoplasmic RNA, endoplasmic reticulum, actin cytoskeleton, Golgi apparatus and plasma membrane. These eight labeled cell compartments are captured through five channels of high-resolution microscopy images (_DNA, RNA, ER, AGP_, and _Mito_). 
Images are then processed using [CellProfiler software](https://cellprofiler.org/) \cite{mcquin2018cellprofiler} to extract thousands of features of each cell’s morphology and form a high-dimensional profile for each single cell.  These features are based on various shape, intensity and texture statistics and are then aggregated for all the single cells in a "well" (a miniature test tube) that are called replicate-level profiles of perturbations. 
Aggregation of replicate-level profiles across all the wells or replicates of a perturbation is called a treatment-level profile. In our study, we used treatment-level profiles in all experiments but have provided replicate-level profiles for researchers interested in further data exploration. 

</details>
  
# Datasets:

- We have gathered the following five available data sets that had both Cell Painting morphological (CP) and L1000 gene expression (GE) profiles, preprocessed the data from different sources and in different formats in a unified .csv format.

    - CDRP-BBBC047-Bray-CP-GE (Cell line: U2OS)
    - CDRPBIO-BBBC036-Bray-CP-GE (Cell line: U2OS)
    - LUAD-BBBC041-Caicedo-CP-GE (Cell line: A549)
    - TA-ORF-BBBC037-Rohban-CP-GE (Cell line: U2OS)
    - LINCS-Pilot1-CP-GE (Cell line: A549)

## References to raw profiles and images:
<details>
<summary>Click to expand</summary>
  
- CDRP-BBBC047-Bray-[CP](https://pubmed.ncbi.nlm.nih.gov/28327978/) - [GE](https://pubmed.ncbi.nlm.nih.gov/29195078/)
- CDRP-bio-BBBC036-Bray-[CP](https://pubmed.ncbi.nlm.nih.gov/28327978/) - [GE](https://pubmed.ncbi.nlm.nih.gov/29195078/)
- LUAD-BBBC041-Caicedo-[CP](https://registry.opendata.aws/cell-painting-image-collection/) - [GE](https://pubmed.ncbi.nlm.nih.gov/27478040/)
- TA-ORF-BBBC037-Rohban-[CP](https://elifesciences.org/articles/24060) - [GE](https://github.com/carpenterlab/2017_rohban_elife/tree/master/input/TA-OE-L1000-B1)
- LINCS-Pilot1-[CP](https://zenodo.org/record/3928744#.YNu3WzZKheV) - [GE](https://figshare.com/articles/dataset/L1000_data_for_profiling_comparison/13181966)
  
</details>


## Preprocessed publicly available Profiles
Preprocessed profiles (~9.5GB) are available on a S3 bucket. They can be downloaded at no cost and no need for registration of any sort, using the command:

```bash
aws s3 sync \
  --no-sign-request \
  s3://cellpainting-gallery/rosetta/broad/workspace/preprocessed_data .
```

See this [wiki](https://github.com/carpenterlab/2016_bray_natprot/wiki/What-do-Cell-Painting-features-mean%3F) for sample Cell Painting images and the meaning of ([CellProfiler](https://cellprofiler.org/)-derived) Cell Painting features. 

- AWS CLI installation instructions can be found [here](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

#### Data version

The [Etags](https://docs.aws.amazon.com/AmazonS3/latest/API/API_Object.html) of these files are listed below

<details>
 <summary>Etag information - Click to expand</summary>
 
```sh
aws s3api list-objects --bucket cellpainting-gallery --prefix rosetta/broad/workspace/preprocessed_data/
```

```json
{
    "Contents": [
        {
            "Key": "rosetta/broad/workspace/preprocessed_data/CDRP-BBBC047-Bray/CellPainting/replicate_level_cp_augmented.csv.gz",
            "LastModified": "2022-02-25T20:24:06.000Z",
            "ETag": "\"8367b77b245035279d21e083fb57564e-261\"",
            "Size": 2183033139,
            "StorageClass": "STANDARD",
            "Owner": {
                "DisplayName": "cellpainting",
                "ID": "b2ff2dec476b541160cb5edae0ba12ffb6f3cd979ce9352e9ca765d92ac2170c"
            }
        },
        {
            "Key": "rosetta/broad/workspace/preprocessed_data/CDRP-BBBC047-Bray/CellPainting/replicate_level_cp_normalized.csv.gz",
            "LastModified": "2022-02-25T20:24:06.000Z",
            "ETag": "\"572869293e0cfacdd8882c2b758fac00-272\"",
            "Size": 2277911750,
            "StorageClass": "STANDARD",
            "Owner": {
                "DisplayName": "cellpainting",
                "ID": "b2ff2dec476b541160cb5edae0ba12ffb6f3cd979ce9352e9ca765d92ac2170c"
            }
        },
        {
            "Key": "rosetta/broad/workspace/preprocessed_data/CDRP-BBBC047-Bray/CellPainting/replicate_level_cp_normalized_variable_selected.csv.gz",
            "LastModified": "2022-02-25T20:24:06.000Z",
            "ETag": "\"510f9c5a93436c8af2f36f0308c78be0-131\"",
            "Size": 1098352960,
            "StorageClass": "STANDARD",
            "Owner": {
                "DisplayName": "cellpainting",
                "ID": "b2ff2dec476b541160cb5edae0ba12ffb6f3cd979ce9352e9ca765d92ac2170c"
            }
        },
        {
            "Key": "rosetta/broad/workspace/preprocessed_data/CDRP-BBBC047-Bray/L1000/replicate_level_l1k.csv.gz",
            "LastModified": "2022-02-25T20:24:06.000Z",
            "ETag": "\"40e1f7285238c5381b9d9fdeebb5a026-32\"",
            "Size": 262406281,
            "StorageClass": "STANDARD",
            "Owner": {
                "DisplayName": "cellpainting",
                "ID": "b2ff2dec476b541160cb5edae0ba12ffb6f3cd979ce9352e9ca765d92ac2170c"
            }
        },
        {
            "Key": "rosetta/broad/workspace/preprocessed_data/CDRP-BBBC047-Bray/L1000/replicate_level_l1k_pclfc.csv.gz",
            "LastModified": "2022-02-25T20:24:06.000Z",
            "ETag": "\"630b98d69d185f530acfb0c272e82031-31\"",
            "Size": 258651159,
            "StorageClass": "STANDARD",
            "Owner": {
                "DisplayName": "cellpainting",
                "ID": "b2ff2dec476b541160cb5edae0ba12ffb6f3cd979ce9352e9ca765d92ac2170c"
            }
        },
        {
            "Key": "rosetta/broad/workspace/preprocessed_data/CDRP-BBBC047-Bray/L1000/replicate_level_l1k_pczscore.csv.gz",
            "LastModified": "2022-02-25T20:24:13.000Z",
            "ETag": "\"5ad1f4b412c8ea9b9abb55a254a7ebbe-72\"",
            "Size": 603440498,
            "StorageClass": "STANDARD",
            "Owner": {
                "DisplayName": "cellpainting",
                "ID": "b2ff2dec476b541160cb5edae0ba12ffb6f3cd979ce9352e9ca765d92ac2170c"
            }
        },
        {
            "Key": "rosetta/broad/workspace/preprocessed_data/CDRP-BBBC047-Bray/L1000/replicate_level_l1k_vczscore.csv.gz",
            "LastModified": "2022-02-25T20:24:13.000Z",
            "ETag": "\"b58b4d31e96964f28165f048bdfd60c8-73\"",
            "Size": 605293966,
            "StorageClass": "STANDARD",
            "Owner": {
                "DisplayName": "cellpainting",
                "ID": "b2ff2dec476b541160cb5edae0ba12ffb6f3cd979ce9352e9ca765d92ac2170c"
            }
        },
        {
            "Key": "rosetta/broad/workspace/preprocessed_data/CDRP-BBBC047-Bray/L1000/treatment_level_l1k.csv.gz",
            "LastModified": "2022-02-25T20:24:27.000Z",
            "ETag": "\"e695e3d5f520553f516516ab8719719f-13\"",
            "Size": 107934871,
            "StorageClass": "STANDARD",
            "Owner": {
                "DisplayName": "cellpainting",
                "ID": "b2ff2dec476b541160cb5edae0ba12ffb6f3cd979ce9352e9ca765d92ac2170c"
            }
        },
        {
            "Key": "rosetta/broad/workspace/preprocessed_data/CDRPBIO-BBBC036-Bray/CellPainting/replicate_level_cp_augmented.csv.gz",
            "LastModified": "2022-02-25T20:24:27.000Z",
            "ETag": "\"3e199aeba5209250e0d2c5948f5bd522-36\"",
            "Size": 298941736,
            "StorageClass": "STANDARD",
            "Owner": {
                "DisplayName": "cellpainting",
                "ID": "b2ff2dec476b541160cb5edae0ba12ffb6f3cd979ce9352e9ca765d92ac2170c"
            }
        },
        {
            "Key": "rosetta/broad/workspace/preprocessed_data/CDRPBIO-BBBC036-Bray/CellPainting/replicate_level_cp_normalized.csv.gz",
            "LastModified": "2022-02-25T20:24:30.000Z",
            "ETag": "\"0b86065f8840aff626d64c6f52a8caf4-38\"",
            "Size": 311539701,
            "StorageClass": "STANDARD",
            "Owner": {
                "DisplayName": "cellpainting",
                "ID": "b2ff2dec476b541160cb5edae0ba12ffb6f3cd979ce9352e9ca765d92ac2170c"
            }
        },
        {
            "Key": "rosetta/broad/workspace/preprocessed_data/CDRPBIO-BBBC036-Bray/CellPainting/replicate_level_cp_normalized_variable_selected.csv.gz",
            "LastModified": "2022-02-25T20:24:32.000Z",
            "ETag": "\"bffd9db9578fcc70bbd7d72e0dfff773-14\"",
            "Size": 117242590,
            "StorageClass": "STANDARD",
            "Owner": {
                "DisplayName": "cellpainting",
                "ID": "b2ff2dec476b541160cb5edae0ba12ffb6f3cd979ce9352e9ca765d92ac2170c"
            }
        },
        {
            "Key": "rosetta/broad/workspace/preprocessed_data/CDRPBIO-BBBC036-Bray/L1000/replicate_level_l1k.csv.gz",
            "LastModified": "2022-02-25T20:24:35.000Z",
            "ETag": "\"5b45e5cb94f0466a2abb11fbac8a655e-4\"",
            "Size": 26842289,
            "StorageClass": "STANDARD",
            "Owner": {
                "DisplayName": "cellpainting",
                "ID": "b2ff2dec476b541160cb5edae0ba12ffb6f3cd979ce9352e9ca765d92ac2170c"
            }
        },
        {
            "Key": "rosetta/broad/workspace/preprocessed_data/LINCS-Pilot1/CellPainting/replicate_level_cp_augmented.csv.gz",
            "LastModified": "2022-02-25T20:24:35.000Z",
            "ETag": "\"9bde4d7112c06ffa1849fbfa4efa22f1-36\"",
            "Size": 296762474,
            "StorageClass": "STANDARD",
            "Owner": {
                "DisplayName": "cellpainting",
                "ID": "b2ff2dec476b541160cb5edae0ba12ffb6f3cd979ce9352e9ca765d92ac2170c"
            }
        },
        {
            "Key": "rosetta/broad/workspace/preprocessed_data/LINCS-Pilot1/CellPainting/replicate_level_cp_normalized.csv.gz",
            "LastModified": "2022-02-25T20:24:36.000Z",
            "ETag": "\"f42af6b4109ef9ed110004def49f6c2c-36\"",
            "Size": 299683743,
            "StorageClass": "STANDARD",
            "Owner": {
                "DisplayName": "cellpainting",
                "ID": "b2ff2dec476b541160cb5edae0ba12ffb6f3cd979ce9352e9ca765d92ac2170c"
            }
        },
        {
            "Key": "rosetta/broad/workspace/preprocessed_data/LINCS-Pilot1/CellPainting/replicate_level_cp_normalized_variable_selected.csv.gz",
            "LastModified": "2022-02-25T20:24:38.000Z",
            "ETag": "\"33783625dc59b0de2bf16c299f5380dd-12\"",
            "Size": 94527797,
            "StorageClass": "STANDARD",
            "Owner": {
                "DisplayName": "cellpainting",
                "ID": "b2ff2dec476b541160cb5edae0ba12ffb6f3cd979ce9352e9ca765d92ac2170c"
            }
        },
        {
            "Key": "rosetta/broad/workspace/preprocessed_data/LINCS-Pilot1/L1000/level_3.csv.gz",
            "LastModified": "2022-02-25T20:24:41.000Z",
            "ETag": "\"8491fe32e9b0b040f10c7d51225d6111-11\"",
            "Size": 89725093,
            "StorageClass": "STANDARD",
            "Owner": {
                "DisplayName": "cellpainting",
                "ID": "b2ff2dec476b541160cb5edae0ba12ffb6f3cd979ce9352e9ca765d92ac2170c"
            }
        },
        {
            "Key": "rosetta/broad/workspace/preprocessed_data/LINCS-Pilot1/L1000/level_4.csv.gz",
            "LastModified": "2022-02-25T20:24:42.000Z",
            "ETag": "\"14679d4b4cae5e12a4e7be8255bd22ff-10\"",
            "Size": 78596325,
            "StorageClass": "STANDARD",
            "Owner": {
                "DisplayName": "cellpainting",
                "ID": "b2ff2dec476b541160cb5edae0ba12ffb6f3cd979ce9352e9ca765d92ac2170c"
            }
        },
        {
            "Key": "rosetta/broad/workspace/preprocessed_data/LINCS-Pilot1/L1000/level_4W.csv.gz",
            "LastModified": "2022-02-25T20:24:43.000Z",
            "ETag": "\"370607c1f148942263037a7e26018303-17\"",
            "Size": 140912507,
            "StorageClass": "STANDARD",
            "Owner": {
                "DisplayName": "cellpainting",
                "ID": "b2ff2dec476b541160cb5edae0ba12ffb6f3cd979ce9352e9ca765d92ac2170c"
            }
        },
        {
            "Key": "rosetta/broad/workspace/preprocessed_data/LINCS-Pilot1/L1000/level_5_modz.csv.gz",
            "LastModified": "2022-02-25T20:24:43.000Z",
            "ETag": "\"5967bd8a92d2c57242436330950f1cd2\"",
            "Size": 3631,
            "StorageClass": "STANDARD",
            "Owner": {
                "DisplayName": "cellpainting",
                "ID": "b2ff2dec476b541160cb5edae0ba12ffb6f3cd979ce9352e9ca765d92ac2170c"
            }
        },
        {
            "Key": "rosetta/broad/workspace/preprocessed_data/LINCS-Pilot1/L1000/level_5_rank.csv.gz",
            "LastModified": "2022-02-25T20:24:43.000Z",
            "ETag": "\"83c8146ea2f8a2a6392643b3c4472727\"",
            "Size": 3631,
            "StorageClass": "STANDARD",
            "Owner": {
                "DisplayName": "cellpainting",
                "ID": "b2ff2dec476b541160cb5edae0ba12ffb6f3cd979ce9352e9ca765d92ac2170c"
            }
        },
        {
            "Key": "rosetta/broad/workspace/preprocessed_data/LINCS-Pilot1/L1000/replicate_level_l1k.csv.gz",
            "LastModified": "2022-02-25T20:24:44.000Z",
            "ETag": "\"872c318560ba21c9d36e805fb97992a4-10\"",
            "Size": 78596337,
            "StorageClass": "STANDARD",
            "Owner": {
                "DisplayName": "cellpainting",
                "ID": "b2ff2dec476b541160cb5edae0ba12ffb6f3cd979ce9352e9ca765d92ac2170c"
            }
        },
        {
            "Key": "rosetta/broad/workspace/preprocessed_data/LUAD-BBBC041-Caicedo/CellPainting/replicate_level_cp_augmented.csv.gz",
            "LastModified": "2022-02-25T20:24:44.000Z",
            "ETag": "\"11a0a26d299f09452455e0c7e44c571c-11\"",
            "Size": 85105940,
            "StorageClass": "STANDARD",
            "Owner": {
                "DisplayName": "cellpainting",
                "ID": "b2ff2dec476b541160cb5edae0ba12ffb6f3cd979ce9352e9ca765d92ac2170c"
            }
        },
        {
            "Key": "rosetta/broad/workspace/preprocessed_data/LUAD-BBBC041-Caicedo/CellPainting/replicate_level_cp_normalized.csv.gz",
            "LastModified": "2022-02-25T20:24:46.000Z",
            "ETag": "\"f91d40a978c96834973f24b96b8a3b02-11\"",
            "Size": 88273100,
            "StorageClass": "STANDARD",
            "Owner": {
                "DisplayName": "cellpainting",
                "ID": "b2ff2dec476b541160cb5edae0ba12ffb6f3cd979ce9352e9ca765d92ac2170c"
            }
        },
        {
            "Key": "rosetta/broad/workspace/preprocessed_data/LUAD-BBBC041-Caicedo/CellPainting/replicate_level_cp_normalized_variable_selected.csv.gz",
            "LastModified": "2022-02-25T20:24:47.000Z",
            "ETag": "\"1ba6936ab1188268850a798e30c4823f-2\"",
            "Size": 16570136,
            "StorageClass": "STANDARD",
            "Owner": {
                "DisplayName": "cellpainting",
                "ID": "b2ff2dec476b541160cb5edae0ba12ffb6f3cd979ce9352e9ca765d92ac2170c"
            }
        },
        {
            "Key": "rosetta/broad/workspace/preprocessed_data/LUAD-BBBC041-Caicedo/L1000/replicate_level_l1k.csv.gz",
            "LastModified": "2022-02-25T20:24:47.000Z",
            "ETag": "\"c1b8cabef1934d213baf797b80c4c32c-2\"",
            "Size": 11448027,
            "StorageClass": "STANDARD",
            "Owner": {
                "DisplayName": "cellpainting",
                "ID": "b2ff2dec476b541160cb5edae0ba12ffb6f3cd979ce9352e9ca765d92ac2170c"
            }
        },
        {
            "Key": "rosetta/broad/workspace/preprocessed_data/LUAD-BBBC041-Caicedo/L1000/replicate_level_l1k_Juan.csv.gz",
            "LastModified": "2022-02-25T20:24:47.000Z",
            "ETag": "\"587d00f75c5fa6164929e3592bf96080-4\"",
            "Size": 25582111,
            "StorageClass": "STANDARD",
            "Owner": {
                "DisplayName": "cellpainting",
                "ID": "b2ff2dec476b541160cb5edae0ba12ffb6f3cd979ce9352e9ca765d92ac2170c"
            }
        },
        {
            "Key": "rosetta/broad/workspace/preprocessed_data/LUAD-BBBC041-Caicedo/L1000/treatment_level_l1k.csv.gz",
            "LastModified": "2022-02-25T20:24:48.000Z",
            "ETag": "\"c7f285af2a39efc64a4c8d57854d6a0e\"",
            "Size": 4575373,
            "StorageClass": "STANDARD",
            "Owner": {
                "DisplayName": "cellpainting",
                "ID": "b2ff2dec476b541160cb5edae0ba12ffb6f3cd979ce9352e9ca765d92ac2170c"
            }
        },
        {
            "Key": "rosetta/broad/workspace/preprocessed_data/TA-ORF-BBBC037-Rohban/CellPainting/replicate_level_cp_augmented.csv.gz",
            "LastModified": "2022-02-25T20:24:48.000Z",
            "ETag": "\"9707bd02924cda850ed6f1e7eba33d9a-4\"",
            "Size": 27548449,
            "StorageClass": "STANDARD",
            "Owner": {
                "DisplayName": "cellpainting",
                "ID": "b2ff2dec476b541160cb5edae0ba12ffb6f3cd979ce9352e9ca765d92ac2170c"
            }
        },
        {
            "Key": "rosetta/broad/workspace/preprocessed_data/TA-ORF-BBBC037-Rohban/CellPainting/replicate_level_cp_normalized.csv.gz",
            "LastModified": "2022-02-25T20:24:48.000Z",
            "ETag": "\"736ef2b85bf5406f27239153f3772218-4\"",
            "Size": 27482072,
            "StorageClass": "STANDARD",
            "Owner": {
                "DisplayName": "cellpainting",
                "ID": "b2ff2dec476b541160cb5edae0ba12ffb6f3cd979ce9352e9ca765d92ac2170c"
            }
        },
        {
            "Key": "rosetta/broad/workspace/preprocessed_data/TA-ORF-BBBC037-Rohban/CellPainting/replicate_level_cp_normalized_variable_selected.csv.gz",
            "LastModified": "2022-02-25T20:24:48.000Z",
            "ETag": "\"1315c2fd175b265d10e929e51d9dfef0\"",
            "Size": 1106334,
            "StorageClass": "STANDARD",
            "Owner": {
                "DisplayName": "cellpainting",
                "ID": "b2ff2dec476b541160cb5edae0ba12ffb6f3cd979ce9352e9ca765d92ac2170c"
            }
        },
        {
            "Key": "rosetta/broad/workspace/preprocessed_data/TA-ORF-BBBC037-Rohban/L1000/replicate_level_l1k.csv.gz",
            "LastModified": "2022-02-25T20:24:49.000Z",
            "ETag": "\"1e643bb1182555a8e7699230a0ea98d1\"",
            "Size": 2022367,
            "StorageClass": "STANDARD",
            "Owner": {
                "DisplayName": "cellpainting",
                "ID": "b2ff2dec476b541160cb5edae0ba12ffb6f3cd979ce9352e9ca765d92ac2170c"
            }
        },
        {
            "Key": "rosetta/broad/workspace/preprocessed_data/TA-ORF-BBBC037-Rohban/L1000/replicate_level_l1k_QNORM.csv.gz",
            "LastModified": "2022-02-25T20:24:49.000Z",
            "ETag": "\"8ffb9c82772442cbbd138a6ab05a9a97\"",
            "Size": 1782302,
            "StorageClass": "STANDARD",
            "Owner": {
                "DisplayName": "cellpainting",
                "ID": "b2ff2dec476b541160cb5edae0ba12ffb6f3cd979ce9352e9ca765d92ac2170c"
            }
        },
        {
            "Key": "rosetta/broad/workspace/preprocessed_data/TA-ORF-BBBC037-Rohban/L1000/replicate_level_l1k_ZSPCQNORM.csv.gz",
            "LastModified": "2022-02-25T20:24:49.000Z",
            "ETag": "\"36783d73bb48bec466aeda707384c7e5\"",
            "Size": 1997953,
            "StorageClass": "STANDARD",
            "Owner": {
                "DisplayName": "cellpainting",
                "ID": "b2ff2dec476b541160cb5edae0ba12ffb6f3cd979ce9352e9ca765d92ac2170c"
            }
        }
    ]
}
```

 </details>
 

#### CP-L1000 Profile descriptions:
We gathered four available data sets that had both Cell Painting morphological (CP) and L1000 gene expression (GE) profiles, preprocessed the data from different sources and in different formats in a unified .csv format, and made the data publicly available. Single cell morphological (CP) profiles were created using CellProfiler software and processed to form aggregated replicate and treatment levels using the R cytominer package [cytominer](https://github.com/cytomining/cytominer/blob/master/vignettes/cytominer-pipeline.Rmd). 
We made the following three types of profiles available for cell-painting modality of each of four datasets:


| Folder  | File name                                                  | Description                                              |
| -------     | ---------------------------------------------------------- | -------------------------------------------------------- |
|CellPainting| `replicate_level_cp_augmented.csv`                                 | Aggregated and Metadata annotated profiles which are the average of single cell profiles in each well.              |
|CellPainting| `replicate_level_cp_normalized.csv.gz`                             | Normalized profiles which are the z-scored aggregated profiles, where the scores are computing using the distribution of negative controls as the reference.                  |
|CellPainting| `replicate_level_cp_normalized_variable_selected.csv.gz`        | Normalized variable selected which are normalized profiles with features selection applied      |
|L1000| `replicate_level_l1k.csv`                                 | Aggregated and Metadata annotated profiles which are the average of single cell profiles in each well.      





### Available functional annotation for each dataset:

| Dataset  | Functional Annotations                                                | Corresponding Metadata Column                                              |
| -------  | ---------------------------------------------------------- | -------------------------------------------------------- |
| CDRP |               MoA                  |      `Metadata_moa`,`Metadata_target`             | 
|CDRP-BIO|             MoA                  |     `Metadata_moa`,`Metadata_target`              |
|LUAD|                   |     |
|TA-ORF|                   |     |
|LINCS|   MoA    | `Metadata_moa` |      


### Number of features for each dataset:

| Dataset  | GE                                                | CP<br/>`normalized`       | CP<br/>`normalized_variable_selected`  |
| -------  | ------------------------------------------------- | ------------------------- | -------------------------------------- |
| CDRP     |               977                                 |      x                    |                                        |
|CDRP-BIO  |               977                                 |      1570                 |              601                       |
|LUAD      |               978                                 |      1569                 |              291                       |
|TA-ORF    |               978                                 |      1677                 |               63                       |
|LINCS     |               978                                 |      1670                 |               119                      | 


<!-- # Running the analysis script notebooks -->



# Lookup table for L1000 genes predictability:
<details>
<summary>Click to expand</summary>
  
[Table](https://github.com/carpenterlab/2021_Haghighi_submitted/blob/main/results/SingleGenePred/Appendix_D.csv)

</details>


# License
We license the data, results, and figures as [CC0 1.0](LICENSE_CC0.md) and the source code as BSD 3-Clause.

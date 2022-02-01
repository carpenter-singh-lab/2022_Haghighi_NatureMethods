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
- TA-ORF-BBBC037-Rohban-[CP](https://elifesciences.org/articles/24060) - [GE]
- LINCS-Pilot1-[CP](https://zenodo.org/record/3928744#.YNu3WzZKheV) - [GE](https://clue.io/)
  
</details>


## Preprocessed publicly available Profiles
Preprocessed profiles (~9.5GB) are available on a S3 bucket. They can be downloaded at no cost and no need for registration of any sort, using the command:

```bash
aws s3 cp \
  --recursive \
  s3://cellpainting-datasets/Rosetta-GE-CP .  
```

See this [wiki](https://github.com/carpenterlab/2016_bray_natprot/wiki/What-do-Cell-Painting-features-mean%3F) for sample Cell Painting images and the meaning of ([CellProfiler](https://cellprofiler.org/)-derived) Cell Painting features. 

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

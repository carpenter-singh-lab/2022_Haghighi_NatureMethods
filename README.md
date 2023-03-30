# High-Dimensional Gene Expression and Morphology Profiles of Cells across 28,000 Genetic and Chemical Perturbations
Populations of cells can be perturbed by various chemical and genetic treatments and the impact on the cells’ gene expression (transcription, i.e. mRNA levels) and morphology (in an image-based assay) can be measured in high dimensions.
The patterns observed in this data can be used for more than a dozen applications in drug discovery and basic biology research.
 We provide a collection of four datasets where both gene expression and morphological data are available; roughly a thousand features are measured for each data type, across more than 28,000 thousand chemical and genetic perturbations.
 We have defined a set of biological problems that can be investigated using these two data modalities and provided baseline analysis and evaluation metrics for addressing each.

 [Link to Paper](https://www.nature.com/articles/s41592-022-01667-0)


# Data Modalities
<details>
<summary>Click to expand</summary>

### Gene expression (GE) profiles
Each cell has DNA in the nucleus which is transcribed into various mRNA molecules which are then translated into proteins that carry out functions in the cell.
The levels of mRNA in the cell are often biologically meaningful - collectively, mRNA levels for a cell are known as its transcriptional state; each individual mRNA level is referred to as the corresponding gene's "expression".
The L1000 assay was used to measure the transcriptional state of cells in the datasets here.
The assay reports a sample's mRNA levels for 978 genes at high-throughput, from the bulk population of cells treated with a given perturbation.
These 978 "landmark" genes capture approximately $80\%$ of the transcriptional variance for the entire genome.
The data processing tools and workflows to produce these profiles are available at https://clue.io/.


### Cell Painting morphological (CP) profiles
We used the Cell Painting assay to measure the morphological state of cells treated with a given perturbation.
The assay captures fluorescence images of cells colored by six well-characterized fluorescent dyes to stain the nucleus, nucleoli, cytoplasmic RNA, endoplasmic reticulum, actin cytoskeleton, Golgi apparatus and plasma membrane.
These eight labeled cell compartments are captured through five channels of high-resolution microscopy images (_DNA, RNA, ER, AGP_, and _Mito_).
Images are then processed using [CellProfiler software](https://cellprofiler.org/) to extract thousands of features of each cell’s morphology and form a high-dimensional profile for each single cell.
These features are based on various shape, intensity and texture statistics and are then aggregated for all the single cells in a "well" (a miniature test tube) that are called replicate-level profiles of perturbations.
Aggregation of replicate-level profiles across all the wells or replicates of a perturbation is called a treatment-level profile.
In our study, we used treatment-level profiles in all experiments but have provided replicate-level profiles for researchers interested in further data exploration.

</details>

# Datasets

- We have gathered the following five available data sets that had both Cell Painting morphological (CP) and L1000 gene expression (GE) profiles, preprocessed the data from different sources and in different formats in a unified .csv format.

    - CDRP-BBBC047-Bray-CP-GE (Cell line: U2OS)
    - CDRPBIO-BBBC036-Bray-CP-GE (Cell line: U2OS)
    - LUAD-BBBC041-Caicedo-CP-GE (Cell line: A549)
    - TA-ORF-BBBC037-Rohban-CP-GE (Cell line: U2OS)
    - LINCS-Pilot1-CP-GE (Cell line: A549)

## References to raw profiles and images
<details>
<summary>Click to expand</summary>

- CDRP-BBBC047-Bray-[CP](https://pubmed.ncbi.nlm.nih.gov/28327978/) - [GE](https://pubmed.ncbi.nlm.nih.gov/29195078/)
- CDRP-bio-BBBC036-Bray-[CP](https://pubmed.ncbi.nlm.nih.gov/28327978/) - [GE](https://pubmed.ncbi.nlm.nih.gov/29195078/)
- LUAD-BBBC041-Caicedo-[CP](https://registry.opendata.aws/cell-painting-image-collection/) - [GE](https://pubmed.ncbi.nlm.nih.gov/27478040/)
- TA-ORF-BBBC037-Rohban-[CP](https://elifesciences.org/articles/24060) - [GE](https://github.com/carpenterlab/2017_rohban_elife/tree/master/input/TA-OE-L1000-B1)
- LINCS-Pilot1-[CP](https://zenodo.org/record/3928744#.YNu3WzZKheV) - [GE](https://figshare.com/articles/dataset/L1000_data_for_profiling_comparison/13181966)

</details>


## Preprocessed publicly available profiles
Preprocessed profiles (~9.5GB) are available on a S3 bucket.
They can be downloaded at no cost and no need for registration of any sort, using the command:

```bash
aws s3 sync \
  --no-sign-request \
  s3://cellpainting-gallery/cpg0003-rosetta/broad/workspace/preprocessed_data .
```

See this [wiki](https://github.com/carpenterlab/2016_bray_natprot/wiki/What-do-Cell-Painting-features-mean%3F) for sample Cell Painting images and the meaning of ([CellProfiler](https://cellprofiler.org/)-derived) Cell Painting features.

- AWS CLI installation instructions can be found [here](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

### Data version

The [Etags](https://docs.aws.amazon.com/AmazonS3/latest/API/API_Object.html) of these files are listed [here](etag.json).

They were generated using:

```sh
aws s3api list-objects --bucket cellpainting-gallery --prefix rosetta/broad/workspace/preprocessed_data/
```
### CP-L1000 Profile descriptions

We gathered four available data sets that had both Cell Painting morphological (CP) and L1000 gene expression (GE) profiles, preprocessed the data from different sources and in different formats in a unified .csv format, and made the data publicly available. Single cell morphological (CP) profiles were created using CellProfiler software and processed to form aggregated replicate and treatment levels using the R cytominer package [cytominer](https://github.com/cytomining/cytominer/blob/master/vignettes/cytominer-pipeline.Rmd).
We made the following three types of profiles available for cell-painting modality of each of four datasets:


| Folder       | File name                                                | Description                                                                                                                                                  |
| ------------ | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| CellPainting | `replicate_level_cp_augmented.csv`                       | Aggregated and Metadata annotated profiles which are the average of single cell profiles in each well.                                                       |
| CellPainting | `replicate_level_cp_normalized.csv.gz`                   | Normalized profiles which are the z-scored aggregated profiles, where the scores are computing using the distribution of negative controls as the reference. |
| CellPainting | `replicate_level_cp_normalized_variable_selected.csv.gz` | Normalized variable selected which are normalized profiles with features selection applied                                                                   |
| L1000        | `replicate_level_l1k.csv`                                | Aggregated and Metadata annotated profiles which are the average of single cell profiles in each well.                                                       |



### Metadata information

This [spreadsheet](https://docs.google.com/spreadsheets/d/1EpqBLJqio8ptGlZe9Ywq1OUJahKSpYNb6S4lJ9yFc0o/edit#gid=174183831) contains a description all the metadata fields across all 8 datasets.

#### Keywords to match tables across modalities for each dataset


| Dataset               | perturbation match column<br/>CP | perturbation match column<br/>GE | Control perturbation value in each of columns <br/>CP and GE | 
| :-------------------- | :------------------------------- | :------------------------------- | :---------------------------- | 
| CDRP-BBBC047-Bray     | Metadata_Sample_Dose             | pert_sample_dose                 | negcon                          |
| CDRPBIO-BBBC036-Bray  | Metadata_Sample_Dose             | pert_sample_dose                 | negcon                          |
| TA-ORF-BBBC037-Rohban | Metadata_broad_sample            | pert_id                          | negcon                          |
| LUAD-BBBC041-Caicedo  | x_mutation_status                | allele                           | negcon                   |
| LINCS-Pilot1          | Metadata_pert_id_dose            | pert_id_dose                     | negcon                          | 

* Two aditional columns can also be used to filter for the "Control perturbation" in each data table:
   -  **pert_type** wich can take 'trt' or 'control' values , and column control_type indicates negcon (otherwise empty).
   -  **control_type** wich can take 'negcon' (for control) or NaN (for treatments) values

#### Number of features for each dataset

| Dataset  | GE  | CP<br/>`normalized` | CP<br/>`normalized_variable_selected` |
| -------- | --- | ------------------- | ------------------------------------- |
| CDRP     | 977 | 1565                | 727                                   |
| CDRP-BIO | 977 | 1570                | 601                                   |
| LUAD     | 978 | 1569                | 291                                   |
| TA-ORF   | 978 | 1677                | 63                                    |
| LINCS    | 978 | 1670                | 119                                   |


# Lookup table for L1000 genes predictability

[Table](results/SingleGenePred/Appendix_D.csv)


# License

We license the data, results, and figures as [CC0 1.0](LICENSE_CC0.md) and the source code as BSD 3-Clause.

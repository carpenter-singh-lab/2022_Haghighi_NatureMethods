<!-- #### 2021_Haghighi_NeurIPS_Dataset_submitted -->
# High-Dimensional Gene Expression and Morphology Profiles of Cells across 28,000 Genetic and Chemical Perturbations



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
  
# Public Datasets (Preprocessed Profiles)
Preprocessed profiles are available on a S3 bucket. They can be downloaded using the command:

```bash
aws s3 cp \
  --recursive \
  s3://cellpainting-datasets/Rosetta-GE-CP .  
```

See this [wiki](https://github.com/carpenterlab/2016_bray_natprot/wiki/What-do-Cell-Painting-features-mean%3F) for sample Cell Painting images and the meaning of ([CellProfiler](https://cellprofiler.org/)-derived) Cell Painting features. 

#### CP Profile descriptions:
We gathered four available data sets that had both Cell Painting morphological (CP) and L1000 gene expression (GE) profiles, preprocessed the data from different sources and in different formats in a unified .csv format, and made the data publicly available. Single cell morphological (CP) profiles were created using CellProfiler software and processed to form aggregated replicate and treatment levels using the R cytominer package [cytominer](https://github.com/cytomining/cytominer/blob/master/vignettes/cytominer-pipeline.Rmd). 
We made the following three types of profiles available:


| File name                                                  | Description                                              |
| ---------------------------------------------------------- | -------------------------------------------------------- |
| `<plate_ID>_augmented.csv`                                 | Aggregated and Metadata annotated profiles which are the average of single cell profiles in each well.              |
| `<plate_ID>_normalized.csv.gz`                             | Normalized profiles which are the z-scored aggregated profiles, where the scores are computing using the distribution of negative controls as the reference.                  |
| `<plate_ID>_normalized_feature_select_plate.csv.gz`        | Normalized variable selected which are normalized profiles with features selection applied      |

# Running the analysis script notebooks

# License
We license the data, results, and figures as [CC0 1.0](LICENSE_CC0.md).

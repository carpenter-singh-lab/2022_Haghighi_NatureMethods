import numpy as np
import scipy.spatial
import pandas as pd
import sklearn.decomposition
from sklearn import preprocessing
from sklearn.metrics import pairwise_distances
# from utils.normalize_funcs import standardize_per_catX
from normalize_funcs import standardize_per_catX

#'dataset_name',['folder_name',[cp_pert_col_name,l1k_pert_col_name],[cp_control_val,l1k_control_val]]
ds_info_dict={'CDRP':['CDRP-BBBC047-Bray',['Metadata_Sample_Dose','pert_sample_dose']],
              'CDRP-bio':['CDRPBIO-BBBC036-Bray',['Metadata_Sample_Dose','pert_sample_dose']],
              'TAORF':['TA-ORF-BBBC037-Rohban',['Metadata_broad_sample','pert_id',]],
              'LUAD':['LUAD-BBBC041-Caicedo',['x_mutation_status','allele']],
              'LINCS':['LINCS-Pilot1',['Metadata_pert_id_dose','pert_id_dose']]}

labelCol='PERT'



################################################################################
def read_replicate_level_profiles(dataset_rootDir,dataset,profileType,per_plate_normalized_flag):
    """
    Reads replicate level CSV files in the form of a dataframe
    Extract measurments column names for each modalities
    Remove columns with low variance (<thrsh_var)
    Remove columns with more NaNs than a certain threshold (>null_vals_ratio)
    
    Inputs:
    dataset_rootDir: datasets root dir
    dataset: any from the available list of ['LUAD', 'TAORF', 'LINCS', 'CDRP-bio', 'CDRP']
    profileType:   Cell Painting profile type that can be 'augmented' , 'normalized', 'normalized_variable_selected'
    per_plate_normalized_flag: if True it will standardize data per plate 
    
    Output:
    cp_data_repLevel, l1k_data_repLevel: dataframes with all the annotations available in the raw data
    """

    dataDir=dataset_rootDir+'/preprocessed_data/'+ds_info_dict[dataset][0]+'/'
        
    cp_data_repLevel=pd.read_csv(dataDir+'/CellPainting/replicate_level_cp_'+profileType+'.csv.gz')    
    l1k_data_repLevel=pd.read_csv(dataDir+'/L1000/replicate_level_l1k.csv.gz')  

    cp_features, l1k_features =  extract_feature_names(cp_data_repLevel, l1k_data_repLevel);
    
    ########## removes nan and inf values
    l1k_data_repLevel=l1k_data_repLevel.replace([np.inf, -np.inf], np.nan)
    cp_data_repLevel=cp_data_repLevel.replace([np.inf, -np.inf], np.nan)
    
    #
    null_vals_ratio=0.05; thrsh_std=0.0001;
    cols2remove_manyNulls=[i for i in cp_features if (cp_data_repLevel[i].isnull().sum(axis=0)/cp_data_repLevel.shape[0])\
                  >null_vals_ratio]   
    cols2remove_lowVars = cp_data_repLevel[cp_features].std()[cp_data_repLevel[cp_features].std() < thrsh_std].index.tolist()

    cols2removeCP = cols2remove_manyNulls + cols2remove_lowVars
#     print(cols2removeCP)

    cp_features = list(set(cp_features) - set(cols2removeCP))
    cp_data_repLevel=cp_data_repLevel.drop(cols2removeCP, axis=1);
    cp_data_repLevel[cp_features] = cp_data_repLevel[cp_features].interpolate()
    
#     cols2removeCP=[i for i in cp_features if cp_data_repLevel[i].isnull().sum(axis=0)>0]
#     print(cols2removeCP)
    
#     cp=cp.fillna(cp.median())

    # cols2removeGE=[i for i in l1k.columns if l1k[i].isnull().sum(axis=0)>0]
    # print(cols2removeGE)
    # l1k_features = list(set(l1k_features) - set(cols2removeGE))
    # print(len(l1k_features))
    # l1k=l1k.drop(cols2removeGE, axis=1);
    l1k_data_repLevel[l1k_features] = l1k_data_repLevel[l1k_features].interpolate()
    # l1k=l1k.fillna(l1k.median())    
    
    ################ Per plate scaling 
    if per_plate_normalized_flag:
        cp_data_repLevel = standardize_per_catX(cp_data_repLevel,'Metadata_Plate',cp_features);
        l1k_data_repLevel = standardize_per_catX(l1k_data_repLevel,'det_plate',l1k_features);    

        cols2removeCP=[i for i in cp_features if (cp_data_repLevel[i].isnull().sum(axis=0)/cp_data_repLevel.shape[0])>0.05]
        cp_data_repLevel=cp_data_repLevel.drop(cols2removeCP, axis=1);
        cp_features = list(set(cp_features) - set(cols2removeCP))
        cp_data_repLevel[cp_features] = cp_data_repLevel[cp_features].interpolate()
    
    return [cp_data_repLevel,cp_features], [l1k_data_repLevel,l1k_features]


################################################################################
def extract_feature_names(cp_data_repLevel, l1k_data_repLevel):
    """
    extract Cell Painting and L1000 measurments names among the column names
    
    Inputs:
    cp_data_repLevel, l1k_data_repLevel: dataframes with all the annotations available in the raw data
    
    Outputs: list of feature names for each modality
    
    """
    # features to analyse
    cp_features=cp_data_repLevel.columns[cp_data_repLevel.columns.str.contains("Cells_|Cytoplasm_|Nuclei_")].tolist()
    l1k_features=l1k_data_repLevel.columns[l1k_data_repLevel.columns.str.contains("_at")].tolist()

    return cp_features, l1k_features


################################################################################
def extract_metadata_column_names(cp_data, l1k_data):
    """
    extract metadata column names among the column names for any level of data
    
    Inputs:
    cp_data_repLevel, l1k_data_repLevel: dataframes with all the annotations available in the raw data
    
    Outputs: list of metadata column names for each modality
    
    """
    cp_meta_col_names=cp_data.columns[~cp_data.columns.str.contains("Cells_|Cytoplasm_|Nuclei_")].tolist()
    l1k_meta_col_names=l1k_data.columns[~l1k_data.columns.str.contains("_at")].tolist()

    return cp_meta_col_names, l1k_meta_col_names

################################################################################
def read_treatment_level_profiles(dataset_rootDir,dataset,profileType,filter_perts,per_plate_normalized_flag):

    """
    Reads replicate level CSV files (scaled replicate level profiles per plate)
    Rename the column names to match across datasets to PERT in both modalities
    Remove perturbations with low rep corr across both (filter_perts='highRepOverlap') 
            or one of the modalities (filter_perts='highRepUnion')
    Form treatment level profiles by averaging the replicates
    Select and keep the metadata columns you want to keep for each dataset
    Merge treatment level profiles to its own metadata
    
    Inputs:
    dataset_rootDir: datasets root dir
    dataset: any from the available list of ['LUAD', 'TAORF', 'LINCS', 'CDRP-bio', 'CDRP']
    profileType:   Cell Painting profile type that can be 'augmented' , 'normalized', 'normalized_variable_selected'

    Output: 
    [cp_data_treatLevel,cp_features], [l1k_data_treatLevel,l1k_features]
    each is a list of dataframe and feature names for each of modalities
    """
    
    [cp_data_repLevel,cp_features], [l1k_data_repLevel,l1k_features] = read_replicate_level_profiles(dataset_rootDir,dataset,profileType,per_plate_normalized_flag);
        

    ############ rename columns that should match to PERT
    labelCol='PERT'
    cp_data_repLevel=cp_data_repLevel.rename(columns={ds_info_dict[dataset][1][0]:labelCol})
    l1k_data_repLevel=l1k_data_repLevel.rename(columns={ds_info_dict[dataset][1][1]:labelCol})    
            
    
    ###### print some data statistics
    print(dataset+': Replicate Level Shapes (nSamples x nFeatures): cp: ',\
          cp_data_repLevel.shape[0],',',len(cp_features),  ',  l1k: ',l1k_data_repLevel.shape[0],',',len(l1k_features))

    print('l1k n of rep: ',l1k_data_repLevel.groupby([labelCol]).size().median())
    print('cp n of rep: ',cp_data_repLevel.groupby([labelCol]).size().median())
    

    ###### remove perts with low rep corr
    if filter_perts=='highRepOverlap':    
        highRepPerts = highRepFinder(dataset,'intersection') + ['DMSO'];
        
        cp_data_repLevel=cp_data_repLevel[cp_data_repLevel['PERT'].isin(highRepPerts)].reset_index()
        l1k_data_repLevel=l1k_data_repLevel[l1k_data_repLevel['PERT'].isin(highRepPerts)].reset_index()  
        
    elif filter_perts=='highRepUnion':
        highRepPerts = highRepFinder(dataset,'union') + ['DMSO'];
        
        cp_data_repLevel=cp_data_repLevel[cp_data_repLevel['PERT'].isin(highRepPerts)].reset_index()
        l1k_data_repLevel=l1k_data_repLevel[l1k_data_repLevel['PERT'].isin(highRepPerts)].reset_index()      
    
    ####### form treatment level profiles
    l1k_data_treatLevel=l1k_data_repLevel.groupby(labelCol)[l1k_features].mean().reset_index();
    cp_data_treatLevel=cp_data_repLevel.groupby(labelCol)[cp_features].mean().reset_index();
    
    ###### define metadata and merge treatment level profiles
#     dataset:[[cp_columns],[l1k_columns]]
#     meta_dict={'CDRP':[['Metadata_moa','Metadata_target'],['CPD_NAME','CPD_TYPE','CPD_SMILES']],
#                'CDRP-bio':[['Metadata_moa','Metadata_target'],['CPD_NAME','CPD_TYPE','CPD_SMILES']],
#               'TAORF':[['Metadata_moa'],['pert_type']],
#               'LUAD':[['Metadata_broad_sample_type','Metadata_pert_type'],[]],
#               'LINCS':[['Metadata_moa', 'Metadata_alternative_moa'],['moa']]}

    meta_dict={'CDRP':[['Metadata_moa','Metadata_target'],[]],
               'CDRP-bio':[['Metadata_moa','Metadata_target'],[]],
              'TAORF':[[],[]],
              'LUAD':[[],[]],
              'LINCS':[['Metadata_moa', 'Metadata_alternative_moa'],['moa']]}
    
    
    meta_cp=cp_data_repLevel[[labelCol]+meta_dict[dataset][0]].\
    drop_duplicates().reset_index(drop=True)
    meta_l1k=l1k_data_repLevel[[labelCol]+meta_dict[dataset][1]].\
    drop_duplicates().reset_index(drop=True)

    cp_data_treatLevel=pd.merge(cp_data_treatLevel,meta_cp, how='inner',on=[labelCol])
    l1k_data_treatLevel=pd.merge(l1k_data_treatLevel,meta_l1k, how='inner',on=[labelCol])
    
    return [cp_data_treatLevel,cp_features], [l1k_data_treatLevel,l1k_features]



################################################################################
def read_paired_treatment_level_profiles(dataset_rootDir,dataset,profileType,filter_perts,per_plate_normalized_flag):

    """
    Reads treatment level profiles
    Merge dataframes by PERT column
    
    Inputs:
    dataset_rootDir: datasets root dir
    dataset: any from the available list of ['LUAD', 'TAORF', 'LINCS', 'CDRP-bio', 'CDRP']
    profileType:   Cell Painting profile type that can be 'augmented' , 'normalized', 'normalized_variable_selected'
    per_plate_normalized_flag: True for scaling per plate

    Output: 
    mergedProfiles_treatLevel: paired treatment level profiles
    cp_features,l1k_features list of feature names for each of modalities
    """
    
    [cp_data_treatLevel,cp_features], [l1k_data_treatLevel,l1k_features]=\
    read_treatment_level_profiles(dataset_rootDir,dataset,profileType,filter_perts,per_plate_normalized_flag)
    

    mergedProfiles_treatLevel=pd.merge(cp_data_treatLevel, l1k_data_treatLevel, how='inner',on=[labelCol])

    print('Treatment Level Shapes (nSamples x nFeatures+metadata):',cp_data_treatLevel.shape,l1k_data_treatLevel.shape,\
          'Merged Profiles Shape:', mergedProfiles_treatLevel.shape)

    
    return mergedProfiles_treatLevel,cp_features,l1k_features


################################################################################
def generate_random_match_of_replicate_pairs(cp_data_repLevel, l1k_data_repLevel,nRep):
    """
    Note that there is no match at the replicate level for this dataset, we either:
        - Forming ALL the possible pairs for replicate level data matching (nRep='all' - string)
        - Randomly sample samples in each modality and form pairs (nRep -> int)
        
    Inputs:
        cp_data_repLevel, l1k_data_repLevel: dataframes with all the annotations available in the raw data
    
    Outputs: 
        Randomly paired replicate level profiles
    
    """
    labelCol='PERT'
    
    if nRep=='all':
        cp_data_n_repLevel=cp_data_repLevel.copy()
        l1k_data_n_repLevel=l1k_data_repLevel.copy()
    else:
#         nR=np.min((cp_data_repLevel.groupby(labelCol).size().min(),l1k_data_repLevel.groupby(labelCol).size().min()))
#     cp_data_n_repLevel=cp_data_repLevel.groupby(labelCol).apply(lambda x: x.sample(n=nR,replace=True)).reset_index(drop=True)
        nR=nRep
        cp_data_n_repLevel=cp_data_repLevel.groupby(labelCol).\
        apply(lambda x: x.sample(n=np.min([nR,x.shape[0]]))).reset_index(drop=True)
        l1k_data_n_repLevel=l1k_data_repLevel.groupby(labelCol).\
        apply(lambda x: x.sample(n=np.min([nR,x.shape[0]]))).reset_index(drop=True)


    mergedProfiles_repLevel=pd.merge(cp_data_n_repLevel, l1k_data_n_repLevel, how='inner',on=[labelCol])

    return mergedProfiles_repLevel

################################################################################
def highRepFinder(dataset,how):
    """
    This function reads pre calculated and saved Replicate Correlation values file and filters perturbations
    using one of the following filters:
        - intersection: intersection of high quality profiles across both modalities
        - union: union of high quality profiles across both modalities
        
    * A High Quality profile is defined as a profile having replicate correlation more than 90th percentile of
      its null distribution
        
    Inputs:
        dataset (str): dataset name
        how (str):  can be intersection or union
    
    Output: list of high quality perurbations
    
    """

    repCorDF=pd.read_excel('../results/RepCor/RepCorrDF.xlsx', sheet_name=None)
    cpRepDF=repCorDF['cp-'+dataset.lower()]
    cpHighList=cpRepDF[cpRepDF['RepCor']>cpRepDF['Rand90Perc']]['Unnamed: 0'].tolist()
    print('CP: from ',cpRepDF.shape[0],' to ',len(cpHighList))
    cpRepDF=repCorDF['l1k-'+dataset.lower()]
    l1kHighList=cpRepDF[cpRepDF['RepCor']>cpRepDF['Rand90Perc']]['Unnamed: 0'].tolist()
#     print("l1kHighList",l1kHighList)
#     print("cpHighList",cpHighList)   
    if how=='intersection':
        highRepPerts=list(set(l1kHighList) & set(cpHighList))
        print('l1k: from ',cpRepDF.shape[0],' to ',len(l1kHighList))
        print('CP and l1k high rep overlap: ',len(highRepPerts))
        
    elif how=='union':
        highRepPerts=list(set(l1kHighList) | set(cpHighList))
        print('l1k: from ',cpRepDF.shape[0],' to ',len(l1kHighList))
        print('CP and l1k high rep union: ',len(highRepPerts))        
        
    return highRepPerts


################################################################################
def read_paired_replicate_level_profiles(dataset_rootDir,dataset,profileType,nRep,filter_perts,per_plate_normalized_flag):

    """
    Reads replicate level CSV files (scaled replicate level profiles per plate)
    Rename the column names to match across datasets to PERT in both modalities
    Remove perturbations with low rep corr across both (filter_perts='highRepOverlap') 
            or one of the modalities (filter_perts='highRepUnion')
    Form treatment level profiles by averaging the replicates
    Select and keep the metadata columns you want to keep for each dataset
    Merge dataframes by PERT column
    
    Inputs:
    dataset_rootDir: datasets root dir
    dataset: any from the available list of ['LUAD', 'TAORF', 'LINCS', 'CDRP-bio', 'CDRP']
    profileType:   Cell Painting profile type that can be 'augmented' , 'normalized', 'normalized_variable_selected'

    Output: 
    mergedProfiles_treatLevel: paired treatment level profiles
    cp_features,l1k_features list of feature names for each of modalities
    """
    
    [cp_data_repLevel,cp_features], [l1k_data_repLevel,l1k_features] = read_replicate_level_profiles(dataset_rootDir,dataset,profileType,per_plate_normalized_flag);
        

    ############ rename columns that should match to PERT
    cp_data_repLevel=cp_data_repLevel.rename(columns={ds_info_dict[dataset][1][0]:labelCol})
    l1k_data_repLevel=l1k_data_repLevel.rename(columns={ds_info_dict[dataset][1][1]:labelCol})    
            
    
    ###### print some data statistics
    print(dataset+': Replicate Level Shapes (nSamples x nFeatures): cp: ',\
          cp_data_repLevel.shape[0],',',len(cp_features),  ',  l1k: ',l1k_data_repLevel.shape[0],',',len(l1k_features))

    print('l1k n of rep: ',l1k_data_repLevel.groupby([labelCol]).size().median())
    print('cp n of rep: ',cp_data_repLevel.groupby([labelCol]).size().median())
    

    ###### remove perts with low rep corr
    if filter_perts=='highRepOverlap':    
        highRepPerts = highRepFinder(dataset,'intersection') + ['DMSO'];
        
        cp_data_repLevel=cp_data_repLevel[cp_data_repLevel['PERT'].isin(highRepPerts)].reset_index()
        l1k_data_repLevel=l1k_data_repLevel[l1k_data_repLevel['PERT'].isin(highRepPerts)].reset_index()  
        
    elif filter_perts=='highRepUnion':
        highRepPerts = highRepFinder(dataset,'union') + ['DMSO'];
        
        cp_data_repLevel=cp_data_repLevel[cp_data_repLevel['PERT'].isin(highRepPerts)].reset_index()
        l1k_data_repLevel=l1k_data_repLevel[l1k_data_repLevel['PERT'].isin(highRepPerts)].reset_index()      
    

    mergedProfiles_repLevel=generate_random_match_of_replicate_pairs(cp_data_repLevel, l1k_data_repLevel,nRep)

    
    return mergedProfiles_repLevel,cp_features,l1k_features



def rename_affyprobe_to_genename(l1k_data_df,l1k_features):
    """
    map input dataframe column name from affy prob id to gene names
    
    """

    meta=pd.read_csv("../affy_probe_gene_mapping.txt",delimiter="\t",header=None, names=["probe_id", "gene"])
    meta_gene_probID=meta.set_index('probe_id')
    d = dict(zip(meta_gene_probID.index, meta_gene_probID['gene']))
    l1k_features_gn=[d[l] for l in l1k_features]
    l1k_data_df = l1k_data_df.rename(columns=d)   

    return l1k_data_df,l1k_features_gn


 

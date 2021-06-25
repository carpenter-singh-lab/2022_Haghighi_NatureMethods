def standardize_per_catX(df,column_name,cp_features):
# column_name='Metadata_Plate'
#     cp_features=df.columns[df.columns.str.contains("Cells_|Cytoplasm_|Nuclei_")]
    df_scaled_perPlate=df.copy()
    df_scaled_perPlate[cp_features]=\
    df[cp_features+[column_name]].groupby(column_name)\
    .transform(lambda x: (x - x.mean()) / x.std()).values
    return df_scaled_perPlate


#imports
import pandas as pd


#This function performs adding labels to gene expression data
#geneCounts is the gene expression data
#covariates is the labels
#num_thresh is the number of non zero values
#return is the labeled gene expression data
def add_label(geneCounts, covariates, num_thresh):
    
    covariates.rename(columns={covariates.columns[0]: "ID" }, inplace=True)
    covar_index = covariates[['ID','Diagnosis']]
    covar_index_T = covar_index.T
    covar_index_T.drop('ID',axis=0,inplace=True)
    df = covar_index_T.rename(columns = covariates['ID'])
    dft = df.T
    
    geneCounts_T = geneCounts.T
    df_na= geneCounts_T.replace(0,pd.np.nan)
    df_na=df_na.dropna(axis=1, thresh = num_thresh)
    df_drop=df_na.replace(pd.np.nan,0)
    df_drop.drop('ensembl_id',axis=0,inplace=True)
    df2 = df_drop.rename(columns = geneCounts['ensembl_id'])
    labeled = pd.concat([dft, df2], axis=1, sort=False)
    
    return labeled


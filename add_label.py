# In[ ]:

import pandas as pd

def add_label(geneCounts, covariates):
    
    covariates.rename(columns={covariates.columns[0]: "ID" }, inplace=True)
    covar_index = covariates[['ID','Diagnosis']]
    covar_index_T = covar_index.T
    covar_index_T.drop('ID',axis=0,inplace=True)
    df = covar_index_T.rename(columns = covariates['ID'])
    dft = df.T
    
    geneCounts_T = geneCounts.T
    geneCounts_T.drop('ensembl_id',axis=0,inplace=True)
    df2 = geneCounts_T.rename(columns = geneCounts['ensembl_id'])
    labeled = pd.concat([dft, df2], axis=1, sort=False)
    
    return labeled


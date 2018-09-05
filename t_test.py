# In[ ]:

#imports
import numpy as np
import pandas as pd

#This function performs t-test for feature selection
#data_x is the gene expression data
#data_y is the lables
#num is the required number of features
#returns are the selected features dataframe
def t_test(data_x, data_y, num):
    
    l1 = data_y[data_y['Diagnosis'] == 0].index
    m1 = data_x.loc[l1].apply(np.mean,0)
    l2 = data_y[data_y['Diagnosis'] == 1].index
    m2 = data_x.loc[l2].apply(np.mean,0)
    
    numerator = np.absolute(m1 - m2)
    
    sd1 = data_x.loc[l1].apply(np.std,0)
    sd2 = data_x.loc[l2].apply(np.std,0)
    
    denominator = np.sqrt((sd1/l1.shape[0])+(sd2/l2.shape[0]))
    
    t_statistics = numerator/denominator
    
    idx = np.argsort(t_statistics)[-num:]
    features = data_x.iloc[:,idx]
    
    return features


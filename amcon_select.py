#imports
import numpy as np
import pandas as pd
import re_idx
import com_list

#This function performs finding the list of common samples
#data is the gene expression data with sample index
#return is the common list
def com_list(data):
    
    lis = list(data.index)

    lis_new = []
    
    for i in np.arange(len(lis)):
        if len(lis[i]) == 7:
            t = lis[i][:3]
        if len(lis[i]) == 8:
            t = lis[i][:4]
        if len(lis[i]) == 9:
            t = lis[i][:5]
        lis_new.append(t)
        
    newlis = np.array(lis_new)
    
    finlis = pd.DataFrame(newlis)[0]
    
    return finlis

#This function performs to rename the dataframe with common list
#data is the gene expression data
#return is the renamed data
def re_idx(data):
    
    re = data.reset_index()
    ret = re.T
    ren_fin = ret.rename(columns = com_list(data))
    ren_fin.drop('index',axis=0,inplace=True)
    
    re_data = ren_fin.T
    
    return re_data

#This function performs selecting common samples for two data set
#data_c and data_t are two data set that some samples id of them are same
#return are two data set with common samples id
def amcon_select(data_c, data_t):
    
    
    finlisCBEb = com_list(data_c)
    finlisTCXb = com_list(data_t)

    C = re_idx(data_c)
    T = re_idx(data_t)

    num_matchCT = 0
    com_lis = []
    for i in finlisCBEb:
        for j in finlisTCXb:
            if i == j:
                com_lis.append(j)
                num_matchCT += 1
    comlis = np.array(com_lis)

    
    finC = C.loc[comlis]
    finT = T.loc[comlis]
    
    
    return finC, finT


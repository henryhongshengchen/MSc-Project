# In[ ]:


import numpy as np
import pandas as pd
import re_idx


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


def re_idx(data):
    
    re = data.reset_index()
    ret = re.T
    ren_fin = ret.rename(columns = com_list(data))
    ren_fin.drop('index',axis=0,inplace=True)
    
    re_data = ren_fin.T
    
    return re_data


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


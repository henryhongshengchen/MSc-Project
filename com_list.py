#inputs
import numpy as np
import pandas as pd


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

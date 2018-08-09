# In[ ]:


import numpy as np
import pandas as pd
import com_list

def re_idx(data):
    
    re = data.reset_index()
    ret = re.T
    ren_fin = ret.rename(columns = com_list(data))
    ren_fin.drop('index',axis=0,inplace=True)
    
    re_data = ren_fin.T
    
    return re_data

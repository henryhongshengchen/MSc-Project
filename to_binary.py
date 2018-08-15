# In[ ]:


import pandas as pd

def to_binary(data):

    index_col = list(data)[1:]
    
    AD = data[data['Diagnosis'] == 'AD']
    Control = data[data['Diagnosis'] == 'Control']
    data_b = pd.concat([AD, Control], axis=0, sort=True)
    
    diagnosis_mapping = {'AD': 1, 'Control': 0}
    data_b['Diagnosis'] = data_b['Diagnosis'].map(diagnosis_mapping)
    
    data_y = pd.DataFrame(data_b['Diagnosis'])
    data_x = data_b[index_col]
    
    return data_x, data_y


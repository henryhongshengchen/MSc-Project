#imports
import pandas as pd

#This function performs selecting labels 'AD' and 'Control' then changing to 1 and 0
#data is gene expression data containing labels
#returns are data_x without labels and data_y with 1 and 0
def label_to_adncon(data):

    index_col = list(data)[1:]
    
    AD = data[data['Diagnosis'] == 'AD']
    Control = data[data['Diagnosis'] == 'Control']
    data_b = pd.concat([AD, Control], axis=0, sort=True)
    
    diagnosis_mapping = {'AD': 1, 'Control': 0}
    data_b['Diagnosis'] = data_b['Diagnosis'].map(diagnosis_mapping)
    
    data_y = pd.DataFrame(data_b['Diagnosis'])
    data_x = data_b[index_col]
    
    return data_x, data_y


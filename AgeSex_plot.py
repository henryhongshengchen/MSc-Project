# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

def AgeSex_plot(data_cov):
    
    C = data_cov
    C = C.dropna()
    AD = C[C['Diagnosis'] == 'AD']
    Control = C[C['Diagnosis'] == 'Control']
    C = pd.concat([AD, Control], axis=0, sort=True)
    
    S = C['Sex']
    A = C['AgeAtDeath']
    D = C['Diagnosis']
    C = C.replace('90_or_above', '90')
    
    plt.figure(figsize=(15,10))
    plt.bar(A.value_counts().index, A.value_counts().values, fill = 'navy', edgecolor = 'k', width = 1)
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.title('Distribution of Age At Death');
    plt.xticks(np.arange(len(A.value_counts().index)),(A.value_counts().index))
    plt.show()
    
    plt.figure(figsize=(15,10))
    sns.kdeplot(C.loc[C['Sex'] == 'M', 'AgeAtDeath'], label = 'Male', shade = True)
    sns.kdeplot(C.loc[C['Sex'] == 'F', 'AgeAtDeath'], label = 'Female', shade = True)
    plt.xlabel('Age')
    plt.ylabel('Density')
    plt.title('Density Plot of Age by Sex')
    plt.show()


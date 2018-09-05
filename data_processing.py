# In[ ]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

def data_processing(data):

    df= data.replace(0,pd.np.nan)
    
    for i in np.arange(1, len(data)):
        
        d=df.dropna(axis=1, thresh = i)
        plt.scatter(x=i, y=d.shape[1])
        
    plt.xlabel('thresh')
    plt.ylabel('number of features')
    plt.title('number of features against thresh')
    plt.show()


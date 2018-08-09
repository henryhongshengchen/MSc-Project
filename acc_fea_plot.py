
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

def acc_fea_plot(accu):
    
    features_num = ['5', '10', '15', '20', '50', '75', '100', '250', '500', '1000', '5000', '10000']
    plt.bar(range(len(features_num)), accu)
    plt.xticks(np.arange(len(features_num)),(features_num))
    plt.xlabel('Number of Features')
    plt.ylabel('Linear Model Accuracy')
    plt.title('Linear Model Accuracy against Number of Features')
    plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

def best_acc_plot(accu):
    
    features_num = ['15000', '3000', '1500', '750', '300', '225', '150', '90', '60', '45', '30', '15']
    plt.bar(range(len(features_num)), accu)
    plt.xticks(np.arange(len(features_num)),(features_num))
    plt.xlabel('Number of Features')
    plt.ylabel('Combined Feature Accuracy')
    plt.title('Combined Feature Model Accuracy against Number of Features')
    plt.show()


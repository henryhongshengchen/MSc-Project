# In[ ]:


import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

def tsne_plot(data_x, data_y, num_com):
    
    XX = data_x.values
    yy = np.ravel(data_y)
    target_names = np.unique(yy)
    print('\nThere are %d unique target valuess in this dataset:' % (len(target_names)), target_names)
    
    tsne = TSNE(n_components=num_com, random_state=1001, perplexity=30, method='barnes_hut', n_iter=1000, verbose=1)
    X_tsne = tsne.fit_transform(XX)

    colors = ['red', 'green']
    plt.figure(2, figsize=(15, 10))

    for color, i, target_name in zip(colors, [0, 1], target_names):
        plt.scatter(X_tsne[yy == i, 0], X_tsne[yy == i, 1], color=color, s=200, alpha=.8, label=target_name, marker='.')
    plt.legend(loc='best', shadow=False, scatterpoints=3)
    plt.title('Scatter plot of t-SNE embedding')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.savefig('t-SNE-porto-01.png', dpi=150)
    plt.show()


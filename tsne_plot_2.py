#imports
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


#This function performs t-SNE plot in 2d
#data_x is the gene expression data
#data_y is the labels
#num_com is the number of components
#return is the plot
def tsne_plot_2(data_x, data_y, num_com):
    
    XX = data_x.values
    yy = np.ravel(data_y)
    target_names = np.unique(yy)
    print('\nThere are %d unique target valuess:' % (len(target_names)), target_names)
    
    tsne = TSNE(n_components=num_com, random_state=1001, perplexity=30, method='barnes_hut', n_iter=1000, verbose=1)
    X_tsne = tsne.fit_transform(XX)

    colors = ['red', 'green']
    plt.figure(2, figsize=(10, 5))

    for color, i, target_name in zip(colors, [0, 1], target_names):
        plt.scatter(X_tsne[yy == i, 0], X_tsne[yy == i, 1], color=color, s=200, alpha=.8, label=target_name, marker='.')
    plt.legend(loc='best', shadow=False, scatterpoints=3)
    plt.title('Scatter plot of t-SNE')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


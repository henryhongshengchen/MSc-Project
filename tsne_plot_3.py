#imports
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns; sns.set()


#This function performs t-SNE plot in 3d
#data_x is the gene expression data
#data_y is the labels
#num_com is the number of components
#return is the plot
def tsne_plot_3(data_x, data_y, num_com):
    
    XX = data_x.values
    yy = np.ravel(data_y)
    target_names = np.unique(yy)
    print('\nThere are %d unique target valuess in this dataset:' % (len(target_names)), target_names)
    
    tsne = TSNE(n_components=num_com, random_state=1001, perplexity=30, method='barnes_hut', n_iter=1000, verbose=1)
    X_tsne = tsne.fit_transform(XX)

    df = pd.DataFrame(X_tsne)

    fig = plt.figure(figsize=(15, 8))
    
    colors = df[0].values
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(df[0], df[1], df[2], alpha=0.8, c = colors, s=30)
    fig.colorbar(p)

    plt.title('Scatter plot of t-SNE')
    ax.set_xlabel('1st')
    ax.set_ylabel('2nd')
    ax.set_zlabel('3rd')
    plt.show()


# In[ ]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()

def pca_plot(data_x, data_y, n_comp):
    
    XX = data_x.values
    yy = np.ravel(data_y)
    target_names = np.unique(yy)
    print('\nThere are %d unique target valuess in this dataset:' % (len(target_names)), target_names)
    
    print('\nRunning PCA ...')
    pca = PCA(n_components=n_comp, svd_solver='full', random_state=1001)
    X_pca = pca.fit_transform(XX)
    print('Explained variance: %.4f' % pca.explained_variance_ratio_.sum())

    print('Individual variance contributions:')
    for j in range(n_comp):
        print(pca.explained_variance_ratio_[j])

    colors = ['blue', 'yellow']
    plt.figure(2, figsize=(15, 10))

    for color, i, target_name in zip(colors, [0, 1], target_names):
        plt.scatter(X_pca[yy == i, 0], X_pca[yy == i, 1], color=color, s=200,
                    alpha=.8, label=target_name, marker='.')
    plt.legend(loc='best', shadow=False, scatterpoints=3)
    plt.title(
            "Scatter plot of the training data projected on the 1st "
            "and 2nd principal components")
    plt.xlabel("Principal axis 1 - Explains %.1f %% of the variance" % (
            pca.explained_variance_ratio_[0] * 100.0))
    plt.ylabel("Principal axis 2 - Explains %.1f %% of the variance" % (
            pca.explained_variance_ratio_[1] * 100.0))

    plt.savefig('pca-porto-01.png', dpi=150)
    plt.show()


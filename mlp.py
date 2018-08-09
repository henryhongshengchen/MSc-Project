
# coding: utf-8

# In[6]:


import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import ActivityRegularization, LeakyReLU
from keras.utils import np_utils
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

def mlp(num_fea, data_x, data_y, num_com):
    
    estimator_c = LinearSVC()
    m = RFE(estimator_c, num_fea, step=0.1)
    x_new = m.fit_transform(data_x, np.ravel(data_y))
    mast=m.get_support()
    
#     fea=[]
#     i=0
#     for bool in mast:
#         if bool:
#             fea.append(i)
#         i+=1
        
    pca = PCA(n_components = num_com)
    pca.fit(x_new)
    x_new = pca.transform(x_new)
    x_new = x_new.tolist()
    length=len(x_new[0])
    
    Y=np_utils.to_categorical(data_y)
    X=np.array(x_new)
    print('number of gene '+str(len(x_new[0])))
    
    model = Sequential()
    model.add(Dense(4*length, activation='linear',input_shape=(length,)))
    model.add(LeakyReLU(alpha=.1))
    model.add(ActivityRegularization(l2=0.001))
    model.add(Dense(4*length, activation='linear'))
    model.add(LeakyReLU(alpha=.1))
    model.add(ActivityRegularization(l2=0.001))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
    model.summary()
    
    train_set, test_set, train_classes, test_classes = train_test_split(X, Y, test_size=0.2, random_state=0)
    h = model.fit(X, Y, validation_split=0.2, epochs=30, batch_size=10, verbose=0)
    
    # summarize history for accuracy
    plt.plot(h.history['binary_accuracy'])
    plt.plot(h.history['val_binary_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()

    # summarize history for loss
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()


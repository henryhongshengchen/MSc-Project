#imports
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.svm import LinearSVC
import numpy as np


#set the number of feature range and parameter C range
features_num = [5, 10, 15, 20, 30, 50, 75, 100, 250, 500, 1000, 5000]
num_costs = 20
cost_range = np.logspace(np.log10(1), np.log10(100), num_costs, base=10.0)


#This function performs RFE feature selection along with training Multi-layer Perceptron
#data_x is the gene expression data
#data_y is the labels
#return is the evaluation of model prediction accuracy
def mlp_fs11(data_x, data_y):
    
    K = 10
    kf = KFold(n_splits=K)

    accur = []

    for num in features_num:

        data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(data_x, data_y, test_size=0.1)

        estimator = LinearSVC()
        selector = RFE(estimator, num, step=0.1)
        data_rfe = selector.fit_transform(data_x_train, np.ravel(data_y_train))
    
        print('selected num of features: ', num)

        data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(data_rfe, data_y_train, test_size=0.1)

        mlp = MLPClassifier(hidden_layer_sizes=(60,),
                              activation='logistic', solver='lbfgs',
                              learning_rate_init = 0.0001,
                              max_iter=1500,
                              alpha = 0.001)
        mlp.fit(data_x_train, np.ravel(data_y_train))
        y_pred = mlp.predict(data_x_test)
        accur.append(accuracy_score(data_y_test, y_pred))
        #print('Accuracy: ', accuracy_score(data_y_test, y_pred), 'Loss: ', mlp.loss_)
        #print(confusion_matrix(data_y_test, y_pred))
        print(classification_report(data_y_test, y_pred))
    
    return accur


#imports
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd


#set the number of feature range and parameter C range
features_num = [5, 10, 15, 20, 30, 50, 75, 100, 250, 500, 1000, 5000]
num_costs = 20
cost_range = np.logspace(np.log10(1), np.log10(100), num_costs, base=10.0)


#This function performs RFE feature selection along with training Multi-layer Perceptron
#finC_x is the common samples of gene expression data in CBE data
#finC_y is the labels corresponding to finC_x
#finT_x is the common samples of gene expression data in TCX data
#finT_y is the labels corresponding to finT_x
#return is the evaluation of model prediction accuracy
def mlp_res1(finC_x, finC_y, finT_x, finT_y):
    
    K = 10
    kf = KFold(n_splits=K)

    accur = []

    dataC_x_train, dataC_x_test, dataC_y_train, dataC_y_test = train_test_split(finC_x, finC_y, test_size=0.1)

    dataT_x_train, dataT_x_test, dataT_y_train, dataT_y_test = train_test_split(finT_x, finT_y, test_size=0.1)

    estimator_c1 = LinearSVC()
    selector_c1 = RFE(estimator_c1, 10000, step=0.1)
    new_x_c1 = selector_c1.fit_transform(dataC_x_train, np.ravel(dataC_y_train))

    estimator_t1 = LinearSVC()
    selector_t1 = RFE(estimator_t1, 10000, step=0.1)
    new_x_t1 = selector_t1.fit_transform(dataT_x_train, np.ravel(dataT_y_train))

    new_x = pd.concat([pd.DataFrame(new_x_c1), pd.DataFrame(new_x_t1)], axis = 1)

    for num in features_num:

        estimator_c = LinearSVC()
        selector_c = RFE(estimator_c, num, step=0.1)
        new_x_c = selector_c.fit_transform(new_x_c1, np.ravel(dataC_y_train))

        estimator_t = LinearSVC()
        selector_t = RFE(estimator_t, num, step=0.1)
        new_x_t = selector_t.fit_transform(new_x_t1, np.ravel(dataT_y_train))

        estimator_ = LinearSVC()
        selector_ = RFE(estimator_, num, step=0.1)
        new_x_ = selector_.fit_transform(new_x, np.ravel(dataT_y_train))

        new_x = pd.concat([pd.DataFrame(new_x_), pd.concat([pd.DataFrame(new_x_c), pd.DataFrame(new_x_t)], axis = 1)], axis = 1)

        print('selected num of features: ', num)

        data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(new_x, dataC_y_train, test_size=0.1)

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


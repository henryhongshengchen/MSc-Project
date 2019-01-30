#imports
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
import numpy as np


#set the number of feature range and parameter C range
features_num = [5, 10, 15, 20, 30, 50, 75, 100, 250, 500, 1000, 5000]
num_costs = 20
cost_range = np.logspace(-8, 8, num_costs)


#This function performs RFE feature selection along with training Gaussian kernel SVC under 10-fold cross validation
#data_x is the gene expression data
#data_y is the labels
#return is the evaluation of model prediction accuracy
def rbf_svc_fs94(data_x, data_y):
    
    K = 10
    kf = KFold(n_splits=K)

    best_acc = []

    for num in features_num:

        data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(data_x, data_y, test_size=0.1)

        estimator = LinearSVC()
        selector = RFE(estimator, num, step=0.1)
        data_rfe = selector.fit_transform(data_x_train, np.ravel(data_y_train))
    
        print('selected num of features: ', num)
    
        cv_accur = 0
        cv_sd = 0

        accur_total = 0
        accur_list = []
            
        for train_index, test_index in kf.split(data_rfe):
            data_x_train, data_x_test = data_rfe[train_index], data_rfe[test_index]
            data_y_train, data_y_test = data_y.values[train_index], data_y.values[test_index]
            data_y_train = np.ravel(data_y_train)
            data_y_test = np.ravel(data_y_test)

            accur = np.zeros(num_costs)
            
            for i in range(num_costs):
                model = SVC(gamma = cost_range[i], kernel = 'rbf')
                model.fit(data_x_train, data_y_train)
                pred = model.predict(data_x_test)
                accur[i] = accuracy_score(data_y_test, pred)

            accur_total += np.max(accur)
            accur_list.append(np.max(accur))
            
        cv_accur = accur_total/K
        cv_sd = np.std(accur_list)
    
        print('Accuracy = ', cv_accur, 'std = ', cv_sd)
    
    best_acc.append(cv_accur)
    
    return best_acc


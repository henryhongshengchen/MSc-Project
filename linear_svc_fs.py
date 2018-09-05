#imports
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import numpy as np


#set the number of feature range and parameter C range
features_num = [5, 10, 15, 20, 30, 50, 75, 100, 250, 500, 1000, 5000]
num_costs = 20
cost_range = np.logspace(np.log10(1), np.log10(100), num_costs, base=10.0)


#This function performs RFE feature selection along with training LinearSVC under 10-fold cross validation
#data_x is the gene expression data
#data_y is the labels
#return is the evaluation of model prediction accuracy
def linear_svc_fs(data_x, data_y):
    
    K = 10
    kf = KFold(n_splits=K)

    best_acc = []

    for num in features_num:

        estimator = LinearSVC()
        selector = RFE(estimator, num, step=0.1)
        data_rfe = selector.fit_transform(data_x, np.ravel(data_y))
    
        print('selected num of features: ', num)
    
        cv_accur = np.zeros(num_costs)
        cv_sd = np.zeros(num_costs)

        for i in range(num_costs):
            accur_total = 0
            accur_list = []
            for train_index, test_index in kf.split(data_rfe):
                data_x_train, data_x_test = data_rfe[train_index], data_rfe[test_index]
                data_y_train, data_y_test = data_y.values[train_index], data_y.values[test_index]
                data_y_train = np.ravel(data_y_train)
                data_y_test = np.ravel(data_y_test)
                model = LinearSVC(C = cost_range[i])
                model.fit(data_x_train, data_y_train)
                pred = model.predict(data_x_test)
                accur = accuracy_score(data_y_test, pred)
                accur_total += accur
                accur_list.append(accur)
            cv_accur[i] = accur_total/10
            cv_sd[i] = np.std(accur_list)
    
        i = np.where(cv_accur == np.max(cv_accur))
        cost = cost_range[i[0]]
        print('C_best = ', cost, 'highest accuracy = ', cv_accur[i[0]], 'std = ', cv_sd[i[0]])
    
        best_acc.append(np.max(cv_accur[i[0]]))
    
    return best_acc


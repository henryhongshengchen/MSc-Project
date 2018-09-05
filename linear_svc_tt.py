#imports
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
import t_test as tt
from sklearn.svm import LinearSVC
import numpy as np

#set the number of feature range and parameter C range
features_num = [5, 10, 15, 20, 30, 50, 75, 100, 250, 500, 1000, 5000]
num_costs = 20
cost_range = np.logspace(np.log10(1), np.log10(100), num_costs, base=10.0)


#This function performs t-test feature selection along with training LinearSVC under 10-fold cross validation
#data_x is the gene expression data
#data_y is the labels
#return is the evaluation of model prediction accuracy
def linear_svc_tt(data_x, data_y):
    
    K = 10
    kf = KFold(n_splits=K)

    best_acc = []

    for num in features_num:

        ttt = tt.t_test(data_x, data_y, num)
        data_rfe = ttt.values
    
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
    
        j = np.where(cv_accur == np.max(cv_accur))
        cost = cost_range[j[0]]
        print('C_best = ', cost, 'highest accuracy = ', cv_accur[j[0]], 'std = ', cv_sd[j[0]])
    
        best_acc.append(np.max(cv_accur[j[0]]))
    
    return best_acc


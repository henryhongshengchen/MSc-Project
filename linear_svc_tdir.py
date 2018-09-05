#imports
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import t_test as tt
import numpy as np
import pandas as pd

#set the number of feature range and parameter C range
features_num = [5000, 1000, 500, 250, 100, 75, 50, 30, 20, 15, 10, 5]
num_costs = 20
cost_range = np.logspace(np.log10(1), np.log10(100), num_costs, base=10.0)


#This function performs t-test feature selection on direct data combination method along with training LinearSVC under 10-fold cross validation
#finC_x is the common samples of gene expression data in CBE data
#finC_y is the labels corresponding to finC_x
#finT_x is the common samples of gene expression data in TCX data
#finT_y is the labels corresponding to finT_x
#return is the evaluation of model prediction accuracy
def linear_svc_tdir(finC_x, finC_y, finT_x, finT_y):

    best_acc = []
    
    K = 10
    kf = KFold(n_splits=K)

    for num in features_num:

        print('selected num of features: ', num)

        ttc = tt.t_test(finC_x, finC_y, num)
        new_x_c = ttc.values

        ttt = tt.t_test(finT_x, finT_y, num)
        new_x_t = ttt.values

        new_x = pd.concat([pd.DataFrame(new_x_c), pd.DataFrame(new_x_t)], axis = 1)

        cv_accur = np.zeros(num_costs)
        cv_sd = np.zeros(num_costs)

        for i in range(num_costs):
            accur_total = 0
            accur_list = []
            for train_index, test_index in kf.split(new_x):
                data_x_train, data_x_test = new_x.values[train_index], new_x.values[test_index]
                data_y_train, data_y_test = finC_y.values[train_index], finC_y.values[test_index]
                data_y_train = np.ravel(data_y_train)
                data_y_test = np.ravel(data_y_test)            
                model = LinearSVC(C = cost_range[i])
                model.fit(data_x_train, data_y_train)
                pred = model.predict(data_x_test)
                accur = accuracy_score(data_y_test, pred)
                accur_total += accur
                accur_list.append(accur)
            cv_accur[i]  =accur_total/10
            cv_sd[i] = np.std(accur_list)

        j = np.where(cv_accur == np.max(cv_accur))
        cost = cost_range[j[0]]
        print('C_best = ', cost, 'highest accuracy = ', cv_accur[j[0]], 'std = ', cv_sd[j[0]])
    
        best_acc.append(np.max(cv_accur[j[0]]))
    
    return best_acc

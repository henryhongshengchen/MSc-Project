#imports
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd

#set the number of feature range and parameter C range
features_num = [5000, 1000, 500, 250, 100, 75, 50, 30, 20, 15, 10, 5]
num_costs = 20
cost_range = np.logspace(np.log10(1), np.log10(100), num_costs, base=10.0)


#This function performs RFE feature selection on resampling data combination method along with training LinearSVC under 10-fold cross validation
#finC_x is the common samples of gene expression data in CBE data
#finC_y is the labels corresponding to finC_x
#finT_x is the common samples of gene expression data in TCX data
#finT_y is the labels corresponding to finT_x
#return is the evaluation of model prediction accuracy
def linear_svc_res(finC_x, finC_y, finT_x, finT_y):

    estimator_c1 = LinearSVC()
    selector_c1 = RFE(estimator_c1, 10000, step=0.1)
    new_x_c1 = selector_c1.fit_transform(finC_x, np.ravel(finC_y))

    estimator_t1 = LinearSVC()
    selector_t1 = RFE(estimator_t1, 10000, step=0.1)
    new_x_t1 = selector_t1.fit_transform(finT_x, np.ravel(finT_y))

    new_x = pd.concat([pd.DataFrame(new_x_c1), pd.DataFrame(new_x_t1)], axis = 1)

    best_acc = []
    
    K = 10
    kf = KFold(n_splits=K)

#    data_y = pd.DataFrame(data_y)
    
    for num in features_num:

        print('selected num of features: ', num)
    
        estimator_c = LinearSVC()
        selector_c = RFE(estimator_c, num, step=0.1)
        new_x_c = selector_c.fit_transform(new_x_c1, np.ravel(finC_y))

        estimator_t = LinearSVC()
        selector_t = RFE(estimator_t, num, step=0.1)
        new_x_t = selector_t.fit_transform(new_x_t1, np.ravel(finT_y))

        estimator_ = LinearSVC()
        selector_ = RFE(estimator_, num, step=0.1)
        new_x_ = selector_.fit_transform(new_x, np.ravel(finC_y))

        new_x = pd.concat([pd.DataFrame(new_x_), pd.concat([pd.DataFrame(new_x_c), pd.DataFrame(new_x_t)], axis = 1)], axis = 1)

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


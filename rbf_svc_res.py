#imports
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
import numpy as np
import pandas as pd


#set the number of feature range and parameter C range
features_num = [5, 10, 15, 20, 30, 50, 75, 100, 250, 500, 1000, 5000]
num_costs = 20
cost_range = np.logspace(-8, 2, num_costs)


#This function performs RFE feature selection along with training Gaussian kernel SVC under 10-fold cross validation for resampling data combination
#finC_x is the common samples of gene expression data in CBE data
#finC_y is the labels corresponding to finC_x
#finT_x is the common samples of gene expression data in TCX data
#finT_y is the labels corresponding to finT_x
#return is the evaluation of model prediction accuracy
def rbf_svc_fs98(finC_x, finC_y, finT_x, finT_y):
    
    dataC_x_train, dataC_x_test, dataC_y_train, dataC_y_test = train_test_split(finC_x, finC_y, test_size=0.1)

    dataT_x_train, dataT_x_test, dataT_y_train, dataT_y_test = train_test_split(finT_x, finT_y, test_size=0.1)

    estimator_c1 = LinearSVC()
    selector_c1 = RFE(estimator_c1, 10000, step=0.1)
    new_x_c1 = selector_c1.fit_transform(dataC_x_train, np.ravel(dataC_y_train))

    estimator_t1 = LinearSVC()
    selector_t1 = RFE(estimator_t1, 10000, step=0.1)
    new_x_t1 = selector_t1.fit_transform(dataT_x_train, np.ravel(dataT_y_train))

    new_x = pd.concat([pd.DataFrame(new_x_c1), pd.DataFrame(new_x_t1)], axis = 1)

    best_acc = []
    
    K = 10
    kf = KFold(n_splits=K)

    for num in features_num:

        print('selected num of features: ', num)
    
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

        cv_accur = 0
        cv_sd = 0

        accur_total = 0
        accur_list = []
        
        for train_index, test_index in kf.split(new_x):
            data_x_train, data_x_test = new_x.values[train_index], new_x.values[test_index]
            data_y_train, data_y_test = finC_y.values[train_index], finC_y.values[test_index]
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

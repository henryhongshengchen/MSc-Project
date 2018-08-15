# In[ ]:


from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd


features_num = [5000, 1000, 500, 250, 100, 75, 50, 30, 20, 15, 10, 5]
num_costs = 20
cost_range = np.logspace(np.log10(1), np.log10(200), num_costs, base=10.0)

def linear_svc_ori(finC_x, finC_y, finT_x, finT_y):

    best_acc = []
    
    K = 10
    kf = KFold(n_splits=K)

    for num in features_num:
    
        estimator_c = LinearSVC()
        selector_c = RFE(estimator_c, num, step=0.1)
        new_x_c = selector_c.fit_transform(finC_x, np.ravel(finC_y))

        estimator_t = LinearSVC()
        selector_t = RFE(estimator_t, num, step=0.1)
        new_x_t = selector_t.fit_transform(finT_x, np.ravel(finT_y))

        new_x = pd.concat([pd.DataFrame(new_x_c), pd.DataFrame(new_x_t)], axis = 1)

        cv_accur = np.zeros(num_costs)

        for i in range(num_costs):
            accur_total = 0
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
            cv_accur[i]  =accur_total/10

        j = np.where(cv_accur == np.max(cv_accur))
        cost = cost_range[j[0]]
        print('C_best = ', cost, 'highest accuracy = ', cv_accur[j[0]])
    
        best_acc.append(np.max(cv_accur[j[0]]))
    
    return best_acc

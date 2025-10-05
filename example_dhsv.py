from beran.estimator import BeranEstimator
from beran.kernels import GaussianKernel
from beran.kernels import OpfKnnKernel, OpfKnnArcKernel
from beran.kernels import PlKnnKernel
from sksurv.ensemble import RandomSurvivalForest
from lifelines import KaplanMeierFitter, CoxPHFitter
from loader.dataset import DhsvDatasetLoader 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import logging

# Load the training data using the dataloader
ds_train = DhsvDatasetLoader('./DHSV_LDA_OEM_train.csv')
ds_test = DhsvDatasetLoader('./DHSV_LDA_OEM_test.csv')

# Gets the samples and the corresponding times
X_train,T_train = ds_train.get_data()
X_test,T_test = ds_test.get_data()

# Converts X to float32
X_train = np.array(X_train,dtype=np.float32)
X_test = np.array(X_test,dtype=np.float32)

# Sort the time and the data
T_train = np.sort(T_train)
T_test = np.sort(T_test)
X_train = X_train[np.argsort(T_train)]
X_test = X_test[np.argsort(T_test)]

# Gets the delta and covariates
delta_train = X_train[:,-1].astype(int)
X_train = X_train[:,:-1]

delta_test = X_test[:,-1].astype(int)
X_test = X_test[:,:-1]

# km = KaplanMeierFitter()
# km.fit(T_train, delta_train, label='Kaplan-Meier Estimate')

# cph_data = pd.DataFrame(X_train)
# cph_data['times'] = T_train
# cph_data['delta'] = delta_train
# cph = CoxPHFitter(penalizer=0.0001)
# cph.fit(cph_data, 'times', 'delta')

# rsf = RandomSurvivalForest()
# rsf_y_aux = np.vstack((delta_train, T_train))
# rsf_y = []
# for (indicator, time) in rsf_y_aux.T:
#     if indicator == 1:
#         rsf_y.append((True, time))
#     else:
#         rsf_y.append((False, time))
# rsf_y = np.array(rsf_y, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

# param_grid = {
#     'n_estimators': range(100, 1000, 2),
#     'max_depth': range(10, 100, 2),
#     'min_samples_split': range(2, 20),
#     'min_samples_leaf': range(1, 10),
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'bootstrap': [True, False]
# }
# randomized_search = RandomizedSearchCV(estimator=rsf, param_distributions=param_grid, n_iter=20, cv=5, random_state=1)
# randomized_search.fit(X_train, rsf_y)
# rsf = randomized_search.best_estimator_

# Initialize the estimator with the given parameters and the kernel
opf_kernel = OpfKnnKernel()
opf_estimator = BeranEstimator(T_train, X_train, delta_train, opf_kernel)

opf_arc_kernel = OpfKnnArcKernel()
opf_arc_estimator = BeranEstimator(T_train, X_train, delta_train, opf_arc_kernel)

# pl_knn_kernel = PlKnnKernel()
# pl_knn_estimator = BeranEstimator(T_train, X_train, delta_train, pl_knn_kernel)

# plots_folder = 'plots/dhsv_result_plots'
# if (not os.path.exists(plots_folder)):
#     os.makedirs(plots_folder)

# cox_results_folder = 'test/results/dhsv/cox_results'
# if (not os.path.exists(cox_results_folder)):
#     os.makedirs(cox_results_folder)

# rsf_results_folder = 'test/results/dhsv/rsf_results'
# if (not os.path.exists(rsf_results_folder)):
#     os.makedirs(rsf_results_folder)

opf_results_folder = 'test/results/dhsv/opf_results'
if (not os.path.exists(opf_results_folder)):
    os.makedirs(opf_results_folder)

opf_arc_results_folder = 'test/results/dhsv/opf_arc_results'
if (not os.path.exists(opf_arc_results_folder)):
    os.makedirs(opf_arc_results_folder)

# pl_knn_results_folder = 'test/results/dhsv/pl_knn_results'
# if (not os.path.exists(pl_knn_results_folder)):
#     os.makedirs(pl_knn_results_folder)

# benk_results_folder = 'test/results/dhsv/opf_results'
# if (not os.path.exists(opf_results_folder)):
#     os.makedirs(plots_folder)

# km_pred = km.predict(T_train)
# km_results = np.vstack([T_train, km_pred])
# np.savetxt('test/results/dhsv/sf_km.csv', km_results, delimiter=',')

# lda = {'Low': 343, 'Medium':1108,'High':2007}
# for i, (k,v) in enumerate(lda.items()):
#         new_samples = np.array([[v,'A'],[v,'B'],[v,'C']])      

# Iterates over the test samples

# rsf_results = []
opf_results = []
opf_arc_results = []
# pl_knn_results = []
# cph_results = []

for x_predict in X_test:
    x_predict = np.expand_dims(x_predict, axis=0)
    print(f'x_predict = {x_predict}')

    # rsf_pred = rsf.predict_survival_function(x_predict)
    # rsf_series = []
    # for fn in rsf_pred:
    #     plt.step(fn.x, fn(fn.x), where="post", label=f'RSF')
    #     rsf_series.append(fn(fn.x))
    # rsf_results.append(rsf_series[0])
    # print(rsf_series[0].shape)

    opf_survival_function = opf_estimator.estimate_sf(x_predict)
    opf_results.append(opf_survival_function)

    # opf_arc_sf = opf_arc_estimator.estimate_sf(x_predict)
    # opf_arc_results.append(opf_arc_sf)

    # pl_knn_sf = pl_knn_estimator.estimate_sf(x_predict)
    # pl_knn_results.append(pl_knn_sf)

    # km.plot_survival_function(ax=plt.gca(), color="yellow", linestyle=":", figsize=(20, 8))

    # cph_pred = cph.predict_survival_function(x_predict, times=T_train)
    # cph_pred.columns = ['Cox estimate']
    # cph_pred.plot(ax=plt.gca())
    # cph_pred = np.array(cph_pred['Cox estimate'])
    # cph_results.append(cph_pred)

# rsf_results = np.vstack([np.unique(T_train), rsf_results])
# np.savetxt('{}/sf_test.csv'.format(rsf_results_folder), rsf_results, delimiter=',')

opf_results = np.vstack([T_train, opf_results])
np.savetxt('{}/sf_test.csv'.format(opf_results_folder), opf_results, delimiter=',')

opf_arc_results = np.vstack([T_train, opf_arc_results])
np.savetxt('{}/sf_test.csv'.format(opf_arc_results_folder), opf_arc_results, delimiter=',')

# pl_knn_results = np.vstack([T_train, pl_knn_results])
# np.savetxt('{}/sf_test.csv'.format(pl_knn_results_folder), pl_knn_results, delimiter=',')

# cph_results = np.vstack([T_train, cph_results])
# np.savetxt('{}/sf_test.csv'.format(cox_results_folder), cph_results, delimiter=',')

    # plt.plot(T_train, opf_survival_function, label="Beran OPF-kNN")
    # plt.plot(T_train, opf_arc_sf, label="Beran OPF-kNN (Arc weights)")
    # plt.plot(T_train, pl_knn_sf, label="Beran Pl-kNN")
    # plt.xlabel("Time")
    # plt.ylabel("Survival function")
    # plt.title("OEM {}, LDA {}".format(z[1], k))
    # plt.legend()
    # plt.savefig('{}/{}_{}.png'.format(plots_folder, z[1], k))
    # plt.close()
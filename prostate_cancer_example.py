from beran.estimator import BeranEstimator
from beran.kernels import GaussianKernel
from beran.kernels import OpfKnnKernel, OpfKnnArcKernel
from beran.kernels import PlKnnKernel
from sksurv.ensemble import RandomSurvivalForest
from lifelines import KaplanMeierFitter, CoxPHFitter
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import logging

logging.disable(logging.INFO)
logging.disable(logging.DEBUG)

data = pd.read_csv('./data/Survival data_Metastatic Prostate cancer Patients at UCI.csv')
data['Time to event -Months'] = data['Time to event -Months']*30
data = data.drop(columns=['Presentation date', 'Patient time of update', 'STATUS'])
delta = data['Event (0=Alive , 1 =Death)'].to_numpy()

X_train, X_test, delta_train, delta_test = train_test_split(data, delta, test_size=0.2, random_state=1)

num_cols = ["Age", "Baseline PSA","Gleason Score"]        
ct = ColumnTransformer(
    transformers=[("num", StandardScaler(), num_cols)],
    remainder="passthrough"         
)
ct.set_output(transform="pandas")   
ct.fit(X_train)
X_train_z = ct.transform(X_train)
X_test_z = ct.transform(X_test)
print(X_train_z)

T_train = X_train_z["remainder__Time to event -Months"]
X_train_z = X_train_z.drop(columns=["remainder__Time to event -Months"])
print(X_train_z.columns.get_loc("remainder__Event (0=Alive , 1 =Death)"))
col_labels = list(X_train_z)
X_train_z = X_train_z.to_numpy()

T_test = X_test_z["remainder__Time to event -Months"]
X_test_z = X_test_z.drop(columns=["remainder__Time to event -Months"]).to_numpy()

# Sort the time and the data
T_train = np.sort(T_train)
T_test = np.sort(T_test)
X_train_z = X_train_z[np.argsort(T_train)]
X_test_z = X_test_z[np.argsort(T_test)]

# Gets the delta and covariates
delta_train = X_train_z[:,-1].astype(int)
X_train_z = X_train_z[:,:-1]

delta_test = X_test_z[:,-1].astype(int)
X_test_z = X_test_z[:,:-1]

# cph_data = pd.DataFrame(X_train_z)
# cph_data['Time to event -Months'] = T_train
# cph_data['Event (0=Alive , 1 =Death)'] = delta_train
# cph = CoxPHFitter()
# cph.fit(cph_data, 'Time to event -Months', 'Event (0=Alive , 1 =Death)')

# km = KaplanMeierFitter()
# km.fit(T_train, delta_train, label='Kaplan-Meier Estimate')

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
# randomized_search.fit(X_train_z, rsf_y)
# rsf = randomized_search.best_estimator_

# Initialize the estimator with the given parameters and the kernel
opf_kernel = OpfKnnKernel()
opf_estimator = BeranEstimator(T_train, X_train_z, delta_train, opf_kernel)

# opf_arc_kernel = OpfKnnArcKernel()
# opf_arc_estimator = BeranEstimator(T_train, X_train_z, delta_train, opf_arc_kernel)

# pl_knn_kernel = PlKnnKernel()
# pl_knn_estimator = BeranEstimator(T_train, X_train_z, delta_train, pl_knn_kernel)

# plots_folder = 'prostate_cancer_plots'
# if (not os.path.exists(plots_folder)):
#     os.makedirs(plots_folder)

# cox_results_folder = 'test/results/heart_failure/cox_results'
# if (not os.path.exists(cox_results_folder)):
#     os.makedirs(cox_results_folder)

# rsf_results_folder = 'test/results/heart_failure/rsf_results'
# if (not os.path.exists(rsf_results_folder)):
#     os.makedirs(rsf_results_folder)

opf_results_folder = 'test/results/heart_failure/opf_results'
if (not os.path.exists(opf_results_folder)):
    os.makedirs(opf_results_folder)

# opf_arc_results_folder = 'test/results/heart_failure/opf_arc_results'
# if (not os.path.exists(opf_arc_results_folder)):
#     os.makedirs(opf_arc_results_folder)

# pl_knn_results_folder = 'test/results/heart_failure/pl_knn_results'
# if (not os.path.exists(pl_knn_results_folder)):
#     os.makedirs(pl_knn_results_folder)

# rsf_results = []
opf_results = []
# opf_arc_results = []
# pl_knn_results = []
# cph_results = []

for x_predict in X_test_z:
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

# opf_arc_results = np.vstack([T_train, opf_arc_results])
# np.savetxt('{}/sf_test.csv'.format(opf_arc_results_folder), opf_arc_results, delimiter=',')

# pl_knn_results = np.vstack([T_train, pl_knn_results])
# np.savetxt('{}/sf_test.csv'.format(pl_knn_results_folder), pl_knn_results, delimiter=',')

# cph_results = np.vstack([T_train, cph_results])
# np.savetxt('{}/sf_test.csv'.format(cox_results_folder), cph_results, delimiter=',')

# sampleRNG = np.random.default_rng(1)
# new_samples = X_test[sampleRNG.choice(len(X_test), 9)]
# i = 0
# for z in new_samples:
#     x_predict = z.reshape(1,-1)
#     print(x_predict)
#     i += 1

#     rsf_pred = rsf.predict_survival_function(x_predict)
#     for fn in rsf_pred:
#         plt.step(fn.x, fn(fn.x), where="post", label=f'RSF')

#     opf_survival_function = opf_estimator.estimate_sf(x_predict)
#     opf_arc_sf = opf_arc_estimator.estimate_sf(x_predict)
#     pl_knn_sf = pl_knn_estimator.estimate_sf(x_predict)

#     km.plot_survival_function(ax=plt.gca(), color="yellow", linestyle=":", figsize=(20, 8))

#     cph_pred = cph.predict_survival_function(x_predict)
#     cph_pred.columns = ['Cox estimate']
#     cph_pred.plot(ax=plt.gca())
    
#     plt.plot(T_train, opf_survival_function, label="Beran OPF-kNN")
#     plt.plot(T_train, opf_arc_sf, label="Beran OPF-kNN (Arc weights)")
#     plt.plot(T_train, pl_knn_sf, label="Beran Pl-kNN")
#     plt.table(z.reshape(1,-1), colLabels=col_labels, bbox=[0.0, -0.3, 1, 0.2])
#     plt.subplots_adjust(bottom = 0.3)
#     plt.xlabel("Time")
#     plt.ylabel("Survival function")
#     plt.title(f"Plot {i}")
#     plt.legend()
#     plt.savefig('{}/{}.png'.format(plots_folder, i))
#     plt.close()
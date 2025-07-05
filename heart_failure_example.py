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
import random
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import logging

logging.disable(logging.INFO)
logging.disable(logging.DEBUG)

data = pd.read_csv('./data/S1Data.csv')
delta = data['Event'].to_numpy()
X_train, X_test, delta_train, delta_test = train_test_split(data, delta, test_size=0.2, random_state=1)

T_train = X_train['TIME']
X_train = X_train.drop(columns='TIME')
col_labels = list(X_train)
X_train = X_train.to_numpy()

T_test = X_test['TIME']
X_test = X_test.drop(columns='TIME').to_numpy()

# Sort the time and the data
T_train = np.sort(T_train)
T_test = np.sort(T_test)
X_train = X_train[np.argsort(T_train)]
X_test = X_test[np.argsort(T_test)]

# Gets the delta and covariates
delta_train = X_train[:,1].astype(int)
print(delta_train)
X_train = np.delete(X_train, 1, axis=1)

delta_test = X_test[:,1].astype(int)
X_test = np.delete(X_test, 1, axis=1)

cph_data = pd.DataFrame(X_train)
cph_data['TIME'] = T_train
cph_data['Event'] = delta_train
cph = CoxPHFitter()
cph.fit(cph_data, 'TIME', 'Event')

km = KaplanMeierFitter()
km.fit(T_train, delta_train, label='Kaplan-Meier Estimate')

rsf = RandomSurvivalForest()
rsf_y_aux = np.vstack((delta_train, T_train))
rsf_y = []
for (indicator, time) in rsf_y_aux.T:
    if indicator == 1:
        rsf_y.append((True, time))
    else:
        rsf_y.append((False, time))
rsf_y = np.array(rsf_y, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

param_grid = {
    'n_estimators': range(100, 1000, 2),
    'max_depth': range(10, 100, 2),
    'min_samples_split': range(2, 20),
    'min_samples_leaf': range(1, 10),
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}
randomized_search = RandomizedSearchCV(estimator=rsf, param_distributions=param_grid, n_iter=20, cv=5)
randomized_search.fit(X_train, rsf_y)
rsf = randomized_search.best_estimator_

# Initialize the estimator with the given parameters and the kernel
opf_kernel = OpfKnnKernel()
opf_estimator = BeranEstimator(T_train, X_train, delta_train, opf_kernel)

opf_arc_kernel = OpfKnnArcKernel()
opf_arc_estimator = BeranEstimator(T_train, X_train, delta_train, opf_arc_kernel)

pl_knn_kernel = PlKnnKernel()
pl_knn_estimator = BeranEstimator(T_train, X_train, delta_train, pl_knn_kernel)

plots_folder = 'plots/heart_failure_plots'
if (not os.path.exists(plots_folder)):
    os.makedirs(plots_folder)

new_samples = X_test[np.random.choice(len(X_test), 9)]
i = 0
for z in new_samples:
    x_predict = z.reshape(1,-1)
    print(x_predict)
    i += 1

    rsf_pred = rsf.predict_survival_function(x_predict)
    for fn in rsf_pred:
        plt.step(fn.x, fn(fn.x), where="post", label=f'RSF')

    opf_survival_function = opf_estimator.estimate_sf(x_predict)
    opf_arc_sf = opf_arc_estimator.estimate_sf(x_predict)
    pl_knn_sf = pl_knn_estimator.estimate_sf(x_predict)

    km.plot_survival_function(ax=plt.gca(), color="yellow", linestyle=":", figsize=(20, 8))

    cph_pred = cph.predict_survival_function(x_predict)
    cph_pred.columns = ['Cox estimate']
    cph_pred.plot(ax=plt.gca())
    
    plt.plot(T_train, opf_survival_function, label="Beran OPF-kNN")
    plt.plot(T_train, opf_arc_sf, label="Beran OPF-kNN (Arc weights)")
    plt.plot(T_train, pl_knn_sf, label="Beran Pl-kNN")
    plt.table(z.reshape(1,-1), colLabels=col_labels, bbox=[0.0, -0.3, 1, 0.2])
    plt.subplots_adjust(bottom = 0.3)
    plt.xlabel("Time")
    plt.ylabel("Survival function")
    plt.title(f"Plot {i}")
    plt.legend()
    plt.savefig('{}/{}.png'.format(plots_folder, i))
    plt.close()
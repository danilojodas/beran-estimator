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

data = pd.read_csv('./data/ai4i2020.csv')
data = data.drop(columns=['UDI', 'Product ID'])
map = {'L': 1, 'M': 2, 'H': 3}
data['Type'] = data['Type'].replace(map)
data['Process temperature [K]'] = (data['Process temperature [K]'] - 273.15).round(2)
data['Air temperature [K]'] = (data['Air temperature [K]'] - 273.15).round(2)

X_train, X_test = train_test_split(data, test_size=0.2, random_state=1)

T_train = X_train['Tool wear [min]']
X_train = X_train.drop(columns='Tool wear [min]')
col_labels = list(X_train)
X_train = X_train.to_numpy()

T_test = X_test['Tool wear [min]']
X_test = X_test.drop(columns='Tool wear [min]').to_numpy()

# Sort the time and the data
T_train = np.sort(T_train)
T_test = np.sort(T_test)
X_train = X_train[np.argsort(T_train)]
X_test = X_test[np.argsort(T_test)]

cph_data = pd.DataFrame(X_train)
cph_data['Tool wear [min]'] = T_train

new_samples = X_test[np.random.choice(len(X_test), 9)]
cph = CoxPHFitter(penalizer=0.0001)
km = KaplanMeierFitter()
rsf = RandomSurvivalForest()
opf_kernel = OpfKnnKernel()
opf_arc_kernel = OpfKnnArcKernel()
pl_knn_kernel = PlKnnKernel()

param_grid = {
        'n_estimators': range(100, 1000, 2),
        'max_depth': range(10, 100, 2),
        'min_samples_split': range(2, 20),
        'min_samples_leaf': range(1, 10),
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False]
    }
randomized_search = RandomizedSearchCV(estimator=rsf, param_distributions=param_grid, n_iter=20, cv=5)

for j in range(7, 12):

    plots_folder = 'plots/{}_failure_plots'.format(data.iloc[:,j].name)
    if (not os.path.exists(plots_folder)):
        os.makedirs(plots_folder)

    # Gets the delta and covariates
    delta_train = X_train[:,j].astype(int)
    cur_train = np.delete(X_train, j, axis=1)

    delta_test = X_test[:,j].astype(int)
    cur_test = np.delete(X_test, j, axis=1)

    cur_samples = np.delete(new_samples, j, axis=1)

    cph_data['Event'] = delta_train
    cph.fit(cph_data, 'Tool wear [min]', 'Event')

    km.fit(T_train, delta_train, label='Kaplan-Meier Estimate')

    rsf_y_aux = np.vstack((delta_train, T_train))
    rsf_y = []
    for (indicator, time) in rsf_y_aux.T:
        if indicator == 1:
            rsf_y.append((True, time))
        else:
            rsf_y.append((False, time))
    rsf_y = np.array(rsf_y, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    randomized_search.fit(cur_train, rsf_y)
    rsf = randomized_search.best_estimator_

    # Initialize the estimator with the given parameters and the kernel
    opf_estimator = BeranEstimator(T_train, cur_train, delta_train, opf_kernel)
    opf_arc_estimator = BeranEstimator(T_train, cur_train, delta_train, opf_arc_kernel)
    pl_knn_estimator = BeranEstimator(T_train, cur_train, delta_train, pl_knn_kernel)

    for i, z in enumerate(cur_samples):
        x_predict = z.reshape(1,-1)
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
        plt.savefig('plots/{}/{}.png'.format(plots_folder, i))
        plt.close()
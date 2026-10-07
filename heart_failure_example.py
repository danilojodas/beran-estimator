from beran.estimator import BeranEstimator
from beran.kernels import GaussianKernel
from beran.kernels import OpfKnnKernel, OpfKnnArcKernel
from beran.kernels import PlKnnKernel
from BENK import BENK, BENKDataGenerator, train_model, tau_loss
from sksurv.ensemble import RandomSurvivalForest
from lifelines import KaplanMeierFitter, CoxPHFitter
from BENK_aux.pytorch_survival import arrs_to_torch_dev

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import torch
import datetime
import logging

logging.disable(logging.INFO)
logging.disable(logging.DEBUG)

data = pd.read_csv('./data/S1Data.csv')
delta = data['Event'].to_numpy()
X_train, X_test, delta_train, delta_test = train_test_split(data, delta, test_size=0.2, random_state=1)

num_cols = ["Age","Ejection.Fraction","Sodium","Creatinine","Pletelets","CPK"]        
ct = ColumnTransformer(
    transformers=[("num", StandardScaler(), num_cols)],
    remainder="passthrough"         
)
ct.set_output(transform="pandas")   
ct.fit(X_train)
X_train_z = ct.transform(X_train)
X_test_z = ct.transform(X_test)

T_train = X_train_z['remainder__TIME']
X_train_z = X_train_z.drop(columns=["remainder__TIME"])
col_labels = list(X_train)
X_train_z = X_train_z.to_numpy()

T_test = X_test_z['remainder__TIME']
X_test_z = X_test_z.drop(columns=["remainder__TIME"])
X_test_z = X_test_z.to_numpy()

# Sort the time and the data
T_train = np.sort(T_train)
T_test = np.sort(T_test)
X_train_z = X_train_z[np.argsort(T_train)]
X_test_z = X_test_z[np.argsort(T_test)]

# Gets the delta and covariates
delta_train = X_train_z[:,6].astype(int)
X_train_z = np.delete(X_train_z, 6, axis=1).astype(np.float32)

delta_test = X_test_z[:,6].astype(int)
X_test_z = np.delete(X_test_z, 6, axis=1).astype(np.float32)
# print(f'X_test.iloc[18] = {X_test.iloc[18].to_numpy()}')

# cph_data = pd.DataFrame(X_train_z)
# cph_data['TIME'] = T_train
# cph_data['Event'] = delta_train
# cph = CoxPHFitter()
# cph.fit(cph_data, 'TIME', 'Event')

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
# opf_kernel = OpfKnnKernel()
# opf_estimator = BeranEstimator(T_train, X_train_z, delta_train, opf_kernel)

# opf_arc_kernel = OpfKnnArcKernel()
# opf_arc_estimator = BeranEstimator(T_train, X_train_z, delta_train, opf_arc_kernel)

# pl_knn_kernel = PlKnnKernel()
# pl_knn_estimator = BeranEstimator(T_train, X_train_z, delta_train, pl_knn_kernel)

train_ratio = 0.7
epochs = 200
batch_size = 64
subset_numbers = [1, 9, 10, 13, 14, 20, 25, 26, 35, 39]
subset_sizes = [28, 69, 49, 55, 85, 9, 65, 30, 54, 74]
feat_num = X_train_z.shape[1]

benk = BENK(feat_num)
T_predict = np.concatenate(([0], T_train))

for j in range(len(subset_sizes)):
    n = subset_sizes[j]
    N = subset_numbers[j]
    data_generator = BENKDataGenerator(X_train_z, T_train, delta_train, batch_size, n, N)
    optimizer = torch.optim.AdamW(benk.parameters(), 0.001)
    train_model(data_generator, benk, tau_loss, optimizer, epochs)

    benk_results_folder = 'test/results/heart_failure/benk_{}_{}_results'.format(N, n)
    if (not os.path.exists(benk_results_folder)):
        os.makedirs(benk_results_folder)

    benk_results = []

    for x_predict in X_test_z:
        x_predict = np.expand_dims(x_predict, axis=0)
        print(f'x_predict = {x_predict}')

        benk_args = arrs_to_torch_dev(X_train_z[None, ...], T_train[None, ...], delta_train[None, ...], x_predict, T_predict[None, ...])
        sf_benk = benk.predict_in_points(*benk_args)
        benk_results.append(sf_benk.flatten())

    benk_results = np.vstack([T_predict, benk_results])
    np.savetxt('{}/sf_test'.format(benk_results_folder), benk_results, delimiter=',')

    benk_results = []

    sampleRNG = np.random.default_rng(1)
    new_samples = X_test_z[sampleRNG.choice(len(X_test_z), 9)]
    new_samples_idx = sampleRNG.choice(len(X_test_z), 9)

    i = 0
    for z in new_samples:
        x_predict = z.reshape(1,-1)
        print(x_predict)
        i += 1

        benk_args = arrs_to_torch_dev(X_train_z[None, ...], T_train[None, ...], delta_train[None, ...], x_predict, T_predict[None, ...])
        sf_benk = benk.predict_in_points(*benk_args)
        benk_results.append(sf_benk.flatten())

    for i in range(len(new_samples)):

        benk_results_aux = np.vstack([T_predict, benk_results[i]])
        np.savetxt(f'{benk_results_folder}/sf_{i+1}.csv', benk_results_aux, delimiter=',')
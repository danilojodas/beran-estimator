from beran.estimator import BeranEstimator
from beran.kernels import GaussianKernel
from beran.kernels import OpfKnnKernel
from beran.kernels import PlKnnKernel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import logging
logging.disable(logging.INFO)
logging.disable(logging.DEBUG)

# Load the pumps dataset
data = pd.read_csv('./examples/pumps.csv')
data = data.drop(columns=['id'])

# Normalizes the pressure_in_bar column (z-score)
data['pressure_in_bar'] = (data['pressure_in_bar'] - data['pressure_in_bar'].mean()) / data['pressure_in_bar'].std()

# Splits the data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=1)

T_train = train_data['time'].to_numpy()
delta_train = train_data['event'].to_numpy().astype(int)
X_train = train_data.drop(columns=['time', 'event']).to_numpy(dtype=np.float32)

T_test = test_data['time'].to_numpy()
delta_test = test_data['event'].to_numpy().astype(int)
X_test = test_data.drop(columns=['time', 'event']).to_numpy(dtype=np.float32)

# Sort the time and the data
sort_idx_train = np.argsort(T_train)
T_train = T_train[sort_idx_train]
X_train = X_train[sort_idx_train]
delta_train = delta_train[sort_idx_train]

sort_idx_test = np.argsort(T_test)
T_test = T_test[sort_idx_test]
X_test = X_test[sort_idx_test]
delta_test = delta_test[sort_idx_test]

print(f'X_train = {X_train}')
print(f'X_test = {X_test}')
print(f'delta_train = {delta_train}')
print(f'delta_test = {delta_test}')

# Initialize the estimator with the OPF-based kernel
kernel = OpfKnnKernel()
estimator_opf = BeranEstimator(T_train, X_train, delta_train, kernel)

# Initialize the estimator with PL-kNN
kernel = PlKnnKernel()
estimator_plknn = BeranEstimator(T_train, X_train, delta_train, kernel)

survival_function_opf = estimator_opf.estimate_sf(X_test)
survival_function_plknn = estimator_plknn.estimate_sf(X_test)

plt.subplot(1,2,1)
plt.plot(T_train, survival_function_plknn)
plt.xlabel("Time")
plt.ylabel("Survival function")
plt.title("Estimated Survival Function - PL-kNN")

plt.subplot(1,2,2)
plt.plot(T_train, survival_function_opf)
plt.xlabel("Time")
plt.ylabel("Survival function")
plt.title("Estimated Survival Function - OPF")

plt.show()

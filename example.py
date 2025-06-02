from beran.estimator import BeranEstimator
from beran.kernels import GaussianKernel
from beran.kernels import OpfKnnKernel
from beran.kernels import PlKnnKernel
from loader.dataset import DhsvDatasetLoader
import numpy as np
import matplotlib.pyplot as plt
import logging
logging.disable(logging.INFO)
logging.disable(logging.DEBUG)

# Example usage
times = np.array([1, 2, 3, 4, 5])
covariates = np.array([[0.5, 0.2], [0.6, 0.3], [0.7, 0.4], [0.8, 0.5], [0.9, 0.6]])  # Two covariates per observation
censoring = np.array([1, 1, 1, 0, 1])
target_covariate = [0.6, 0.35]  # Target covariate (two-dimensional)
bandwidth = [0.1, 0.1]  # Bandwidth for each covariate

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

print(f'X_train = {X_train}')
print(f'X_test = {X_test}')
print(f'delta_train = {delta_train}')
print(f'delta_test = {delta_test}')

# Initialize the estimator with the given parameters and the kernel
kernel = OpfKnnKernel()
estimator = BeranEstimator(T_train, X_train, delta_train, kernel)

survival_function = estimator.estimate_sf(X_test)
print("Estimated survival function:", survival_function)

plt.plot(T_train, survival_function)
plt.xlabel("Time")
plt.ylabel("Survival function")
plt.title("Estimated Survival Function")
plt.show()
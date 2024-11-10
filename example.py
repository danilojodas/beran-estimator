from beran.estimator import BeranEstimator
from beran.kernels import GaussianKernel
import numpy as np
import matplotlib.pyplot as plt

# Example usage
times = np.array([1, 2, 3, 4, 5])
covariates = np.array([[0.5, 0.2], [0.6, 0.3], [0.7, 0.4], [0.8, 0.5], [0.9, 0.6]])  # Two covariates per observation
censoring = np.array([1, 1, 1, 0, 1])
target_covariate = [0.6, 0.35]  # Target covariate (two-dimensional)
bandwidth = [0.1, 0.1]  # Bandwidth for each covariate

# Initialize the estimator with the given parameters and the kernel
kernel = GaussianKernel(bandwidth)
estimator = BeranEstimator(times, covariates, censoring, kernel)

survival_function = estimator.estimate_sf(target_covariate)
print("Estimated survival function:", survival_function)

plt.plot(times, survival_function)
plt.xlabel("Time")
plt.ylabel("Survival function")
plt.title("Estimated Survival Function")
plt.show()
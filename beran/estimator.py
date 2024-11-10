import numpy as np
import matplotlib.pyplot as plt

class BeranEstimator:

    def __init__(self, times, covariates, censoring, kernel):
        """
        Initialize the Beran estimator.

        Parameters
        ----------
        times : array
            The times (e.g. failure or censoring times).
        covariates : array
            The covariates associated with the times.
        censoring : array
            A boolean array indicating whether the time is a failure (True) or
            censoring (False).
        kernel : object
            The kernel object to use for the estimation.
        """
        self.times = times
        self.covariates = covariates
        self.censoring = censoring
        self.kernel = kernel
        
    # Beran estimator for two covariates
    def estimate_sf(self, target_covariate):
        """
        Compute the estimated survival function for a given target covariate using the Beran estimator.

        Parameters
        ----------
        target_covariate : array
            The target covariate values to estimate the survival function.

        Returns
        -------
        survival_function : array
            The estimated survival function values for each time point in self.times.
        """
        
        n = len(self.times)
        survival_function = np.zeros(n)

        for i in range(n):
            # Find individuals still at risk at time t_i
            at_risk = self.times >= self.times[i]
            
            # Compute kernel weights based on covariates
            #kernel_weights = np.array([multivariate_kernel(target_covariate, covariates[j], bandwidth) for j in range(n)])
            kernel_weights = [self.kernel.call(target_covariate, self.covariates[j]) for j in range(n)]
            
            # Sum of kernel weights for individuals still at risk
            weighted_sum_at_risk = np.sum(kernel_weights * at_risk)

            # Compute hazard for time t_i
            hazard = np.sum(kernel_weights * self.censoring * (self.times == self.times[i])) / weighted_sum_at_risk

            # Update survival function using cumulative product
            if i == 0:
                survival_function[i] = 1 - hazard
            else:
                survival_function[i] = survival_function[i-1] * (1 - hazard)

        return survival_function
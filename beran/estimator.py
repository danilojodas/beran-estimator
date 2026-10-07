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
        kernel_weights = []

        for i in range(n):
            # Find individuals still at risk at time t_i
            at_risk = self.times >= self.times[i]

            #Train the kernel with those individuals at risk at time t_i
            X_train = self.covariates[at_risk]
            y_train = self.censoring[at_risk]

            # Compute kernel weights
            weight = self.kernel.call(target_covariate, X_train, y_train)

            # Store kernel weights
            kernel_weights.append(weight)

        kernel_weights = np.array(kernel_weights)

        for i in range(n):
            # Find individuals still at risk at time t_i
            at_risk = self.times >= self.times[i]
            
            # Sum of kernel weights for individuals still at risk
            weighted_sum_at_risk = np.sum(kernel_weights * at_risk)

            # Compute hazard for time t_i
            hazard = kernel_weights[i] * self.censoring[i] / (weighted_sum_at_risk+1e-10)

            # Update survival function using cumulative product
            if i == 0:
                survival_function[i] = 1 - hazard
            else:
                survival_function[i] = survival_function[i-1] * (1 - hazard)

        return survival_function
from scipy.stats import norm

class GaussianKernel:
    def __init__(self, h):
        """
        Initialize the kernel with the bandwidth.
        
        Parameters
        ----------
        h : array
            The bandwidths of the kernels.
        """
        self.h = h
    
    # Gaussian kernel for one dimension
    def gaussian_kernel(self,x, xi, h):
        """
        Gaussian kernel for one dimension.
        
        Parameters
        ----------
        x : float
            Value to evaluate the kernel at.
        xi : float
            Center of the kernel.
        h : float
            Bandwidth of the kernel.
        
        Returns
        -------
        float
            The value of the kernel at x.
        """
        return norm.pdf((x - xi) / h)

    # Multivariate kernel as a product of univariate Gaussian kernels
    def multivariate_kernel(self, x, xi):
        """
        Multivariate kernel as a product of univariate Gaussian kernels.

        Parameters
        ----------
        x : list of float
            Values to evaluate the kernel at.
        xi : list of float
            Centers of the kernels.

        Returns
        -------
        float
            The product of the univariate Gaussian kernels evaluated at x.
        """
        kernel_product = 1.0
        for i in range(len(x)):
            kernel_product *= self.gaussian_kernel(x[i], xi[i], self.h[i])
        
        return kernel_product
    
    # Function to call the multivariate kernel when instantiating the class
    def call(self, x, xi):
        """
        Call the multivariate kernel function for the given inputs.

        Parameters
        ----------
        x : list of float
            Values to evaluate the kernel at.
        xi : list of float
            Centers of the kernels.

        Returns
        -------
        float
            The product of the univariate Gaussian kernels evaluated at x.
        """
        return self.multivariate_kernel(x, xi)
    
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
    def call(self, x, xi, y=None):
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
    
class PlKnnKernel:
    def __init__(self):
        pass
    
    def call(self, x, xi, y=None):
        # OBS: PRECISA IMPORTAR A CLASSE DO PL-kNN
        
        # Criar um objeto da classe PL-kNN

        # Treinar o PL-kNN com X e y

        # Chamar o método predict do PL-kNN na variável xi

        # Recuperar os vizinhos mais proximos e retornar a soma dos seus pesos
        pass
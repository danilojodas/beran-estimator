from scipy.stats import norm
from .pl_nn import PlNearestNeighbors
from opfython.models.knn_supervised import KNNSupervisedOPF
from sklearn.model_selection import train_test_split
import opfython.utils.constants as c

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

    def get_nn_weights(self, x, xi, y):
        """
        Recover the weights of the nearest neighbors calculated by PL-kNN.

        Parameters
        ----------
        x: list of float
            Values to evaluate the kernel at.
        xi: list of float
            Centers of the kernels.
        y: list of float
            Censoring indicators for each kernel;
        
        Returns
        -------
        list of float
            The weights of the nearest neighbors of x
        """
        plnn = PlNearestNeighbors()
        plnn.fit(xi, y)
        plnn.predict(x)
        nn_weights = plnn.nearest_neighbors[:,-2]
        
        return nn_weights

    def pl_knn_kernel(self, x, xi, y):
        """
        PL-kNN kernel.

        Parameters
        ----------
        x: list of float
            Values to evaluate the kernel at.
        xi: list of float
            Centers of the kernels.
        y: list of float
            Censoring indicators for each kernel;
        
        Returns
        -------
        float
            The sum of the weights of the nearest neighbors of x
        """
        nn_weights = self.get_nn_weights(x, xi, y)
        sum = 0.0
        for i in range(len(nn_weights)):
            sum += nn_weights[i]

        return sum
    
    def call(self, x, xi, y=None):
        """
        Call the pl-kNN kernel function for the given inputs.

        Parameters
        ----------
        x: list of float
            Values to evaluate the kernel at.
        xi: list of float
            Centers of the kernels.
        y: list of float
            Censoring indicators for each kernel;
        
        Returns
        -------
        float
            The sum of the weights of the nearest neighbors of x
        """
        return self.pl_knn_kernel(x, xi, y)
    
class OpfKnnKernel:
    def __init__(self):
        pass

    def get_opf_costs(self, x, xi, y=None):
        costs = []
        if len(xi) > 2:
            x_train, x_val, y_train, y_val = train_test_split(xi, y, test_size=0.25, random_state=1)
            print(f'x_train: {x_train.shape}')
            print(f'x_val: {x_val.shape}')
            print(f'y_train: {y_train.shape}')
            print(f'y_val: {y_val.shape}')

            opf = KNNSupervisedOPF()
            opf.fit(x_train, y_train, x_val, y_val)
            print('fitou')
            opf.predict(x)
            subgraph = opf.subgraph

            for k in range(subgraph.best_k):
                if opf.distances[k] != c.FLOAT_MAX:
                    neighbour = int(opf.neighbours_idx[k])
                    costs.append(subgraph.nodes[neighbour].cost)

        return costs


    def opf_knn_kernel(self, x, xi, y=None):
        opf_costs = self.get_opf_costs(x, xi, y)
        sum = 0.0
        count = 0.00001

        for cost in opf_costs:
            sum += cost
            count += 1
        
        return sum/count
    
    def call(self, x, xi, y=None):
        return self.opf_knn_kernel(x, xi, y)
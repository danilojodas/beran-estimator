import random
import numpy as np

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

class DhsvDatasetLoader():
    """A class that implements the instructions to load and return all samples of the DHSV dataset.

    Args:
        file_name (str): Name of the file to be loaded.
        normalize (bool): Indicates whether to apply the normalization. Default is True.
    """    
    def __init__(self,file_name,normalize=True):
        # Initializes the class properties
        self.mean = None
        self.std = None
        self.data = None
        self.oem = None
        self.labelencoder = None
        self.onehotencoder = None
        self.normalize = normalize

        # Loads the data
        data = np.loadtxt(file_name,delimiter=',',dtype=object)

        # Gets the features separately
        # 0: LDA, 1: time, 2: delta, 3: OEM
        lda = data[:,0].reshape(-1,1).astype(np.float32)
        delta = data[:,2].reshape(-1,1).astype(int)
        oem = data[:,3]
        time = data[:,1].reshape(-1,1).astype(int)

        # Keeps non-encoded OEM
        self.oem = oem

        # Transforms OEM into numbers
        self.labelencoder = LabelEncoder()
        oem = self.labelencoder.fit_transform(oem).reshape(-1,1)

        # Performs the One Hot Encoding of OEM
        self.onehotencoder = OneHotEncoder()
        oem = self.onehotencoder.fit_transform(oem).toarray()

        # Normalizes LDA using the mean and standard deviation of the data
        if (normalize):
            #lda = StandardScaler().fit_transform(lda)
            self.mean = np.mean(lda)
            self.std = np.std(lda)
            lda = (lda - self.mean) / self.std

        # Concatenates all columns together (LDA, OEM, delta time)
        self.data = np.concatenate((lda,oem,delta,time),axis=1)

    def get_data(self,):
        """A function that returns the features and the corresponding labels.

        Returns:
            tuple: The features and labels as separated arrays.
        """
        # Returns LDA, delta and OEM as features, while time is the label
        features = self.data[:,:-1]
        time = self.data[:,-1]

        return features, time
    
    def encode_feature_vector(self,x):
        # Gets the LDA and OEM
        lda = x[:,0].astype(np.float32)
        oem = x[:,1]

        # Normalizes LDA and encodes OEM
        lda = (lda - self.mean) / self.std
        oem = self.labelencoder.transform(oem).reshape(-1,1)
        oem = self.onehotencoder.transform(oem).toarray()

        # Concatenates the normalized and the encoded features
        new_x = np.expand_dims(np.insert(oem,0,lda),axis=0).astype(np.float32)

        return new_x
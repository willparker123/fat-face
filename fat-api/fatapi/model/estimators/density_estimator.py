
from typing import Callable
from fatapi.helpers import check_type
import numpy as np

class DensityEstimator(object):
    """
    Abstract class used for KDE and GS kernels to get density scores from data
    
    Parameters
    ----------
    distance_function() : (X: np.ndarray, Y: np.npdarray)
        Calculates distance between X and Y
        -- Default is Euclidean distance
    transformation_function() : (X: np.ndarray)
        Transforms X
        -- Default is -np.log(X)
    
    Methods
    -------
    fit(X: np.ndarray) : np.ndarray
        Method for fitting density estimator to X
    score(X: np.ndarray, K?: int) : np.ndarray
        Method for calculating a score after transforming x and comparing against distances of X
    score_samples(X: np.ndarray, K?: int) : np.ndarray
        Method for calculating a score when predicting X and comparing with Y
    """
    def __init__(self, classifier, **kwargs) -> None:
        if kwargs.get("distance_function"): 
            self._distance_function = check_type(kwargs.get("distance_function"), Callable, "__init__")
        else:
            self._distance_function = lambda x, y: np.linalg.norm(x.reshape(-1, 1) - y.reshape(-1, 1))
            
        if kwargs.get("transformation_function"): 
            self._transformation_function = check_type(kwargs.get("transformation_function"), Callable, "__init__")
        else:
            self._transformation_function = lambda x: -np.log(x)
        
    def _fit(self, X):
        self.X = X
        self.n_samples = X.shape[0]   
    
    def score(self, X: np.ndarray, K: int=10):
        distances = np.zeros(self.n_samples)
        for idx in range(self.n_samples):
            distances[idx] = self.distance_function(X, self.X[idx, :])
        return self.transformation_function(np.sort(distances)[K])
    
    def score_samples(self, X: np.ndarray, K: int=10):
        n_samples_test = X.shape[0]
        if n_samples_test == 1:
            return self.score_samples_single(X)
        else:
            scores = np.zeros((n_samples_test, 1))
            for idx in range(n_samples_test):
                scores[idx] = self.score(X[idx, :], K)
            return scores

    @property
    def distance_function(self) -> Callable:
        """
        Sets and changes the distance_function method of the density estimator
        -------
        Callable
        """
        
        return self._distance_function

    @distance_function.setter
    def distance_function(self, distance_function) -> None:
        self._distance_function = check_type(distance_function, Callable, "distance_function.setter")
        
    @property
    def transformation_function(self) -> Callable:
        """
        Sets and changes the transformation_function method of the density estimator
        -------
        Callable
        """
        
        return self._transformation_function

    @transformation_function.setter
    def transformation_function(self, transformation_function) -> None:
        self._transformation_function = check_type(transformation_function, Callable, "transformation_function.setter")
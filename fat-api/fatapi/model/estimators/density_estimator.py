
from typing import Callable, Optional
from fatapi.helpers import check_type
import numpy as np

class DensityEstimator(object):
    """
    Abstract class used for kde and GS kernels to get density scores from data
    
    Parameters
    ----------
    distance_function() : (X: np.ndarray, Y: np.npdarray) -> float
        Calculates distance between X and Y
        -- Default is Euclidean distance
    transformation_function() : (X: np.ndarray) -> np.ndarray
        Transforms X
        -- Default is -np.log(X)
    
    Methods
    -------
    fit() : (X: np.ndarray)
        Method for fitting density estimator to X
    score() : (X: np.ndarray, K?: int) -> np.ndarray
        Method for calculating a score after transforming x and comparing against distances of X
    score_samples() : (X: np.ndarray, K?: int) -> np.ndarray
        Method for calculating a score when predicting X and comparing with Y
    """
    def __init__(self, **kwargs) -> None:
        if kwargs.get("estimator"):
            self.estimator = kwargs.get("estimator")
            try:
                if callable(getattr(self.estimator, "fit")):
                    pass
            except:
                raise ValueError("Invalid argument in __init__: estimator does not have function fit")
            try:
                if callable(getattr(self.estimator, "score_samples")):
                    pass
            except:
                raise ValueError("Invalid argument in __init__: estimator does not have function score_samples")
            self._fit = self.estimator.fit
            self._score_samples = self.estimator.score_samples
            self._score = None
        else:
            self._fit = self.base_fit
            self._score = self.base_score
            self._score_samples = self.base_score_samples

        if kwargs.get("distance_function"): 
            self._distance_function = check_type(kwargs.get("distance_function"), "__init__", Callable[[np.ndarray, np.ndarray], float])
        else:
            self._distance_function = lambda x, y: np.linalg.norm(x.reshape(-1, 1) - y.reshape(-1, 1))
            
        if kwargs.get("transformation_function"): 
            self._transformation_function = check_type(kwargs.get("transformation_function"), "__init__", Callable[[np.ndarray], np.ndarray])
        else:
            self._transformation_function = lambda x: -np.log(x)
        
    def base_fit(self, X: np.ndarray) -> None:
        self.X = X
        self.n_samples = X.shape[0]   
    
    def base_score(self, X: np.ndarray, K: int=10) -> np.ndarray:
        n_samples = X.shape[0]
        distances = np.zeros(n_samples)
        for idx in range(n_samples):
            distances[idx] = self.distance_function(X, self.X[idx, :])
        return self.transformation_function(np.sort(distances)[K])
    
    def base_score_samples(self, X: np.ndarray, K: int=10) -> np.ndarray:
        n_samples_test = X.shape[0]
        if n_samples_test == 1:
            return self.score(X)
        else:
            scores = np.zeros((n_samples_test, 1))
            for idx in range(n_samples_test):
                scores[idx] = self.score(X[idx, :], K)
            return scores

    @property
    def distance_function(self) -> Callable[[np.ndarray, np.ndarray], float]:
        """
        Sets and changes the distance_function method of the density estimator
        -------
        Callable
        """
        
        return self._distance_function

    @distance_function.setter
    def distance_function(self, distance_function) -> None:
        self._distance_function = check_type(distance_function, "distance_function.setter", Callable[[np.ndarray, np.ndarray], float])
        
    @property
    def transformation_function(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Sets and changes the transformation_function method of the density estimator
        -------
        Callable
        """
        
        return self._transformation_function

    @transformation_function.setter
    def transformation_function(self, transformation_function) -> None:
        self._transformation_function = check_type(transformation_function, "transformation_function.setter", Callable[[np.ndarray], np.ndarray])

    @property
    def fit(self) -> Callable[[np.ndarray], None]:
        """
        Sets and changes the fit method of the density estimator
        -------
        Callable
        """
        return self._fit

    @fit.setter
    def fit(self, fit) -> None:
        self._fit = check_type(fit, "fit.setter", Callable[[np.ndarray], None])

    @property
    def score(self) -> Callable[[np.ndarray, Optional[int]], np.ndarray]:
        """
        Sets and changes the score method of the density estimator
        -------
        Callable
        """
        return self._score

    @score.setter
    def score(self, score) -> None:
        self._score = check_type(score, "score.setter", Callable[[np.ndarray, Optional[int]], np.ndarray])

    @property
    def score_samples(self) -> Callable[[np.ndarray, Optional[int]], np.ndarray]:
        """
        Sets and changes the score_samples method of the density estimator
        -------
        Callable
        """
        return self._score_samples

    @score_samples.setter
    def score_samples(self, score_samples) -> None:
        self._score_samples = check_type(score_samples, "score_samples.setter", Callable[[np.ndarray, Optional[int]], np.ndarray])
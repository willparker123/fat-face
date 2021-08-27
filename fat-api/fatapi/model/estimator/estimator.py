import numpy as np

from fatapi.data import Data
from typing import Callable, List
class Estimator():
    """
    Abstract class for ML models to apply interpretability / explainability methods to
    
    Parameters
    ----------
    fit(X: numpy.array, columns?: List[int]) -> numpy.array:
        Fits the estimator to X - creates category dict internally
    transform(X: numpy.array, columns?: List[int]) -> numpy.array:
        Generate transformed data from X using columns[] - requires fit
    inverse_transform(X: numpy.array, columns?: List[int]) -> numpy.array:
        Generate original data from transformed X using columns[] - requires fit
    
    Methods
    -------
    fit(X: numpy.array, columns?: List[int]) -> numpy.array:
        Fits the estimator to X - creates category dict internally
    transform(X: numpy.array, columns?: List[int]) -> numpy.array:
        Generate transformed data from X using columns[] - requires fit
    inverse_transform(X: numpy.array, columns?: List[int]) -> numpy.array:
        Generate original data from transformed X using columns[] - requires fit
    get_categories() -> dict:
        Get categories and values from fitted data as dict (in form {"gender":["Female", "Male"],"category":[1,2,3]})
    """
    def __init__(self, fit, transform, inverse_transform) -> None:
        if (callable(fit)):
            self.fit = fit
        else:
            raise ValueError("Invalid argument in __init__: fit is not a function")
        if (callable(transform)):
            self.transform = transform
        else:
            raise ValueError("Invalid argument in __init__: transform is not a function")
        if (callable(inverse_transform)):
            self.inverse_transform = inverse_transform
        else:
            raise ValueError("Invalid argument in __init__: inverse_transform is not a function")

    @property
    def fit(self) -> Callable:
        """
        Sets and changes the fit method of the estimator
        -------
        Callable
        """
        
        return self.fit

    @fit.setter
    def fit(self, fit) -> None:
        if callable(fit):
            self.fit = fit
        else:
            raise ValueError("Invalid argument in fit.setter: fit is not a function")
        
    @property
    def fit(self) -> Callable:
        """
        Sets and changes the fit method of the estimator
        -------
        Callable
        """
        
        return self.fit

    @fit.setter
    def fit(self, fit) -> None:
        if callable(fit):
            self.fit = fit
        else:
            raise ValueError("Invalid argument in fit.setter: fit is not a function")
        

from fatapi.model.estimators import Transformer
from fatapi.helpers import check_type
from typing import Callable
import numpy as np

class BlackBox(object):
    """
    Abstract class representing an ML model with predictive methods
    
    Parameters
    ----------
    classifier : attr(predict, predict_proba, fit, score)
    
    Methods
    -------
    predict() : (X: np.ndarray) -> np.ndarray
        Method for predicting the class label of X
    predict_proba() : (X: np.ndarray) -> np.ndarray
        Method for predicting the probability of the prediction of X
    fit() : (X: np.ndarray, Y?: np.ndarray) -> np.ndarray
        Method for fitting model to X, Y
    score()? : (X: np.ndarray, Y?: np.ndarray) -> np.ndarray
        Method for calculating a score when predicting X and comparing with Y

    """
    def __init__(self, classifier, **kwargs) -> None:
        self.classifier = classifier
        try:
            if callable(getattr(classifier, "predict")):
                pass
        except:
            raise ValueError("Invalid argument in __init__: classifier does not have function predict")
        try:
            if callable(getattr(classifier, "predict_proba")):
                pass
        except:
            raise ValueError("Invalid argument in __init__: classifier does not have function predict_proba")
        try:
            if callable(getattr(classifier, "fit")):
                pass
        except:
            raise ValueError("Invalid argument in __init__: classifier does not have function fit")
        if callable(getattr(classifier, "score")):
            self._score = self.classifier.score
            pass
        else:
            raise ValueError("Invalid argument in __init__: score is not a function")
        self._fit = self.classifier.fit
        self._predict = self.classifier.predict
        self._predict_proba = self.classifier.predict_proba

    @property
    def fit(self) -> Callable:
        """
        Sets and changes the fit method of the model
        -------
        Callable
        """
        return self._fit

    @fit.setter
    def fit(self, fit) -> None:
        self._fit = check_type(fit, Callable, "fit.setter")
        
    @property
    def predict(self) -> Callable:
        """
        Sets and changes the predict method of the model
        -------
        Callable
        """
        
        return self._predict

    @predict.setter
    def predict(self, predict) -> None:
        self._predict = check_type(predict, Callable, "predict.setter")
        
    @property
    def predict_proba(self) -> Callable:
        """
        Sets and changes the predict_proba method of the model
        -------
        Callable
        """

        return self._predict_proba

    @predict_proba.setter
    def predict_proba(self, predict_proba) -> None:
        self._predict_proba = check_type(predict_proba, Callable, "predict_proba.setter")
        
    @property
    def score(self) -> Callable:
        """
        Sets and changes the score method of the model
        -------
        Callable
        """
        return self._score

    @score.setter
    def score(self, score) -> None:
        self._score = check_type(score, Callable, "score.setter")
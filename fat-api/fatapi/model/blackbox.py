
from fatapi.model.estimators import Transformer
from fatapi.helpers import check_type
from typing import Callable, Optional
import numpy as np

class BlackBox(object):
    """
    Abstract class representing an ML model with predictive methods
    
    Parameters
    ----------
    classifier : attr(predict, predict_proba, fit, score)
        Object which has the methods above used as a BlackBox in Models and Methods
    
    Methods
    -------
    predict() : (X: np.ndarray) -> np.ndarray
        Method for predicting the class label of X
    predict_proba() : (X: np.ndarray) -> np.ndarray
        Method for predicting the probability of the prediction of X
    fit() : (X: np.ndarray, Y?: np.ndarray)
        Method for fitting model to X, Y
    score()? : (X: np.ndarray, Y: np.ndarray) -> np.ndarray
        Method for calculating a score when predicting X and comparing with Y

    """
    def __init__(self, classifier=None, **kwargs) -> None:
        if classifier is not None:
            self.classifier = classifier
            if hasattr(classifier, "predict"):
                self._predict = check_type(self.classifier.predict, "__init__", Callable[[np.ndarray], np.ndarray])
            else:
                raise ValueError("Invalid argument in __init__: classifier does not have function predict")
            if hasattr(classifier, "predict_proba"):
                self._predict_proba = check_type(self.classifier.predict_proba, "__init__", Callable[[np.ndarray], np.ndarray])
            else:
                raise ValueError("Invalid argument in __init__: classifier does not have function predict_proba")
            if hasattr(classifier, "fit"):
                self._fit = check_type(self.classifier.fit, "__init__", Callable[[np.ndarray, Optional[np.ndarray]], None])
            else:
                raise ValueError("Invalid argument in __init__: classifier does not have function fit")
            if hasattr(classifier, "score"):
                self._score = check_type(self.classifier.score, "__init__", Callable[[np.ndarray, np.ndarray], np.ndarray])
            else:
                raise ValueError("Invalid argument in __init__: classifier does not have function score")#
        else:
            if 'fit' in kwargs and 'predict' in kwargs and 'predict_proba' in kwargs and 'score' in kwargs:
                self._fit = check_type(kwargs.get("fit"), "__init__", Callable[[np.ndarray, Optional[np.ndarray]], None])
                self._predict = check_type(kwargs.get("predict"), "__init__", Callable[[np.ndarray], np.ndarray])
                self._predict_proba = check_type(kwargs.get("predict_proba"), "__init__", Callable[[np.ndarray], np.ndarray])
                self._score = check_type(kwargs.get("score"), "__init__", Callable)

    @property
    def fit(self) -> Callable[[np.ndarray, Optional[np.ndarray]], None]:
        """
        Sets and changes the fit method of the model

        """
        return self._fit

    @fit.setter
    def fit(self, fit) -> None:
        self._fit = check_type(fit, "fit.setter", Callable[[np.ndarray, Optional[np.ndarray]], None])
        
    @property
    def predict(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Sets and changes the predict method of the model

        """
        
        return self._predict

    @predict.setter
    def predict(self, predict) -> None:
        self._predict = check_type(predict, "predict.setter", Callable[[np.ndarray], np.ndarray])
        
    @property
    def predict_proba(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Sets and changes the predict_proba method of the model

        """

        return self._predict_proba

    @predict_proba.setter
    def predict_proba(self, predict_proba) -> None:
        self._predict_proba = check_type(predict_proba, "predict_proba.setter", Callable[[np.ndarray], np.ndarray])
        
    @property
    def score(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """
        Sets and changes the score method of the model

        """
        return self._score

    @score.setter
    def score(self, score) -> None:
        self._score = check_type(score, "score.setter", Callable[[np.ndarray, np.ndarray], np.ndarray])
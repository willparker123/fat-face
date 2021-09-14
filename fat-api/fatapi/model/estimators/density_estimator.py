
from typing import Callable
import numpy as np

class DensityEstimator(object):
    """
    Abstract class used for KDE and GS kernels to get density scores from data
    
    Parameters
    ----------
    classifier : attr(predict, predict_proba, fit, score)
    
    Methods
    -------
    fit(X: np.array, Y?: np.array) : np.array()
        Method for fitting model to X, Y
    score_samples(X: np.array, Y?: np.array) : np.array()
        Method for calculating a score when predicting X and comparing with Y
    score(X: np.array, Y?: np.array)? : np.array()
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
        try:
            if callable(getattr(classifier, "score")):
                pass
        except:
            raise ValueError("Invalid argument in __init__: classifier does not have function score")

    @property
    def fit(self) -> Callable:
        """
        Sets and changes the fit method of the model
        -------
        Callable
        """
        if hasattr(self, '_fit'):
            return self._fit
        else:
            return self.classifier.fit

    @fit.setter
    def fit(self, fit) -> None:
        if callable(fit):
            self._fit = fit
        else:
            raise ValueError("Invalid argument in fit.setter: _fit is not a function")
        
    @property
    def predict(self) -> Callable:
        """
        Sets and changes the predict method of the model
        -------
        Callable
        """
        if hasattr(self, '_predict'):
            return self._predict
        else:
            return self.classifier.predict

    @predict.setter
    def predict(self, _predict) -> None:
        if callable(_predict):
            self._predict = _predict
        else:
            raise ValueError("Invalid argument in predict.setter: _predict is not a function")
        
    @property
    def predict_proba(self) -> Callable:
        """
        Sets and changes the predict_proba method of the model
        -------
        Callable
        """
        if hasattr(self, 'predict_probaf'):
            return self.predict_probaf
        else:
            return self.classifier.predict_proba

    @predict_proba.setter
    def predict_proba(self, predict_probaf) -> None:
        if callable(predict_probaf):
            self.predict_probaf = predict_probaf
        else:
            raise ValueError("Invalid argument in predict_proba.setter: predict_probaf is not a function")
        
    @property
    def score(self) -> Callable:
        """
        Sets and changes the score method of the model
        -------
        Callable
        """
        if hasattr(self, '_score'):
            return self._score
        else:
            return self.classifier.score

    @score.setter
    def score(self, _score) -> None:
        if callable(_score):
            self._score = _score
        else:
            raise ValueError("Invalid argument in score.setter: _score is not a function")
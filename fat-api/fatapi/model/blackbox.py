
from fatapi.model.estimator import Estimator
from typing import Callable
import numpy as np

class BlackBox():
    """
    Abstract class representing an ML model with predictive methods
    
    Parameters
    ----------
    classifier : attr(predict, predict_proba, fit, score)
    
    Methods
    -------
    predict(X: np.array)? : np.array()
        Method for predicting the class label of X
    predict_proba(X: np.array)? : np.array()
        Method for predicting the probability of the prediction of X
    fit(X: np.array, Y?: np.array)? : np.array()
        Method for fitting model to X, Y
    score(X: np.array, Y?: np.array)? : np.array()
        Method for calculating a score when predicting X and comparing with Y

    """
    def __init__(self, classifier, **kwargs) -> None:
        self.classifier = classifier
        try:
            predict = getattr(classifier, "predict")
            if callable(predict):
                self.predict = self.classifier.predict
        except:
            raise ValueError("Invalid argument in __init__: classifier does not have function predict")
        try:
            predict_proba = getattr(classifier, "predict_proba")
            if callable(predict_proba):
                self.predict_proba = self.classifier.predict_proba
        except:
            raise ValueError("Invalid argument in __init__: classifier does not have function predict_proba")
        try:
            fit = getattr(classifier, "fit")
            if callable(fit):
                self.fit = self.classifier.fit
        except:
            raise ValueError("Invalid argument in __init__: classifier does not have function fit")
        try:
            score = getattr(classifier, "score")
            if callable(score):
                self.score = self.classifier.score
        except:
            raise ValueError("Invalid argument in __init__: classifier does not have function score")

    @property
    def fit(self) -> Callable:
        """
        Sets and changes the fit method of the model
        -------
        Callable
        """
        
        return self.fit

    @fit.setter
    def fit(self, fitf) -> None:
        if callable(fitf):
            self.fit = fitf
        else:
            raise ValueError("Invalid argument in fit.setter: fitf is not a function")
        
    @property
    def predict(self) -> Callable:
        """
        Sets and changes the predict method of the model
        -------
        Callable
        """
        
        return self.predict

    @predict.setter
    def predict(self, predictf) -> None:
        if callable(predictf):
            self.predict = predictf
        else:
            raise ValueError("Invalid argument in predict.setter: predictf is not a function")
        
    @property
    def predict_proba(self) -> Callable:
        """
        Sets and changes the predict_proba method of the model
        -------
        Callable
        """
        
        return self.predict_proba

    @predict_proba.setter
    def predict_proba(self, predict_probaf) -> None:
        if callable(predict_probaf):
            self.predict_proba = predict_probaf
        else:
            raise ValueError("Invalid argument in predict_proba.setter: predict_probaf is not a function")
        
    @property
    def score(self) -> Callable:
        """
        Sets and changes the score method of the model
        -------
        Callable
        """
        
        return self.score

    @score.setter
    def score(self, scoref) -> None:
        if callable(scoref):
            self.score = scoref
        else:
            raise ValueError("Invalid argument in score.setter: scoref is not a function")
        
    @property
    def encoder(self) -> Estimator:
        """
        Sets and changes the encoder method of the model
        -------
        Estimator
        """
        
        return self.encoder

    @encoder.setter
    def encoder(self, encoder) -> None:
        if callable(encoder):
            self.encoder = encoder
        else:
            raise ValueError("Invalid argument in encoder.setter: encoder is not an Estimator")
        
    @property
    def scaler(self) -> Callable:
        """
        Sets and changes the scaler method of the model
        -------
        Callable
        """
        
        return self.scaler
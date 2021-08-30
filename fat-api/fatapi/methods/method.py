from fatapi.model import BlackBox
from fatapi.model.estimator import Estimator
from fatapi.data import Data
from typing import Callable, List
import numpy as np

class ExplainabilityMethod():
    """
    Abstract class for ML models to apply interpretability / explainability methods to
    
    Parameters
    ----------

    factuals : fatapi.data.Data
        Data object containing features of datapoints to be analysed in the ExplainabilityMethod
    factuals_target? : fatapi.data.Data
        Data object containing target features of datapoints to be analysed in the ExplainabilityMethod
    predict()? : (X: np.array) -> np.array()
        Method for predicting the class label of X
        -- Only required if model not supplied
    model? : fatapi.model.Model
        Model object used to get prediction values, class predictions and Estimator objects
        -- Only required if predict not supplied
    
    Methods
    -------
    train(X: np.array, Y: np.array):
        Calls self.fit()
    encode_normalize_order_factuals:
        Uses encoder and scaler from black-box-model to preprocess data as needed.
    """
    def __init__(self, data: Data, target: Data=None, blackbox: BlackBox=None, X_tofit: List[int]=None, Y_tofit: List[int]=None, **kwargs) -> None:
        if not blackbox and not (kwargs.get('fit') and kwargs.get('predict') and kwargs.get('predict_proba') and kwargs.get('score')):
            raise ValueError(f"Missing arguments in __init__: {'' if blackbox else 'blackbox'} {'' if kwargs.get('fit') else 'fit'} {'' if kwargs.get('predict') else 'predict'} {'' if kwargs.get('predict_proba') else 'predict_proba'} {'' if kwargs.get('score') else 'score'}")
        if blackbox and (kwargs.get('fit') and kwargs.get('predict') and kwargs.get('predict_proba') and kwargs.get('score')):
            raise ValueError("Invalid arguments in __init__: please provide blackbox or (predict, predict_proba, fit, score)")
        if kwargs.get('blackbox'):
            bb=kwargs.get('blackbox')
            super().__init__(bb)
            self.blackbox = bb
            self.fit = self.blackbox.fit
            self.predict = self.blackbox.predict
            self.predict_proba = self.blackbox.predict_proba
            self.score = self.blackbox.score
        if (kwargs.get('fit') and kwargs.get('predict') and kwargs.get('predict_proba') and kwargs.get('score')):
            if kwargs.get('fit'):
                if (callable(kwargs.get('fit'))):
                    self.encoder = kwargs.get('fit')
                else:
                    raise ValueError("Invalid argument in __init__: fit is not a function")
            if kwargs.get('predict'):
                if (callable(kwargs.get('predict'))):
                    self.encoder = kwargs.get('predict')
                else:
                    raise ValueError("Invalid argument in __init__: predict is not a function")
            if kwargs.get('predict_proba'):
                if (callable(kwargs.get('predict_proba'))):
                    self.encoder = kwargs.get('predict_proba')
                else:
                    raise ValueError("Invalid argument in __init__: predict_proba is not a function")
            if kwargs.get('score'):
                if (callable(kwargs.get('score'))):
                    self.encoder = kwargs.get('score')
                else:
                    raise ValueError("Invalid argument in __init__: score is not a function")
        if kwargs.get('encoder'):
            if (type(kwargs.get('encoder'))==Estimator):
                self.encoder = kwargs.get('encoder')
            else:
                raise ValueError("Invalid argument in __init__: encoder is not an Estimator")
        if kwargs.get('scaler'):
            if (type(kwargs.get('scaler'))==Estimator):
                self.encoder = kwargs.get('scaler')
            else:
                raise ValueError("Invalid argument in __init__: scaler is not an Estimator")
        if type(data)==Data:
            if not data.isEncoded and (kwargs.get('encoder') and kwargs.get('scaler')):
                d : Data = data
                super().__init__(d)
                self.data = d
            else:
                raise ValueError("Invalid argument in __init__: data must be isEncoded or (scaler, encoder) must be supplied")
        else:
            raise ValueError("Invalid argument in __init__: data must be of type fatapi.data.Data")
        if target:
            if (target.dataset.shape[0] == data.dataset.shape[0]):
                if type(target)==Data:
                    if not target.isEncoded and (kwargs.get('encoder') and kwargs.get('scaler')):
                        t : Data = target
                        super().__init__(t)
                        self.target = t
                    else:
                        raise ValueError("Invalid argument in __init__: target must be isEncoded or (scaler, encoder) must be supplied")
                else:
                    raise ValueError("Invalid argument in __init__: target is not of type fatapi.data.Data")
            else:
                raise ValueError("Invalid argument in __init__: target is not of same shape[0] as data")
        if X_tofit:
            self.X_tofit = X_tofit
        if target:
            if Y_tofit:
                self.Y_tofit = Y_tofit
        else:
            raise ValueError("Warning in __init__: no target supplied but Y_tofit supplied")
        try:
            d, t = self.get_data_tofit()
            if t:
                self.train(d, t)
            else:
                self.train(d)
        except:
            raise ValueError("Invalid argument in model.fit: has to take (X: numpy.array, Y: numpy.array)")

    def get_data_tofit(self):
        d = self.data.dataset
        t = self.target.dataset
        if self.X_tofit:
            d = keep_cols(d, self.X_tofit)
        if self.Y_tofit and t:
            t = keep_cols(t, self.Y_tofit)
        return d, t

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
    def scaler(self) -> Estimator:
        """
        Sets and changes the scaler method of the model
        -------
        Callable
        """
        
        return self.scaler

    @scaler.setter
    def scaler(self, scalerf) -> None:
        if callable(scalerf):
            self.scaler = scalerf
        else:
            raise ValueError("Invalid argument in scaler.setter: scalerf is not a function")
        
    def encode(self, X: np.array=None):
        #todo
        return X
    
    def decode(self, X: np.array=None):
        #todo
        return X
    
    def scale(self, X: np.array=None):
        #todo
        return X
    
    def unscale(self, X: np.array=None):
        #todo
        return X
    
    def train(self, X: np.array=None, Y: np.array=None):
        if not X and Y:
            raise ValueError("Invalid argument to model.train: X not provided - please provide only X or X and Y or nothing")
        else:
            if X and Y:
                self.fit(X,Y)
                return self
            else:
                X_, Y_ = self.get_data_tofit()
                self.fit(X_,Y_)
                return self
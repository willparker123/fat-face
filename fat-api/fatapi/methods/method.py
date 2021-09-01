from fatapi.model import BlackBox, Model
from fatapi.model.estimator import Estimator
from fatapi.data import Data
from typing import Callable, List
import numpy as np

class ExplainabilityMethod():
    """
    Abstract class for AI explainability methods such as FACE, CEM, PDP, ALE...
    Contains methods for generating counterfactuals and the data / model to apply the method to
    
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
        Model object used to get prediction values and class predictions
        -- Only required if predict not supplied
    
    Methods
    -------
    get_counterfactuals() : (X: np.array, Y: np.array) -> np.array
        Calls self.fit()
    encode_normalize_order_factuals() : (factuals?: fatapi.data.Data, factuals_target?: fatapi.data.Data, model?: fatapi.model.Model, 
                                            scaler?: fatapi.model.estimator.Estimator, encoder?: fatapi.model.estimator.Estimator) -> numpy.array
        Uses encoder and scaler from black-box-model to preprocess data as needed.
    """
    def __init__(self, factuals=None, factuals_target=None, **kwargs) -> None:
        if not (kwargs.get('predict') or kwargs.get('model')) or (kwargs.get('predict') and kwargs.get('model')):
            raise ValueError(f"Invalid arguments in __init__: please provide model or predict function")
        if kwargs.get('model'):
            if type(kwargs.get('model'))==Model:
                m = kwargs.get('model')
                self.model = m
                super().__init__(m)
                self.predict = self.model.predict
            else:
                raise ValueError(f"Invalid argument in __init__: model is not of type Model")
        if kwargs.get('predict'):
            if callable(kwargs.get('predict')):
                self.predict = kwargs.get('predict')
            else:
                raise ValueError(f"Invalid argument in __init__: predict is not a function")
        if not factuals and factuals_target:
            raise ValueError("Invalid argument in __init__: factual targets supplied with no factuals - provide factuals argument if targets are the features")
        else:
            if factuals:
                self.factuals = factuals
            else:
                print("Warning: No datapoints supplied as factuals - counterfactual methods require arguments")
            if factuals_target:
                self.factuals_target = factuals_target
            else:
                print("Warning: No targets supplied for factuals (datapoints) - counterfactual methods require arguments")
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
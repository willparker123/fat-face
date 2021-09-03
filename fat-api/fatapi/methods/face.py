from fatapi.model import Model
from fatapi.model.estimator import Estimator
from fatapi.data import Data
from typing import Callable, Tuple, Union
from fatapi.methods import ExplainabilityMethod
import numpy as np

class FACEMethod(ExplainabilityMethod):
    """
    Abstract class for the FACE algorithm as an AI explainability method.
    
    Contains methods for generating counterfactuals and the data / model to apply FACE to, 
    as well as visualisation and extracting internal representations
    
    Parameters
    ----------
    factuals? : fatapi.data.Data
        Data object containing features of datapoints to be used in ExplainabilityMethod methods
    factuals_target? : fatapi.data.Data
        Data object containing target features of datapoints to be used in ExplainabilityMethod methods
    predict()? : (X: np.array) -> np.array()
        Method for predicting the class label of X
        -- Only required if model not supplied
    model? : fatapi.model.Model
        Model object used to get prediction values and class predictions
        -- Only required if predict not supplied
    explain() : (self, X: numpy.array, Y?: numpy.array, predict()?: Callable) -> numpy.array
        Generates counterfactual datapoints from X and Y using predict function or model predict function or argument predict function
    
    Methods
    -------
    explain() : (X?: numpy.array, Y?: numpy.array, predict()?: Callable) -> numpy.array
        Generates counterfactual datapoints from X and Y using predict function or model predict function or argument predict function
        -- Uses factuals and factuals_target from preprocess_factuals if no X and Y given
    preprocess_factuals() : (factuals?: fatapi.data.Data, factuals_target?: fatapi.data.Data, model?: fatapi.model.Model, 
                                            scaler?: fatapi.model.estimator.Estimator, encoder?: fatapi.model.estimator.Estimator) -> numpy.array
        Uses encoder and scaler from black-box-model or argument to preprocess data as needed.
    """
    def __init__(self, **kwargs) -> None:
        if not (kwargs.get('kernel')):
            raise ValueError(f"Invalid arguments in __init__: please provide kernel")
        if kwargs.get('kernel'):
            print("kernel init")

    @property
    def predict(self) -> Callable:
        """
        Sets and changes the predict method of the explainability method
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
    def model(self) -> Model:
        """
        Sets and changes the model attribute of the explainability method
        -------
        Callable
        """
        
        return self.model

    @predict.setter
    def model(self, model) -> None:
        if type(model)==Model:
            self.model = model
        else:
            raise ValueError("Invalid argument in model.setter: model is not a fatapi.model.Model")
          
    @property
    def factuals(self) -> Data:
        """
        Sets and changes the default factuals the explainability method applies to
        -------
        Callable
        """
        
        return self.factuals

    @factuals.setter
    def factuals(self, factuals) -> None:
        if type(factuals)==Data:
            self.factuals = factuals
        else:
            raise ValueError("Invalid argument in factuals.setter: factuals is not of type fatapi.data.Data")
        
    @property
    def explain(self, X: np.array=None, Y: np.array=None, predict: Callable=None) -> Union[np.array, Tuple[np.array, np.array]]:
        """
        Sets and changes the method to get explainability information from a set of factuals
        -------
        Callable
        """
        X_, Y_ = self.preprocess_factuals()
        if X:
            X_ = X
        if Y:
            Y_ = Y
            
        if not (X_.shape[0]==Y_.shape[0]):
            raise ValueError("Invalid argument in explain: different number of points in data and target")
        if Y_:
            return self.explain(X_, Y_)
        else:
            return self.explain(X_)

    @explain.setter
    def explain(self, explainf) -> None:
        if callable(explainf):
            self.explain = explainf
        else:
            raise ValueError("Invalid argument in explain.setter: explain is not a function")
        
    def preprocess_factuals(self, factuals: Data=None, factuals_target: Data=None, model: Model=None, scaler: Estimator=None, encoder: Estimator=None) -> Union[Tuple[np.array, np.array], np.array]:
        """
        Processes non-encoded feature and target data using model's scaler / encoder or using arguments
        -------
        Callable
        """
        
        if not self.factuals and not factuals:
            raise ValueError(f"Missing arguments in preprocess_factuals: must provide {'' if self.factuals else 'self.factuals'} or {'' if factuals else 'factuals'}")
        if self.factuals:
            facts = self.factuals
        if factuals:
            facts = factuals
        if not self.factuals_target:
            if factuals_target:
                print(f"Warning: targets for factuals provided in preprocess_factuals but no default self.factuals_target")
            else:
                print(f"Warning: no targets for factuals provided to preprocess_factuals - features only (X)")
        if self.factuals_target:
            facts_target = self.factuals_target
        if factuals_target:
            facts_target = factuals_target
        X_ = facts.dataset
        Y_ = facts_target.dataset
        if facts.isEncoded and facts_target.isEncoded:
            return X_, Y_
        else:
            if not (self.model or model or scaler or encoder):
                raise ValueError(f"Missing arguments in preprocess_factuals: must provide {'' if self.model else 'self.model'} or {'' if model else 'model'} or {'' if scaler else 'scaler'} or {'' if encoder else 'encoder'}")
            else:
                if self.model:
                    encodef = self.model.encode()
                    scalef = self.model.scale()
                if model:
                    encodef = model.encode()
                if encoder:
                    encodef = encoder.encode()
                if scaler:
                    scalef = scaler.encode()
                if encodef:
                    X_ = encodef(facts.dataset, facts.categoricals)
                if scalef:
                    X_ = scalef(facts.dataset, facts.numericals)
                if facts_target: 
                    if encodef:
                        Y_ = encodef(facts_target.dataset, facts_target.categoricals)
                    if scalef:
                        Y_ = scalef(facts_target.dataset, facts_target.numericals)
            return X_, Y_
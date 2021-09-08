from fatapi.model import Model
from fatapi.model.estimator import Estimator
from fatapi.data import Data
from typing import Callable, Tuple, Union
import numpy as np

class ExplainabilityMethod():
    """
    Abstract class for AI explainability methods such as FACE, CEM, PDP, ALE...
    Contains methods for generating counterfactuals and the data / model to apply the method to
    
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
    explain()? : (self, X: numpy.array, Y?: numpy.array, _predict()?: Callable) -> numpy.array
        Generates counterfactual datapoints from X and Y using predict function or model predict function or argument predict function
    
    Methods
    -------
    explain() : (X?: numpy.array, Y?: numpy.array, _predict()?: Callable) -> numpy.array
        Generates counterfactual datapoints from X and Y using predict function or model predict function or argument predict function
        -- Uses factuals and factuals_target from preprocess_factuals if no X and Y given
    preprocess_factuals() : (factuals?: fatapi.data.Data, factuals_target?: fatapi.data.Data, model?: fatapi.model.Model, 
                                            scaler?: fatapi.model.estimator.Estimator, encoder?: fatapi.model.estimator.Estimator) -> numpy.array
        Uses encoder and scaler from black-box-model or argument to preprocess data as needed.
    """
    def __init__(self, factuals=None, factuals_target=None, **kwargs) -> None:
        if not (kwargs.get('predict') or kwargs.get('model')) or (kwargs.get('predict') and kwargs.get('model')):
            raise ValueError(f"Invalid arguments in __init__: please provide model or predict function but not both")
        if kwargs.get('model'):
            if type(kwargs.get('model'))==Model:
                m: Model = kwargs.get('model')
                self._model: Model = m
                self._predict = self._model.predict
            else:
                raise ValueError(f"Invalid argument in __init__: model is not of type Model")
        if kwargs.get('predict'):
            if callable(kwargs.get('predict')):
                self._predict = kwargs.get('predict')
            else:
                raise ValueError(f"Invalid argument in __init__: predict is not a function")
        self._explain = lambda **kwargs: kwargs
        if kwargs.get('explain'):
            if callable(kwargs.get('explain')):
                self._explain = kwargs.get('explain')
            else:
                raise ValueError(f"Invalid argument in __init__: explain is not a function")
        if not factuals and factuals_target:
            raise ValueError("Invalid argument in __init__: factual targets supplied with no factuals - provide factuals argument if targets are the features")
        else:
            if factuals:
                self._factuals = factuals
            else:
                print("Warning: No datapoints supplied as factuals - counterfactual methods require factuals argument")
            if factuals_target:
                self._factuals_target = factuals_target
            else:
                print("Warning: No targets supplied for factuals (datapoints) - counterfactual methods require factuals_target argument")

    @property
    def predict(self) -> Callable:
        """
        Sets and changes the predict method of the explainability method
        -------
        Callable
        """
        
        return self._predict

    @predict.setter
    def predict(self, _predict) -> None:
        if callable(_predict):
            self._predict = _predict
        else:
            raise ValueError("Invalid argument in predict.setter: _predict is not a function")
        
    @property
    def model(self) -> Model:
        """
        Sets and changes the model attribute of the explainability method
        -------
        Callable
        """
        
        return self._model

    @predict.setter
    def model(self, model) -> None:
        if type(model)==Model:
            self._model = model
        else:
            raise ValueError("Invalid argument in model.setter: model is not a fatapi.model.Model")
          
    @property
    def factuals(self) -> Data:
        """
        Sets and changes the default factuals the explainability method applies to
        -------
        Callable
        """
        
        return self._factuals

    @factuals.setter
    def factuals(self, factuals) -> None:
        if type(factuals)==Data:
            self._factuals = factuals
        else:
            raise ValueError("Invalid argument in factuals.setter: factuals is not of type fatapi.data.Data")
        
    @property
    def factuals_target(self) -> Data:
        """
        Sets and changes the default factuals_target the explainability method applies to
        -------
        Callable
        """
        
        return self._factuals_target

    @factuals_target.setter
    def factuals_target(self, factuals_target) -> None:
        if type(factuals_target)==Data:
            if (factuals_target.shape[0]==factuals.shape[0]):
                self._factuals_target = factuals_target
            else:
                raise ValueError("Invalid argument in factuals_target.setter: factuals_target has a different number of points than factuals")
        else:
            raise ValueError("Invalid argument in factuals_target.setter: factuals_target is not of type fatapi.data.Data")

    def explain(self, X: np.array=[], Y: np.array=[], predict: Callable=None) -> Union[np.array, Tuple[np.array, np.array]]:
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
            
        if predict:
            _predict = predict
        else:
            _predict = self._predict
        if not (X_.shape[0]==Y_.shape[0]):
            raise ValueError("Invalid argument in explain: different number of points in data and target")
        if len(Y_)>0:
            return self._explain(X=X_, Y=Y_, predict=_predict)
        else:
            return self._explain(X=X_, predict=_predict)

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
        facts_target = None
        if self.factuals_target:
            facts_target = self.factuals_target
        if factuals_target:
            facts_target = factuals_target
        Y_ = []
        if facts_target:
            Y_ = facts_target.dataset
        X_ = facts.dataset
        if facts.isEncoded and facts_target.isEncoded:
            return X_, Y_
        else:
            if not (self.model or model or scaler or encoder):
                raise ValueError(f"Missing arguments in preprocess_factuals: must provide {'' if self.model else 'self.model'} or {'' if model else 'model'} or {'' if scaler else 'scaler'} or {'' if encoder else 'encoder'}")
            else:
                if self.model:
                    _encode = self.model.encode()
                    scalef = self.model.scale()
                if model:
                    _encode = model.encode()
                if encoder:
                    _encode = encoder.encode()
                if scaler:
                    scalef = scaler.encode()
                if _encode:
                    X_ = _encode(facts.dataset, facts.categoricals)
                if scalef:
                    X_ = scalef(facts.dataset, facts.numericals)
                if facts_target: 
                    if _encode:
                        Y_ = _encode(facts_target.dataset, facts_target.categoricals)
                    if scalef:
                        Y_ = scalef(facts_target.dataset, facts_target.numericals)
            return X_, Y_
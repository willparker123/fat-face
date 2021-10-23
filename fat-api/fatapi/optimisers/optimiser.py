
from typing import Callable, Optional, Union
from fatapi.helpers import check_type
import numpy as np

class Optimiser(object):
    """
    Abstract class used for a gradient ascent / descent optimiser (e.g. in CEMMethod).
    
    Note: Initialised variables will be overwritten by Method defaults and supplied Method args - 
    call setters before explain() in Method to override the arguments for the optimiser in the Method constructor
    
    
    Parameters
    ----------
    objective() : (value: np.ndarray, **kwargs) -> np.ndarray
        The objective function the optimiser is trying to solve, taking in at least one argument 'value'.
        Accepts any other arguments that may be required as **kwargs
        (e.g. delta, beta, gamma, c, autoencoder for CEM)
    step_function()? : (initial_value: np.ndarray, objective: Callable[..., np.ndarray], prev_loss: np.ndarray, learning_rate: float, iteration: int, 
                        max_iterations: int, decay_function?: Callable[[float, int, int], float], **kwargs) -> np.ndarray
        The function used at each timestep (until max_iterations or stop_condition) that is used for learning
    predict()? : (X: np.ndarray) -> np.ndarray
        Method for predicting the class label of X
        -- Only required if needed in optimise() or objective()
    predict_proba()? : (X: np.ndarray) -> np.ndarray
        Method for getting the probability of X being the predicted class label
        -- Only required if needed in optimise() or objective()
    initial_learning_rate?: float
        Initial learning rate / step size for learning
        -- Default is 1e-2
    decay_function()?: (learning_rate: float, iteration: int, max_iterations: int, **kwargs) -> float
        Function which decays the learning rate over the iterations of learning
        -- Default is lambda lr, i, m, **kwargs = lr (Identity on initial_learning_rate [lr])
    max_iterations?: int
        Maximum iterations to complete an optimisation cycle over - stopping condition for learning
        -- Default is 1000
    stop_condition()?: (iteration?: int, max_iterations?: int, **kwargs) -> bool 
        Extra stopping conditions for the optimiser / learning cycle
        -- Default is lambda *args, **kwargs = False [Identity]

    Methods
    -------
    objective() : (value: np.ndarray, **kwargs) -> np.ndarray
        The objective function the optimiser is trying to solve, taking in at least one argument 'value'.
        Accepts any other arguments that may be required as **kwargs
        (e.g. delta, beta, gamma, c, autoencoder for CEM)
    optimise() : (objective: Callable[..., np.ndarray], max_iterations: int, initial_learning_rate: Union[float, int], 
                    decay_function: Callable[[float, int, int], float]) -> np.ndarray
        Optimises the supplied objective function using a supplied learning rate and optional decay function, 
        or using those set in the Optimiser object
    """
    def __init__(self, objective: Callable[[np.ndarray], np.ndarray], **kwargs):
        self._step_function = lambda d, obj_f, prev_loss, lr, i, max_iter, decay_f, *args, **kwargs: np.subtract(d, np.array(d.shape()).fill(lr*(obj_f(d, **kwargs)-prev_loss)))
        if 'step_function' in kwargs:
            self._step_function = check_type(kwargs.get("step_function"), "__init__", Callable[[np.ndarray, Callable[..., np.ndarray], float, float, int, int, Callable[[float, int, int], float]], np.ndarray])
        self._objective = objective
        if 'predict' in kwargs:
            self._predict = check_type(type(kwargs.get("predict")), "__init__", Callable[[np.ndarray], np.ndarray])
        if 'predict_proba' in kwargs:
            self._predict_proba = check_type(kwargs.get("predict_proba"), "__init__", Callable[[np.ndarray], np.ndarray])
        self._decay_function = lambda lr, i, m, **kwargs: lr
        if 'decay_function' in kwargs:
            self._decay_function = check_type(kwargs.get("decay_function"), "__init__", Callable[[float, int, int], float])
        self._max_iterations = 1000
        if 'max_iterations' in kwargs:
            if kwargs.get('max_iterations') >= 1:
                self._max_iterations = check_type(kwargs.get("max_iterations"), "__init__", int)
            else:
                raise ValueError(f"Invalid argument in __init__: max_iterations must be >= 1")
        self._initial_learning_rate = 1e-2
        if 'initial_learning_rate' in kwargs:
            if kwargs.get('initial_learning_rate') >= 0:
                self._initial_learning_rate = check_type(kwargs.get("initial_learning_rate"), "__init__", float, int)
            else:
                raise ValueError(f"Invalid argument in __init__: initial_learning_rate must be >= 0")
        self._stop_condition = lambda *args, **kwargs: False
        if 'stop_condition' in kwargs:
            self._stop_condition = check_type(kwargs.get("stop_condition"), "__init__", Callable[..., bool])
        
    def optimise(self, initial_value: np.ndarray, objective: Callable[..., np.ndarray]=None, max_iterations: int=None, 
                 initial_learning_rate: float=None, decay_function: Callable[[float, int, int], float]=None, **kwargs) -> np.ndarray:
        obj_f = self.objective
        max_iter = self.max_iterations
        init_lr = self.initial_learning_rate
        decay_f = self.decay_function
        if objective is not None:
            obj_f = objective
        if max_iterations is not None:
            max_iter = max_iterations
        if initial_learning_rate is not None:
            init_lr = initial_learning_rate
        if decay_function is not None:
            decay_f = decay_function
        lr = init_lr
        d = initial_value
        prev_loss = obj_f(value=d, **kwargs)
        for i in range(max_iter):
            d = self.step_function(initial_value=d, objective=obj_f, prev_loss=prev_loss, learning_rate=lr, iteration=i, max_iterations=max_iter, decay_function=decay_f, **kwargs)
            lr = decay_f(lr, i, max_iter, **kwargs)
            if self.stop_condition(initial_value=d, objective=obj_f, learning_rate=lr, iteration=i, max_iterations=max_iter, **kwargs):
                break
    
    @property
    def step_function(self) -> Callable[[np.ndarray, Callable[..., np.ndarray], float, float, int, int, Callable[[float, int, int], float]], np.ndarray]:
        """
        Sets and changes the step_function function used for optimisation - altering the value using learning_rate, objective and any other params

        """
        
        return self._optimise

    @step_function.setter
    def step_function(self, step_function) -> None:
        self._step_function = check_type(step_function, "step_function.setter", Callable[[np.ndarray, Callable[..., np.ndarray], float, float, int, int, Callable[[float, int, int], float]], np.ndarray])
    
    @property
    def objective(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Sets and changes the objective function to optimise

        """
        
        return self._objective

    @objective.setter
    def objective(self, objective) -> None:
        self._objective = check_type(objective, "objective.setter", Callable[[np.ndarray], np.ndarray])
    
    @property
    def predict(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Sets and changes the predict function used in objective / optimise if one is required but not supplied

        """
        
        return self._objective

    @predict.setter
    def predict(self, predict) -> None:
        self.predict = check_type(predict, "predict.setter", Callable[[np.ndarray], np.ndarray])
    
    @property
    def predict_proba(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Sets and changes the predict_proba function used in objective() / optimise() if one is required but not supplied

        """
        
        return self._objective

    @predict_proba.setter
    def predict_proba(self, predict_proba) -> None:
        self.predict_proba = check_type(predict_proba, "predict_proba.setter", Callable[[np.ndarray], np.ndarray])
    
    @property
    def decay_function(self) -> Callable[[float, int, int], float]:
        """
        Sets and changes the decay_function function used to change the learning_rate in optimise()

        """
        
        return self._objective

    @decay_function.setter
    def decay_function(self, decay_function) -> None:
        self.decay_function = check_type(decay_function, "decay_function.setter", Callable[[float, int, int], float])
    
    @property
    def max_iterations(self) -> int:
        """
        Sets and changes the maximum iterations over the optimiser

        """
        
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, max_iterations) -> None:
        if max_iterations >= 1:
            self._max_iterations = check_type(max_iterations, "max_iterations.setter", int)
        else:
            raise ValueError("Invalid argument in max_iterations.setter: max_iterations must be >= 1")

    @property
    def initial_learning_rate(self) -> Union[float, int]:
        """
        Sets and changes the learning rate of the optimiser

        """
        
        return self._initial_learning_rate

    @initial_learning_rate.setter
    def initial_learning_rate(self, initial_learning_rate) -> None:
        if initial_learning_rate > 0:
            self._initial_learning_rate = check_type(initial_learning_rate, "initial_learning_rate.setter", float, int)
        else:
            raise ValueError("Invalid argument in initial_learning_rate.setter: initial_learning_rate must be > 0")
    
    @property
    def stop_condition(self) -> Callable[..., bool]:
        """
        Sets and changes the stop_condition function used as an alternative check to max_iterations; if max_iterations OR stop_condition
        Set max_iterations to MAX_INT if you want this to be the only condition

        """
        
        return self._objective

    @stop_condition.setter
    def stop_condition(self, stop_condition) -> None:
        self.stop_condition = check_type(stop_condition, "stop_condition.setter", Callable[..., bool])
    
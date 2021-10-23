
from typing import Callable, Union
from fatapi.helpers import check_type
from fatapi.optimisers import Optimiser
import numpy as np

class FISTAOptimiser(Optimiser):
    """
    The Fast Iterative Shrinkage-Thresholding Algorithm gradient descent / ascent optimiser - useful for large-scale dense matrix data
    
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
    autoencoder?: (X: np.ndarray, **kwargs) -> np.ndarray
        Autoencoder which transforms datapoint X to get a more useful counterfactual result by making X closer to a data manifold
        -- Only needed for certain methods (e.g. CEMMethod)
    beta?: Union[float, int]
        Parameter for regularisation / optimisation in CEM
        -- Only needed for certain methods (e.g. CEMMethod)
    
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
    def __init__(self, objective: Callable[[np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray]], np.ndarray]=None, **kwargs):
        self._step_function = self.fista_step_function
        if 'step_function' in kwargs:
            self._step_function = check_type(kwargs.get("step_function"), "__init__", Callable[[np.ndarray, Callable[..., np.ndarray], float, float, int, int, Callable[[float, int, int], float]], np.ndarray])
        self._objective = objective
        self._autoencoder = lambda X, **kwargs: X
        if 'autoencoder' in kwargs:
            self._autoencoder = check_type(type(kwargs.get("autoencoder")), "__init__", Callable[[np.ndarray], np.ndarray])
        self._beta = 1
        if 'beta' in kwargs:
            if kwargs.get('beta') >= 0:
                self._beta = check_type(kwargs.get("beta"), "__init__", float, int)
            else:
                raise ValueError(f"Invalid argument in __init__: beta must be >= 0")
        
    def optimise(self, initial_value: np.ndarray, objective: Callable[..., np.ndarray]=None, max_iterations: int=None, 
                 initial_learning_rate: float=None, decay_function: Callable[[float, int, int], float]=None, **kwargs) -> np.ndarray:
        obj_f = self.objective
        obj_f = self.objective
        max_iter = self.max_iterations
        init_lr = self.initial_learning_rate
        decay_f = self.decay_function
        b = self.beta
        ae = self.autoencoder
        if objective is not None:
            obj_f = objective
        if max_iterations is not None:
            max_iter = max_iterations
        if initial_learning_rate is not None:
            init_lr = initial_learning_rate
        if decay_function is not None:
            decay_f = decay_function
        if 'beta' in kwargs:
            b = check_type(kwargs.get("beta"), "__init__", float, int)
        if 'autoencoder' in kwargs:
            ae = check_type(kwargs.get("autoencoder"), "__init__", Callable[[np.ndarray], np.ndarray])
        lr = init_lr
        d = initial_value
        prev_loss = obj_f(value=d, **kwargs)
        for i in range(max_iter):
            d = self.step_function(initial_value=d, objective=obj_f, prev_loss=prev_loss, learning_rate=lr, iteration=i, max_iterations=max_iter, decay_function=decay_f, **kwargs)
            lr = decay_f(lr, i, max_iter, **kwargs)
            if self.stop_condition(initial_value=d, objective=obj_f, learning_rate=lr, iteration=i, max_iterations=max_iter, **kwargs):
                break
    
    #TODO
    def fista_step_function(self, initial_value: np.ndarray, objective: Callable[..., np.ndarray], prev_loss: np.ndarray, learning_rate: Union[float, int], iteration: int, max_iterations: int, 
                  decay_function: Callable[[float, int, int], float], autoencoder: Callable[[np.ndarray], np.ndarray], beta: Union[float, int], **kwargs) -> np.ndarray:
        #lambda d, obj_f, prev_loss, lr, i, max_iter, decay_f, **kwargs: np.subtract(d, np.array(d.shape()).fill(lr*(obj_f(d, **kwargs)-prev_loss)))
        y_k = initial_value
        objective(y_k, autoencoder=autoencoder, beta=beta, **kwargs)
        return
    
    @property
    def autoencoder(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Sets and changes the autoencoder which transforms X to be closer to the data manifold

        """
        
        return self._autoencoder

    @autoencoder.setter
    def autoencoder(self, autoencoder) -> None:
        self._autoencoder = check_type(autoencoder, "__init__", Callable[[np.ndarray], np.ndarray])

    @property
    def beta(self) -> Union[float, int]:
        """
        Sets and changes the beta variable of the CEM algorithm

        """
        
        return self._beta

    @beta.setter
    def beta(self, beta) -> None:
        if beta >= 0:
            self._beta = check_type(beta, "beta.setter", float, int)
        else:
            raise ValueError("Invalid argument in beta.setter: beta must be >= 0")

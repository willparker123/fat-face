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
    explain()? : (self, X: numpy.array, Y?: numpy.array, predict()?: Callable) -> numpy.array
        Generates counterfactual datapoints from X and Y using predict function or model predict function or argument predict function
    kernel_type? : String
        String specifying which kernel to use to build weights from data - default is "KDE" but can be "KDE"/"KNN"/"Epsilon"/"GS"
        -- Only required if kernel not supplied
    kernel()? : (X_1: numpy.array, X_2: numpy.array, Y_1?: numpy.array, Y_2?: numpy.array) -> Float
        Kernel function to build weights using two datapoints
        -- Only required if kernel_type is not supplied or to override kernel function
    k_neighbours? : Int
        Number of neighbours to take into account when using KNN kernel
        -- Only required if kernel_type is "KNN"
    t_distance? : Float
        Threshold of distance (epsilon) between nodes to constitute a non-zero weight
        -- Default is 0.25
    t_density? : Float
        Threshold of density value between nodes to constitute a non-zero weight when building feasible paths (number of neighbours when kernel_type=="KNN")
        -- Only required if kernel_type is "KDE"
        -- Default is 0.001 
    t_prediction? : Float [0-1]
        Threshold of prediction value from predict function to warrant 
        -- Default is 0.5
    shortest_path() : (X: graph: numpy.array)
    conditions()? : (X_1: numpy.array, X_2: numpy.array, Y_1?: numpy.array, Y_2?: numpy.array) -> Boolean
        Additional conditions which check for feasible paths between nodes - must return a Boolean

    Methods
    -------
    explain() : (X?: numpy.array, Y?: numpy.array, predict()?: Callable) -> numpy.array
        Generates counterfactual datapoints from X and Y using predict function or model predict function or argument predict function
        -- Uses factuals and factuals_target from preprocess_factuals if no X and Y given
    preprocess_factuals() : (factuals?: fatapi.data.Data, factuals_target?: fatapi.data.Data, model?: fatapi.model.Model, 
                                            scaler?: fatapi.model.estimator.Estimator, encoder?: fatapi.model.estimator.Estimator) -> numpy.array
        Uses encoder and scaler from black-box-model or argument to preprocess data as needed.
    build_graph() : (X: numpy.array, Y: numpy.array, t_distance?: Float, t_density?: Float, 
                                    t_prediction?: Float, conditions()?: Callable) -> numpy.array
        Builds graph for distances between nodes in the feature space - returns adjacency matrix of weights between [i,j]; 
        i, j are indicies of datapoints in X (rows)
    get_graph() : () -> numpy.array
        Returns the graph which build_graph() produces
    """
    def __init__(self, *args, **kwargs) -> None:
        super(FACEMethod, self).__init__(*args, **kwargs)
        if not (kwargs.get('kernel') or kwargs.get('kernel_type')):
            raise ValueError(f"Invalid arguments in __init__: please provide kernel or kernel_type")
        ktype = ""
        if kwargs.get('kernel_type'):
            ktype = kwargs.get('kernel_type')
        if not (ktype.lower()=="KDE".lower() or ktype.lower()=="KNN".lower() or ktype.lower()=="E".lower() or ktype.lower()=="GS".lower()):
            raise ValueError(f"Invalid arguments in __init__: kernel_type must be 'KDE', 'KNN', 'E' or 'GS'")
        else:
            self._kernel_type = ktype
            ks = {"kde" : self.kernel_KDE, "knn" : self.kernel_KNN, "e" : self.kernel_E, "gs" : self.kernel_GS}
            self._kernel = ks[self._kernel_type.lower()]
        if kwargs.get('kernel'):
            if callable(kwargs.get('kernel')):
                self._kernel = kwargs.get('kernel')
            else:
                raise ValueError("Invalid argument in __init__: kernel must be a function")
        self.t_distance = 0.25
        self.t_prediction = 0.5
        self.t_density = 0.001
        self.k_neighbours = 3
        if kwargs.get('t_distance'):
            self.t_distance = kwargs.get('t_distance')
        if kwargs.get('t_prediction'):
            self.t_prediction = kwargs.get('t_prediction')
        if kwargs.get('t_density'):
            if not self._kernel_type=="KDE":
                print("Warning in __init__: t_density supplied but kernel may not be KDE")
            self.t_density = kwargs.get('t_density')
        if kwargs.get('k_neighbours'):
            self.k_neighbours = kwargs.get('k_neighbours')
        if self._kernel_type=="KNN":
            if not kwargs.get('k_neighbours'):
                print("Warning in __init__: k_neighbours not supplied - default (3) being used")
        if kwargs.get('conditions'):
            if callable(kwargs.get('conditions')):
                self._conditions = kwargs.get('conditions')
            else:
                raise ValueError("Invalid argument in __init__: conditions must be a function that returns a bool")


    def kernel_KDE():
        return None
        
    def kernel_KNN():
        return None
        
    def kernel_E():
        return None
    #def explain(self, X: np.array=[], Y: np.array=[], predict: Callable=None) -> Union[np.array, Tuple[np.array, np.array]]:
    #    return [[0,0],[0,0]]

from fatapi.model import Model
from fatapi.model.estimator import Estimator
from fatapi.data import Data
from fatapi.helpers import dijkstra
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
    kernel()? : (self: FACEMethod, X_1: numpy.array, X_2: numpy.array, Y_1?: numpy.array, Y_2?: numpy.array) -> Float
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
    shortest_path()? : (X: numpy.array, graph: numpy.array, data: numpy.array)
        Shortest path algorithm to find the path between each instance (row) of X using data 
        (datapoints corresponding to indexes [i, j] of graph) and graph (adjacency matrix)
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
        self._shortest_path = dijkstra
        self._conditions = lambda **kwargs: kwargs
        self._weight_function = lambda x: -np.log(x)
        if kwargs.get('shortest_path'):
            if callable(kwargs.get('shortest_path')):
                self._shortest_path = kwargs.get('shortest_path')
            else:
                raise ValueError(f"Invalid argument in __init__: shortest_path must be a function")
        if kwargs.get('t_distance'):
            if type(kwargs.get('t_distance'))==float or type(kwargs.get('t_distance'))==int:
                self.t_distance = kwargs.get('t_distance')
            else:
                raise ValueError(f"Invalid argument in __init__: t_distance must be a number")
        if kwargs.get('t_prediction'):
            if type(kwargs.get('t_prediction'))==float or type(kwargs.get('t_prediction'))==int:
                if kwargs.get('t_prediction') >= 0 and kwargs.get('t_prediction') < 1:
                    self.t_prediction = kwargs.get('t_prediction')
                else:
                    raise ValueError(f"Invalid argument in __init__: t_prediction must be between 0 and 1")
            else:
                raise ValueError(f"Invalid argument in __init__: t_prediction must be a number")
        if kwargs.get('t_density'):
            if not self._kernel_type=="KDE":
                print("Warning in __init__: t_density supplied but kernel may not be KDE")
            if type(kwargs.get('t_density'))==float or type(kwargs.get('t_density'))==int:
                if kwargs.get('t_density') >= 0 and kwargs.get('t_density') < 1:
                    self.t_density = kwargs.get('t_density')
                else:
                    raise ValueError(f"Invalid argument in __init__: t_density must be between 0 and 1")
            else:
                raise ValueError(f"Invalid argument in __init__: t_density must be a number")
        if kwargs.get('k_neighbours'):
            if type(kwargs.get('k_neighbours'))==int:
                self.k_neighbours = kwargs.get('k_neighbours')
            else:
                raise ValueError("Invalid argument in __init__: k_neighbours must be an integer")
        if self._kernel_type=="KNN":
            if not kwargs.get('k_neighbours'):
                print("Warning in __init__: k_neighbours not supplied - default (3) being used")
        if kwargs.get('conditions'):
            if callable(kwargs.get('conditions')):
                self._conditions = kwargs.get('conditions')
            else:
                raise ValueError("Invalid argument in __init__: conditions must be a function that returns a bool")
        if kwargs.get('weight_function'):
            if callable(kwargs.get('weight_function')):
                self._weight_function = kwargs.get('weight_function')
            else:
                raise ValueError("Invalid argument in __init__: weight_function must be a function that returns a float")
        self._explain = self.explain_FACE
        self.graph = None

    @property
    def shortest_path(self) -> Callable:
        """
        Sets and changes the shortest_path algorithm
        -------
        Callable
        """
        
        return self._shortest_path

    @shortest_path.setter
    def shortest_path(self, shortest_path) -> None:
        if callable(shortest_path):
            self._shortest_path = shortest_path
        else:
            raise ValueError("Invalid argument in shortest_path.setter: shortest_path is not a function")

    @property
    def conditions(self) -> bool:
        """
        Sets and changes the extra conditions feasible paths must pass
        -------
        Callable
        """
        
        return self._conditions

    @conditions.setter
    def conditions(self, conds) -> None:
        if callable(conds):
            self._conditions = conds
        else:
            raise ValueError("Invalid argument in conditions.setter: conditions must be a function that returns a bool")

    @property
    def weight_function(self) -> float:
        """
        Sets and changes the extra weight_function feasible paths must pass
        -------
        Callable
        """
        
        return self._weight_function

    @weight_function.setter
    def weight_function(self, conds) -> None:
        if callable(conds):
            self._weight_function = conds
        else:
            raise ValueError("Invalid argument in weight_function.setter: weight_function must be a function that returns a bool")

    @property
    def kernel(self) -> Callable:
        """
        Sets and changes the kernel algorithm
        -------
        Callable
        """
        
        return self._shortest_path

    @kernel.setter
    def kernel(self, kernel) -> None:
        if callable(kernel):
            self._kernel = kernel
        else:
            raise ValueError("Invalid argument in kernel.setter: kernel is not a function")  

    @property
    def kernel_type(self) -> Callable:
        """
        Sets and changes the kernel_type
        -------
        Callable
        """
        
        return self._kernel_type

    @kernel_type.setter
    def kernel_type(self, kernel_type) -> None:
        if type(kernel_type)==str:
            if kernel_type.lower()=="KDE".lower() or kernel_type.lower()=="KNN".lower() or kernel_type.lower()=="E".lower() or kernel_type.lower()=="GS".lower():
                self._kernel_type = kernel_type
            else:
                raise ValueError("Invalid argument in kernel.setter: kernel_type is not 'kde', 'knn', 'e' or 'gs'") 
        else:
            raise ValueError("Invalid argument in kernel.setter: kernel_type is not a string")  

    def kernel_KDE(self, ):
        return None
        
    def kernel_KNN():
        return None
        
    def kernel_E():
        return None

    def kernel_GS():
        return None

    def check_edge(X_1: np.array, X_2: np.array, weight: float):
        return True

    def get_kernel_image(X: np.array, t_prediction: float, t_density: float, t_distance: float, k_neighbours: int):
        if self.kernel_type=="kde":
                    temp = kernel(X[i], X[j], t_density=t_density, t_distance=t_distance)
                if self.kernel_type=="knn":
                    temp = kernel(X[i], X[j], k_neighbours=k_neighbours, t_distance=t_distance)
                if self.kernel_type=="e":
                    
                if self.kernel_type=="gs":

                else:
                    temp = kernel(X[i], X[j], t_density, t_distance, t_prediction)
                check_
    
    def get_predictions(X: np.array, predictf: Callable):
        n_samples = X.shape[0]
        g = np.zeros([n_samples, 1], dtype=float)
        for x in range(n_samples):
            g[x] = predictf(X[x])
        return g

    def build_graph(self, X: np.array, kernel_image: np.array, predict_image: np.array, shortest_path: Callable, conditions: Callable=True, nsamples: int=None):
        n_samples = X.shape[0]
        if nsamples:
            n_samples = nsamples
        g = np.zeros([n_samples, n_samples], dtype=float)
        for i in range(n_samples):
            for j in range(i):
                if check_edge(X[i], X[j], kernel_image[i, j]):
                    g[i, j] = kernel_image[i, j]
        g = g + g.T - np.diag(np.diag(g))
        return g

    def explain_FACE(self, X: np.array=[], Y: np.array=[], facts: np.array=[], facts_target: np.array=[], t_prediction=None, t_density=None, t_distance=None, k_neighbours=None, shortest_path: Callable=None, predict: Callable=None, conditions: Callable=None) -> Union[np.array, Tuple[np.array, np.array]]:
        if X and not Y:
            raise ValueError("Invalid arguments in explain_FACE: target needed for data; X supplied but not Y")
        if self.factuals and not self.factuals_target and not (X and Y):
            raise ValueError("Invalid arguments in explain_FACE: factuals_target expected for self.factuals; factuals_target not supplied")
        kernel = self.kernel
        cs_f = self.conditions
        if conditions:
            cs_f = conditions
        pred_f = self.predict
        if predict:
            pred_f = predict
        sp_f = self.shortest_path
        if shortest_path:
            sp_f = shortest_path
        t_den = self.t_density
        t_dist = self.t_distance
        t_pred = self.t_prediction
        k_n = self.k_neighbours
        if t_density:
            t_den = t_density
        if t_distance:
            t_dist = t_distance
        if t_prediction:
            t_pred = t_prediction
        if k_neighbours:
            k_n = k_neighbours
        prediction_image = get_predictions(X, pred_f)
        kernel_image = get_kernel_image(X, t_pred, t_den, t_dist, k_n)
        if X and Y:
            graph = self.build_graph(X, kernel_image, prediction_image, sp_f, cs_f)
        else:
            graph = self.build_graph(self.factuals.dataset, kernel_image, prediction_image, sp_f, cs_f, self.factuals.n_data)
        self.graph = graph
        return graph

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
import networkx as nx
import udm as udm


class FACE:
    """Base class for CE methods based on shortest path over a graph

    Attributes:
        data: pandas DataFrame containing data (n_samples, n_features)
        clf: binary classifier with a predict method
        dist_metric: metric for scipy.spatial.distance.cdist function
        dist_threshold: maximum distance between nodes for edge to be added
        pred_threshold: prediction threshold for classification of CE
    """
    def __init__(self, data, y, indexes_ordinal, indexes_nominal, dist_metric, dist_threshold=10, pred_threshold=None):
        """ Inits BaseFACE class
        """
        self.data = data
        self.dist_metric = dist_metric
        self.dist_threshold = dist_threshold
        self.pred_threshold = pred_threshold
        self.indexes_nominal = indexes_nominal
        self.indexes_ordinal = indexes_ordinal
        self.y = y
        self.G = nx.Graph()
        print(self.dist_metric)


    def _threshold_function(self):
        """Distance function that determines whether to add edge to graph

        Args:
            x: array
            y: array

        Returns:
            distance between points
        """
        if self.dist_metric == "euclidean":
            dist = cdist(self.data, self.data, metric=self.dist_metric)
        else:
            dist = udm.categorical_distance(self.data, self.indexes_ordinal, self.indexes_nominal)
        return dist


    def _threshold_function_euclidean(self, x, y):
        """Distance function that determines whether to add edge to graph

        Args:
            x: array
            y: array

        Returns:
            distance between points
        """
        dist = tdist.categorical_distance(self.data, self.indexes_ordinal, self.indexes_nominal)
        return dist


    def prune_nodes(self):
        """Method to remove nodes that do not meet a condition or threshold

        Returns:

        """
        pass

    def prune_edges(self):
        """Method to remove edges that do not meet a threshold

        Returns:

        """
        pass

    def add_nodes_and_edges(self, new_point=False):
        """Creates nodes and edges with weights.

        If new_point is False then creates nodes and edges (if threshold is met) for all data points.
        If now_point is True then creates 1 extra node and adds edges to all other nodes.

        Args:
            new_point: boolean whether a new point has just been added to data

        Returns:

        """

        edge_weights = []

        if new_point is False:
            self.G.add_nodes_from(list(self.data.index))
            print(f'{self.G.number_of_nodes()} nodes have been added to graph.')
            self.prune_nodes()

            dist_matrix = pd.DataFrame(self._threshold_function())

            """weight_matrix = pd.DataFrame(self._weight_function(self.data.values, self.data.values),
                                         index=self.data.index, columns=self.data.index)"""

            nodes = list(self.G.nodes)
            for i, node_from in enumerate(nodes[:-1]):
                for node_to in nodes[i + 1:]:
                    dist = dist_matrix.loc[node_from][node_to]
                    if dist < self.dist_threshold:
                        weight = dist_matrix.loc[node_from][node_to]
                        edge_weights.append((node_from, node_to, {'weight': weight}))

            self.G.add_edges_from(edge_weights)
            print(f'{self.G.number_of_edges()} edges have been added to graph.')
            self.prune_edges()

        else:
            new_node = list(self.G.nodes)[-1] + 1
            self.G.add_node(new_node)
            print(f'1 node has been added to graph.')
            self.prune_nodes()

            dist_matrix = pd.DataFrame(self._threshold_function())
            """weight_matrix = pd.DataFrame(self._weight_function(self.data.values[-1].reshape(1, -1),
                                                               self.data.values[:-1]),
                                         index=[self.data.index[-1]], columns=self.data.index[:-1])"""

            for node_to in list(self.G.nodes)[:-1]:
                dist = dist_matrix.loc[new_node][node_to]
                if dist < self.dist_threshold:
                    weight = dist_matrix.loc[new_node][node_to]
                    edge_weights.append((new_node, node_to, {'weight': weight}))

            self.G.add_edges_from(edge_weights)
            print(f'{len(edge_weights)} edges have been added.')
            self.prune_edges()

    def generate_counterfactual(self, instance, label):
        """Generates counterfactual to flip prediction of example using dijkstra shortest path for the graph created

        Args:
            instance: instance to generate CE

        Returns:
            path to CE as instances from data as a pandas DataFrame
            probability of CE (if prediction_prob specified)

        """
        if self.G.number_of_nodes() == 0:
            self.add_nodes_and_edges()

        instance = instance.reshape(1, -1)
        target = int(abs(label - 1))
        target_data = self.data[self.y == target]
        #target_data = self.y
        #target_data = self.data[self.y['encoded_income'] == target]
        target_nodes = list(set(list(self.G.nodes())).intersection(target_data.index))
        #target_nodes = target_data.indexes

        example_in_data = self.data[self.data.eq(instance)].dropna()
        if len(example_in_data) > 0:
            start_node = example_in_data.iloc[0].name
        else:
            start_node = self.data.iloc[-1].name + 1
            self.data = self.data.append(pd.Series(instance.squeeze(), index=list(self.data), name=start_node))

            self.add_nodes_and_edges(new_point=True)

        assert start_node in list(self.G.nodes), "Instance does not meet thresholds"
        _, path = nx.multi_source_dijkstra(self.G, target_nodes, target=start_node)
        path = path[::-1]


        path_df = self.data.loc[path].reset_index(drop=True)
        #pred = self.clf.predict(path_df).astype(int).reshape(-1, 1)
        #prob_df = pd.DataFrame(pred, columns=['prediction'])

        return path_df


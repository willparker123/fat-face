import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
import networkx as nx
import logging
import importlib as imp
imp.reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')


class BaseFACE:
    """Base class for CE methods based on shortest path over a graph

    Attributes:
        data: pandas DataFrame containing data (n_samples, n_features)
        clf: classifier with a predict method and predict_proba if probabilities wanted
        pred_threshold: prediction threshold for classification of CE
        dist_metric: metric for scipy.spatial.distance.cdist function
        dist_threshold: maximum distance between nodes for edge to be added
    """

    def __init__(
            self,
            data: pd.DataFrame,
            clf,
            pred_threshold: float = None,
            bidirectional: bool = False,
            dist_metric: str = 'euclidean',
            dist_threshold: float = 1
    ):
        self.data = data
        self.clf = clf
        self.prediction = pd.DataFrame(self.clf.predict(data).astype(int), columns=['prediction'], index=data.index)
        self.pred_threshold = pred_threshold
        self.bidirectional = bidirectional
        self.dist_metric = dist_metric
        self.dist_threshold = dist_threshold
        if bidirectional is False:
            self.G = nx.Graph()
        else:
            self.G = nx.DiGraph()
        self.add_nodes_and_edges()
        self.connected_nodes = None

    def _threshold_function(
            self,
            XA: np.ndarray,
            XB: np.ndarray
    ) -> np.ndarray:
        """Function that determines whether to add edge to graph

        Args:
            XA: array containing (n_samples, n_features)
            XB: array containing (n_samples, n_features)

        Returns:
            binary matrix of size len(XA) * len(XB)
        """
        return cdist(XA, XB, metric=self.dist_metric) < self.dist_threshold

    def _weight_function(
            self,
            XA: np.ndarray,
            XB: np.ndarray,
            threshold_matrix: np.ndarray
    ) -> np.ndarray:
        """Distance or density function that calculates weights for graph

        Default uses distance measure for weights but can be overridden in subclasses.

        Args:
            XA: array containing (n_samples, n_features)
            XB: array containing (n_samples, n_features)
            threshold_matrix: binary matrix of size len(XA) * len(XB)

        Returns:
            weight between points
        """
        return cdist(XA, XB, metric=self.dist_metric) * threshold_matrix

    def prune_nodes(self):
        """Method to remove nodes that do not meet a condition or threshold

        Returns:

        """
        unconnected_nodes = [node for node, deg in self.G.degree if deg == 0]
        self.G.remove_nodes_from(unconnected_nodes)
        if len(unconnected_nodes) > 0:
            logging.info(f' {len(unconnected_nodes)} nodes removed as unconnected. Graph now has {len(self.G.nodes)}')

    def prune_edges(self):
        """Method to remove edges that do not meet a threshold

        Returns:

        """
        pass

    def add_nodes_and_edges(
            self,
            new_node: int = None
    ):
        """Creates nodes and edges with weights.

        If new_point is False then creates nodes and edges (if threshold is met) for all data points.
        If now_point is True then creates 1 extra node and adds edges to all other nodes.

        Args:
            new_node: boolean whether a new point has just been added to data

        Returns:

        """
        if new_node is None:
            self.G.add_nodes_from(list(self.data.index))
            logging.info(f' Graph has been created with {self.G.number_of_nodes()} nodes.')

        else:
            self.G.add_node(new_node)
            logging.info(f' 1 node has been added to graph. Graph now has {len(self.G.nodes())} nodes.')

        if new_node is None:
            XA = self.data.loc[list(self.G.nodes)]
            XB = self.data.loc[list(self.G.nodes)]
        else:
            XA = self.data.loc[list(self.G.nodes)].iloc[:-1]
            XB = pd.DataFrame(self.data.loc[list(self.G.nodes)].iloc[-1]).T

        threshold_matrix = self._threshold_function(XA.values, XB.values)
        weight_matrix = pd.DataFrame(self._weight_function(XA.values, XB.values, threshold_matrix), index=XA.index,
                                     columns=XB.index)

        edge_weights = []
        for node_from in weight_matrix.index:
            for node_to in weight_matrix.columns:
                weight = weight_matrix.loc[node_from][node_to]
                if weight != 0:
                    edge_weights.append((node_from, node_to, {'weight': weight}))

        if self.bidirectional is True:
            for node_from in weight_matrix.columns:
                for node_to in weight_matrix.index:
                    weight = weight_matrix.loc[node_from][node_to]
                    if weight != 0:
                        edge_weights.append((node_from, node_to, {'weight': weight}))

        self.G.add_edges_from(edge_weights)
        logging.info(f' {len(edge_weights)} edges have been added to graph.')
        self.prune_edges()
        self.prune_nodes()

    def generate_counterfactual(
            self,
            instance: np.ndarray,
            target_class: int = None
    ) -> (pd.DataFrame, pd.DataFrame):
        """Generates counterfactual to flip prediction of example using dijkstra shortest path for the graph created

        Args:
            instance: instance to generate CE
            target_class: target class for CE, if none then opposite class for binary classification

        Returns:
            path to CE as instances from data as a pandas DataFrame
            probability of CE (if prediction_prob specified)

        """
        logging.info(f' Generating counterfactual for instance {instance} using {self.__class__}.')
        instance = instance.reshape(1, -1)
        assert target_class != self.clf.predict(instance), "Target class is the same as the current prediction."

        example_in_data = self.data[self.data.eq(instance)].dropna()
        if len(example_in_data) > 0:
            start_node = example_in_data.iloc[0].name
        else:
            start_node = list(self.G.nodes)[-1] + 1
            self.data = self.data.append(pd.Series(instance.squeeze(), index=list(self.data), name=start_node))
            self.prediction = self.prediction.append(pd.Series(self.clf.predict(instance).astype(int),
                                                               index=list(self.prediction), name=start_node))
            self.add_nodes_and_edges(start_node)

        assert start_node in list(self.G.nodes), "Instance does not meet thresholds."
        self.connected_nodes = list(nx.node_connected_component(self.G, start_node))

        if target_class is None:
            target_class = np.logical_not(self.clf.predict(instance)).astype(int)
        target_nodes = self.data.loc[self.connected_nodes][(self.prediction.loc[self.connected_nodes] == target_class)
                                                            .values].index

        assert len(target_nodes) > 0, "No target nodes that meet thresholds."
        logging.info(f' {len(target_nodes)} potential counterfactuals.')

        lengths = np.zeros(len(target_nodes))
        paths = []
        for i, target_node in enumerate(target_nodes):
            length, path = nx.single_source_dijkstra(self.G, start_node, target_node)
            lengths[i] = length
            paths.append(path)
        sort_shortest = lengths.argsort()

        i = 0
        if self.pred_threshold is not None:
            pred_probs = self.clf.predict_proba(self.data.loc[paths[sort_shortest[0]][-1]].values.reshape(1, -1))\
                .squeeze()[target_class]
            while pred_probs[-1] < self.pred_threshold:
                assert i < len(target_nodes), "Prediction threshold not met."
                pred_probs = self.clf.predict_proba(self.data.loc[paths[sort_shortest[i]][-1]].values.reshape(1, -1))\
                    .squeeze()[target_class]
                i += 1

        path_df = self.data.loc[paths[sort_shortest[i-1]]]
        pred_df = self.prediction.loc[paths[sort_shortest[i-1]]]

        if self.pred_threshold is not None:
            prob = np.take_along_axis(self.clf.predict_proba(path_df), pred_df.values, axis=1)
            pred_df['probability'] = prob

        return path_df, pred_df

    def plot_path(
            self,
            path: list[int]
    ):
        """Plots the subgraph of nodes that are connected to the instance and shows the path to the CE

        Args:
            path: list of nodes of the shortest path

        Returns:

        """
        fig, ax = plt.subplots(figsize=(12, 12))
        subG = self.G.subgraph(self.connected_nodes)
        pos = nx.drawing.nx_agraph.graphviz_layout(subG, prog="neato")
        nx.draw_networkx(subG, pos=pos, with_labels=False, node_size=200,
                         node_color=[["purple", "y"][self.prediction.loc[node].item()] for node in self.connected_nodes]
                         )
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(subG, pos=pos, edgelist=path_edges, edge_color='r', width=3)
        fig.show()

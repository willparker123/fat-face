from collections import defaultdict
import math
from functools import partial

class Graph():
    def __init__(self, 
                 undirected = True):
        self.edges = defaultdict(list)
        self.weights = {}
        self.undirected = undirected
    
    def add_edge(self, from_node, to_node, weight):
        self.edges[from_node].append(to_node)   
        self.weights[(from_node, to_node)] = weight
        if self.undirected:
            self.edges[to_node].append(from_node)
            self.weights[(to_node, from_node)] = weight
            
def return_value(dictionary, key, path=[]):
    try:
        val = dictionary[key]
        path.append(val)
        return return_value(dictionary, val, path)
    except:
        return path[::-1]
    
def dijsktra(graph, initial, end):
    # shortest paths is a dict of nodes
    # whose value is a tuple of (previous node, weight)
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()
    
    while current_node != end:
        visited.add(current_node)
        destinations = graph.edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
            weight = graph.weights[(current_node, next_node)] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)
        
        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return -1 #"Route Not Possible"
        # next node is the destination with the lowest weight
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])
    
    # Work back through destinations in shortest path
    path = []
    dist = 0
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        if next_node:
            dist += graph.weights[(current_node, next_node)]
        current_node = next_node
        
    # Reverse path
    path = path[::-1]
    return dist, path

def dijsktra_tosome(graph, initial, targets):
    paths = {}; dists = {}
    dijsktra_partial = partial(dijsktra, graph, initial)
    for target in targets:
        sat = dijsktra_partial(target)
        if sat != -1:
            dist, path = sat
            paths[target] = path
            dists[target] = dist
    return dists, paths
        
def dijsktra_toall(graph, initial):
    edges = graph.edges
    weights = graph.weights
 
    nodes_status = {item: [math.inf, 'T'] for item in edges.keys() if item != initial}
    nodes_status[initial] = [0, 'P']

    previous_nodes = {}
    current_node = initial
    while True:
        distance_to_current_node = nodes_status[current_node][0]
        neighbours = edges[current_node]
        for neighbour in neighbours:
            if (current_node, neighbour) in weights.keys():
                if (neighbour in nodes_status.keys() 
                    and nodes_status[neighbour][1] == 'T'):              
                    current_distance = nodes_status[neighbour][0]
                    traversal_cost = weights[(current_node, neighbour)]
                    proposed_distance = distance_to_current_node + traversal_cost
                    if proposed_distance < current_distance:                    
                        nodes_status[neighbour][0] = proposed_distance
                        previous_nodes[neighbour] = current_node

        filtered_weights = {item: value[0] for item, value 
                                    in nodes_status.items() if value[1] == 'T'}
        if not filtered_weights:
            break
        best_node = min(filtered_weights, key=filtered_weights.get)
        nodes_status[best_node][1] = 'P'
        current_node = best_node
      
    paths = {item: return_value(previous_nodes, item, [item]) 
                for item in previous_nodes.keys()}
       
    return nodes_status, paths


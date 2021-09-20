import numpy as np

def dijkstra(graph: np.ndarray, start: int, end: int, start_edges: np.ndarray):
    # shortest paths is a dict of nodes
    # whose value is a tuple of (previous node, weight)#
    shortest_paths = {start: (None, 0)}
    current_node = start
    visited = set()
    
    while current_node != end:
        visited.add(current_node)
        if start < 0:
            destinations = [start_edges.index(x) for x in start_edges if x>0]
        else:
            destinations = [list(graph[current_node, :]).index(x) for x in list(graph[current_node, :]) if x>0]
        weight_to_current_node = shortest_paths[current_node][1]
        
        for next_node in destinations:
            weight = graph[current_node, next_node] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)
        
        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            print(f"Error in dijkstra: cannot find a route from start node [{start}] to end node [{end}]")
            return -1, [] #"Route Not Possible"
        # next node is the destination with the lowest weight
        
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])
    
    # Work back through destinations in shortest path
    path = []
    distance = 0
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        if not next_node==None:
            distance += shortest_paths[current_node][1]
        current_node = next_node
    # Reverse path
    path = path[::-1]

    return (distance, path)

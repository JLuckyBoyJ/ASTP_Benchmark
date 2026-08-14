import numpy as np
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: set, distance_matrix: np.ndarray) -> int:
    """Select the next node to visit from the unvisited nodes."""
    c1, c2, c3, c4 = 0.4, 0.3, 0.2, 0.1
    scores = {}
    for node in unvisited_nodes:
        outgoing = [distance_matrix[node][i] for i in unvisited_nodes if i != node]
        average_outgoing = np.mean(outgoing) if outgoing else 0.0
        std_outgoing = np.std(outgoing) if outgoing else 0.0
        score = (c1 * distance_matrix[current_node][node]
                 - c2 * average_outgoing
                 + c3 * std_outgoing
                 - c4 * distance_matrix[node][destination_node])
        scores[node] = score
    next_node = min(scores, key=scores.get)
    return next_node

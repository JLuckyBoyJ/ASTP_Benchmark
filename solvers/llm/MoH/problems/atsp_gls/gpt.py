# The algorithm updates the edge cost matrix by incorporating a decay for underutilized arcs 
# and an increase based on their usage, balancing the exploration of untried paths while 
# maintaining beneficial connections to encourage searching for better tours.
import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # Create a copy of the edge distance matrix to avoid modifying the original input
    updated_edge_distance = edge_distance.copy()
    n = edge_distance.shape[0]
    decay_factor = 0.1  # Decay factor for underutilized edges
    usage_threshold = 5  # Threshold below which decay is applied
    
    # Extract the edges used in the local optimal tour
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
  
    # Create arrays for vectorized operations
    u_indices = local_opt_tour
    v_indices = np.roll(local_opt_tour, -1)
    edges_used = edge_n_used[u_indices, v_indices]
  
    # Apply decay to edges that have been used less than the threshold
    decay_mask = edges_used < usage_threshold
    updated_edge_distance[u_indices[decay_mask], v_indices[decay_mask]] *= (1 - decay_factor)
  
    # Update the edge distances by adding the usage count
    updated_edge_distance[u_indices, v_indices] += edges_used
    
    return updated_edge_distance

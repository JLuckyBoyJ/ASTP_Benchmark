import numpy as np
def arc_badness(distance_matrix, tour, penalty_count, iteration):
    u = np.asarray(tour)
    v = np.roll(u, -1)
    return distance_matrix[u, v].astype(float)

import numpy as np
def arc_badness(distance_matrix: np.ndarray, tour: np.ndarray, penalty_count: np.ndarray, iteration: int) -> np.ndarray:
    u = np.asarray(tour)
    v = np.roll(u, -1)
    return distance_matrix[u, v].astype(float)

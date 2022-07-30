import numpy as np

from common.grid_search_parallel import grid_search
from common.aggregation_algorithm import L1AggregationAlgorithm
from load_data import load_data

x_est, y_est, x_val, y_val = load_data()

# kernels_ranges = [np.arange(50, 91, 10), np.arange(50, 91, 10), np.arange(0, 11, 10)]
# R_range = np.arange(20, 41, 5)

# kernels_ranges = [np.arange(50, 91, 10), np.arange(50, 91, 10), np.arange(0, 11, 10)]
# R_range = np.arange(10, 21, 5)

# kernels_ranges = [np.arange(40, 101, 10), np.arange(40, 101, 10), np.arange(0, 21, 10)]
# R_range = np.arange(20, 31, 5)

# kernels_ranges = [np.arange(10, 31, 10), np.arange(70, 101, 10), [0]]
# R_range = np.arange(20, 31, 5)

kernels_ranges = [np.arange(10, 31, 10), np.arange(0, 61, 10), [0]]
R_range = np.arange(20, 31, 5)

result = grid_search(L1AggregationAlgorithm, x_est, y_est, x_val, y_val, kernels_ranges, R_range)

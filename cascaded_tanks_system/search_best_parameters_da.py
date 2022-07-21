import numpy as np

from common.grid_search_parallel import grid_search
from common.dual_averaging_algorithm import EntropicDualAveragingAlgorithm
from common.aggregation_algorithm import L1AggregationAlgorithm
from load_data import load_data

x_est, y_est, x_val, y_val = load_data()

kernels_ranges = [np.arange(50, 81, 10), np.arange(50, 81, 10), np.arange(0, 11, 10)]
R_range = np.arange(25, 41, 5)

result = grid_search(EntropicDualAveragingAlgorithm, x_est, y_est, x_val, y_val, kernels_ranges, R_range)


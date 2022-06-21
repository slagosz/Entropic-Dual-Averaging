from common.grid_search import grid_search
from common.dual_averaging_algorithm import EntropicDualAveragingAlgorithm
import numpy as np
from load_data import load_data

x_est, y_est, x_val, y_val = load_data()

kernels_ranges = [np.arange(1, 10), np.arange(0, 5)]
R_range = np.arange(5, 8)

grid_search(EntropicDualAveragingAlgorithm, x_est, y_est, x_val, y_val, kernels_ranges, R_range)

# for l1 aggregation...
# from common.aggregation_algorithm import L1AggregationAlgorithm
# grid_search(L1AggregationAlgorithm, x_est, y_est, x_val, y_val, kernels_ranges, R_range)
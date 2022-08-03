import os

import numpy as np

from common.grid_search_parallel_cv import grid_search
from common.dual_averaging_algorithm import EntropicDualAveragingAlgorithm
from load_data import load_data

x_est, y_est, x_val, y_val = load_data()

kernels_ranges = [np.arange(10, 81, 10), np.arange(10, 81, 10), [0]]
R_range = np.arange(5, 41, 5)
R_range = np.arange(1, 11, 1)

outdir = os.path.join(os.path.dirname(__file__), 'results')
result = grid_search(outdir, EntropicDualAveragingAlgorithm, x_est, y_est, kernels_ranges, R_range, n_jobs=60)

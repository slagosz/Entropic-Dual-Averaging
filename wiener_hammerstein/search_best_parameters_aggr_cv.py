import os

import numpy as np

from common.grid_search_parallel_cv import grid_search
from common.aggregation_algorithm import L1AggregationAlgorithm
from .load_data import load_data


def run():
    x_est, y_est, x_val, y_val = load_data()

    kernels_ranges = [np.arange(10, 81, 10), np.arange(0, 81, 10), [0]]
    R_range = np.concatenate([np.arange(1, 11, 1), np.arange(15, 41, 5)])

    outdir = os.path.join(os.path.dirname(__file__), 'results')
    grid_search(outdir, L1AggregationAlgorithm, x_est, y_est, kernels_ranges, R_range, n_jobs=60)


if __name__ == "__main__":
    run()

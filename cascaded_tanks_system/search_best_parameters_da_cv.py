import os

import numpy as np

from common.grid_search_parallel_cv import grid_search
from common.dual_averaging_algorithm import EntropicDualAveragingAlgorithm
from .load_data import load_data


def run():
    x_est, y_est, x_val, y_val = load_data()

    kernels_ranges = [np.arange(10, 121, 10), np.arange(0, 101, 10), np.arange(0, 21, 10)]
    R_range = np.arange(5, 41, 5)

    outdir = os.path.join(os.path.dirname(__file__), 'results')
    grid_search(outdir, EntropicDualAveragingAlgorithm, x_est, y_est, kernels_ranges, R_range, n_jobs=60)


if __name__ == "__main__":
    run()

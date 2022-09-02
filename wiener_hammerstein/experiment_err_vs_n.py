import os
import numpy as np

from common.experiment_err_vs_n import run_experiment, plot_results
from load_data import load_data


if __name__ == "__main__":
    # %% setup model parameters
    lowest_cv_err_da_parameters = dict(kernels=(30, 0, 0), R=15)
    lowest_val_err_da_parameters = dict(kernels=(40, 20, 0), R=10)

    lowest_cv_err_aggr_parameters = dict(kernels=(50, 30, 0), R=3)
    lowest_val_err_aggr_parameters = dict(kernels=(70, 40, 0), R=3)

    # %% setup experiment parameters
    N_range = np.arange(200, 1001, 100)

    results_directory = os.path.join(os.path.dirname(__file__), 'results')

    results = run_experiment(load_data, N_range, lowest_cv_err_da_parameters, lowest_val_err_da_parameters,
                             lowest_cv_err_aggr_parameters, lowest_val_err_aggr_parameters, results_directory)

    plot_results(results, results_directory)

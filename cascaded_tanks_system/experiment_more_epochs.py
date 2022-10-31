import os

from common.experiment_more_epochs import run_experiment, plot_results
from .load_data import load_data


def run():
    # %% setup model parameters
    lowest_cv_err_da_parameters = dict(kernels=(10, 90, 0), R=35)
    lowest_val_err_da_parameters = dict(kernels=(10, 90, 20), R=30)

    # %% setup experiment parameters
    epochs_range = [1, 2, 3, 4, 5]

    results_directory = os.path.join(os.path.dirname(__file__), 'results')

    fn = 'err_vs_num_of_epochs_lowest_cv_error'
    results = run_experiment(load_data, epochs_range, lowest_cv_err_da_parameters, results_directory, fn)
    plot_results(results, fn)

    fn = 'err_vs_num_of_epochs_lowest_val_error'
    results = run_experiment(load_data, epochs_range, lowest_val_err_da_parameters, results_directory, fn)
    plot_results(results, fn)

if __name__ == "__main__":
    run()

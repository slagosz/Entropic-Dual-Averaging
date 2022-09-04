import os
import numpy as np

from common.experiment_more_epochs import run_experiment, plot_results
from load_data import load_data


if __name__ == "__main__":
    # %% setup model parameters
    da_parameters = dict(kernels=(10, 90, 0), R=35)

    # %% setup experiment parameters
    epochs_range = np.arange(1, 6)

    results_directory = os.path.join(os.path.dirname(__file__), 'results')

    results = run_experiment(load_data, epochs_range, da_parameters, results_directory)

    plot_results(results)

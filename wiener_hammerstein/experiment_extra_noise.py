import os

from .load_data import load_data
from common.experiment_extra_noise import run_experiment, plot_results


def run():
    SNR_range = [25, 20, 15, 10, 5, 1]

    kernels_da = (40, 20, 0)
    R_da = 10
    da_reference_error = 0.0025245513119415983  # calculated in the other experiment

    kernels_aggr = (70, 40, 0)
    R_aggr = 3
    aggr_reference_error = 0.0006744781383219112  # calculated in the other experiment

    results_directory = os.path.join(os.path.dirname(__file__), 'results')

    results = run_experiment(load_data, SNR_range, kernels_da, R_da, kernels_aggr, R_aggr, results_directory)
    plot_results(results, da_reference_error, aggr_reference_error, "wiener_hammerstein")


if __name__ == "__main__":
    run()

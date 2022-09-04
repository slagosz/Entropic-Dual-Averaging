import os

from load_data import load_data
from common.experiment_extra_noise import run_experiment, plot_results

if __name__ == "__main__":
    SNR_range = [25, 20, 15, 10, 5, 1]

    kernels_da = (10, 90, 20)
    R_da = 30
    da_reference_error = 1.0737385190134519  # calculated in the other experiment

    kernels_aggr = (10, 70, 10)
    R_aggr = 25
    aggr_reference_error = 0.881353908668148  # calculated in the other experiment

    results_directory = os.path.join(os.path.dirname(__file__), 'results')

    results = run_experiment(load_data, SNR_range, kernels_da, R_da, kernels_aggr, R_aggr, results_directory)
    plot_results(results, da_reference_error, aggr_reference_error)

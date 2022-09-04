import os
import pickle

import numpy as np
from tqdm import tqdm

from common.run_experiment import estimate_and_validate_DA, estimate_and_validate_l1_aggregation
import matplotlib.pyplot as plt


def generate_results(load_data_function, SNR_range, kernels_da, R_da, kernels_aggr, R_aggr, num_of_experiments=5, seed=34):
    x_est, y_est, x_val, y_val = load_data_function()

    P_output = np.mean(np.array(y_est) ** 2)
    extra_noise_variance_range = P_output / np.array(SNR_range)

    errors_da = np.zeros(len(SNR_range))
    errors_aggr = np.zeros(len(SNR_range))

    N = len(x_est)

    np.random.seed(seed)

    for _ in tqdm(range(0, num_of_experiments)):
        noise_signal = np.random.randn(N)

        for i, variance in enumerate(extra_noise_variance_range):
            z = noise_signal * np.sqrt(variance)
            y_est_noisy = y_est + z

            err_da, _, _ = estimate_and_validate_DA(x_est, y_est_noisy, x_val, y_val, kernels_da, R_da)
            err_aggr, _, _ = estimate_and_validate_l1_aggregation(x_est, y_est_noisy, x_val, y_val, kernels_aggr, R_aggr)

            errors_da[i] = err_da / num_of_experiments
            errors_aggr[i] = err_aggr / num_of_experiments

    results = dict(errors_da=errors_da, errors_aggr=errors_aggr, kernels_da=kernels_da, R_da=R_da,
                   kernels_aggr=kernels_aggr, R_aggr=R_aggr, SNR_range=SNR_range,
                   extra_noise_variance_range=extra_noise_variance_range)

    return results


def plot_results(results, da_reference_error, aggr_reference_error):
    plt.close()
    plt.style.use('../common/style.mplstyle')

    plt.figure(figsize=(3.7, 2.4))

    SNR_range = results['SNR_range']
    errors_da = results['errors_da']
    errors_aggr = results['errors_aggr']

    plt.plot(SNR_range, errors_da / da_reference_error, '.-')
    plt.plot(SNR_range, errors_aggr / aggr_reference_error, '.--')

    plt.xlabel('SNR')
    plt.ylabel(r'$\textrm{err / err}_0$')
    plt.legend(['EDA', 'CA'])
    plt.grid()

    plt.savefig(f'err_extra_noise.pdf')


def run_experiment(load_data_function, SNR_range, kernels_da, R_da, kernels_aggr, R_aggr, results_directory):
    filename = f'extra_noise_results.pz'
    fp = os.path.join(results_directory, filename)

    if os.path.isfile(fp):
        print(f"Loading results from file...")
        with open(fp, 'rb') as f:
            results = pickle.load(f)
    else:
        results = generate_results(load_data_function, SNR_range, kernels_da, R_da, kernels_aggr, R_aggr)
        with open(fp, 'wb') as f:
            pickle.dump(results, f)

    return results

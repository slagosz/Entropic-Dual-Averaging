import os
import pickle

import numpy as np
from tqdm import tqdm

from common.run_experiment import estimate_and_validate_DA, estimate_and_validate_l1_aggregation
import matplotlib.pyplot as plt
from load_data import load_data


def run_experiment(extra_noise_variance_range, kernels_da, R_da, kernels_aggr, R_aggr, num_of_experiments=5, seed=34):
    x_est, y_est, x_val, y_val = load_data()

    errors_da = {}
    errors_aggr = {}

    N = len(x_est)

    np.random.seed(seed)

    for _ in tqdm(range(0, num_of_experiments)):
        noise_signal = np.random.randn(N)
        for variance in extra_noise_variance_range:
            z = noise_signal * np.sqrt(variance)
            y_est_noisy = y_est + z

            err_da, _, _ = estimate_and_validate_DA(x_est, y_est_noisy, x_val, y_val, kernels_da, R_da)
            err_aggr, _, _ = estimate_and_validate_l1_aggregation(x_est, y_est_noisy, x_val, y_val, kernels_aggr, R_aggr)

            errors_da[variance] = errors_da.get(variance, 0) + err_da / num_of_experiments
            errors_aggr[variance] = errors_aggr.get(variance, 0) + err_aggr / num_of_experiments

    results = dict(errors_da=errors_da, errors_aggr=errors_aggr, kernels_da=kernels_da, R_da=R_da,
                   kernels_aggr=kernels_aggr, R_aggr=R_aggr)

    return results


def plot_results(results):
    plt.close()
    plt.style.use('../common/style.mplstyle')

    plt.figure(figsize=(3.7, 2.4))

    sigma, err_da = zip(*sorted(results['errors_da'].items()))
    plt.plot(sigma, err_da, '.-')

    sigma, err_aggr = zip(*sorted(results['errors_aggr'].items()))
    plt.plot(sigma, err_aggr, '.--')

    plt.xlabel('$\sigma_Z^2$')
    plt.ylabel('err')
    plt.legend(['EDA', 'CVXAGGR'])
    plt.grid()

    plt.savefig(f'err_extra_noise.pdf')


if __name__ == "__main__":
    _, y_est, _, _ = load_data()

    P_output = np.mean(np.array(y_est) ** 2)
    SNR_range = [25, 20, 15, 10, 5]
    extra_noise_variance_range = P_output / np.array(SNR_range)
    # extra_noise_variance_range = [0, 0.125, 0.25, 0.5, 1, 1.5, 2, 3]

    kernels_da = (10, 30, 0)
    R_da = 10
    kernels_aggr = (30, 50, 0)
    R_aggr = 6

    results_directory = os.path.join(os.path.dirname(__file__), 'results')
    filename = f'extra_noise_results.pz'
    fp = os.path.join(results_directory, filename)

    if os.path.isfile(fp):
        print(f"Loading results from file...")
        with open(fp, 'rb') as f:
            results = pickle.load(f)
    else:
        results = run_experiment(extra_noise_variance_range, kernels_da, R_da, kernels_aggr, R_aggr)
        with open(fp, 'wb') as f:
            pickle.dump(results, f)

    plot_results(results)

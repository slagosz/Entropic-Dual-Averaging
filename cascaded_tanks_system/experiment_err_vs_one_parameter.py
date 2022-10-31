import json
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import common.config
from common.run_experiment import estimate_and_validate_DA, estimate_and_validate_l1_aggregation
from .load_data import load_data


def run():
    # %% load data
    x_est, y_est, x_val, y_val = load_data()

    # algorithm parameters
    algorithm_choice = 'aggregation' # or  'DA'

    if algorithm_choice == 'DA':
        estimate_and_validate_function = estimate_and_validate_DA
        best_kernels = (60, 80, 0)
        best_R = 30
    else:
        estimate_and_validate_function = estimate_and_validate_l1_aggregation
        best_kernels = (10, 70)
        best_R = 25

    # change R, kernels fixed
    offset = 10
    jump = 5
    R_range = np.arange(best_R - offset, best_R + offset + 1, jump, dtype=int)

    errors_R = {}
    for R in tqdm(R_range):
        errors_R[str(R)], _, _ = estimate_and_validate_function(x_est, y_est, x_val, y_val, best_kernels, R)

    with open(f'errors_R_{algorithm_choice}.json', 'w') as f:
        json.dump(errors_R, f)

    #%% plot errors
    plt.close()
    plt.style.use(common.config.STYLE_FP)

    plt.figure(figsize=(3.7, 2.4))
    plt.plot(errors_R.keys(), errors_R.values(), '.-')
    plt.xlabel('R')
    plt.ylabel('err')
    plt.grid()

    plt.savefig(os.path.join(common.config.PLOTS_DIR, f'err_R_{algorithm_choice}.pdf'))

    #%% change the first order kernel, other parameters are fixed
    offset = 10
    jump = 5
    best_1st_order_kernel = best_kernels[0]
    kernel_1st_order_range = np.arange(best_1st_order_kernel - offset, best_1st_order_kernel + offset + 1, jump)

    errors_kernel = {}
    for kernel_1st_order in tqdm(kernel_1st_order_range):
        kernels = list(best_kernels)
        kernels[0] = kernel_1st_order
        errors_kernel[str(kernel_1st_order)], _, _ = estimate_and_validate_function(x_est, y_est, x_val, y_val, kernels, best_R)

    with open(f'errors_1st_order_kernel_{algorithm_choice}.json', 'w') as f:
        json.dump(errors_kernel, f)

    #%% plot errors
    plt.close()
    plt.style.use(common.config.STYLE_FP)

    plt.figure(figsize=(3.7, 2.4))
    plt.plot(errors_kernel.keys(), errors_kernel.values(), '.-')
    plt.xlabel('1st order kernel')
    plt.ylabel('err')
    plt.grid()

    plt.savefig(os.path.join(common.config.PLOTS_DIR, f'err_1st_order_kernel_{algorithm_choice}.pdf'))


if __name__ == "__main__":
    run()

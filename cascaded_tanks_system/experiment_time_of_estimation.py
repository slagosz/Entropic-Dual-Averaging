import os.path
import pickle
import math

import matplotlib.pyplot as plt
import numpy as np

from common.aggregation_algorithm import L1AggregationAlgorithm
from common.dual_averaging_algorithm import EntropicDualAveragingAlgorithm
from common.run_experiment import estimate_and_validate_volterra_model
from load_data import load_data


def measure_estimation_time(x_est, y_est, kernels, algorithm_class, R, num_of_experiments):
    execution_time_list = []
    for _ in range(0, num_of_experiments):
        _, execution_time, _ = estimate_and_validate_volterra_model(x_est, y_est, None, None, kernels, algorithm_class, R)
        execution_time_list.append(execution_time)

    return np.mean(execution_time_list)


def run_experiment(N_list, model_memory_length_list, num_of_experiments=5):
    x_est, y_est, _, _ = load_data()
    R = 1

    D_list = [math.comb(M, 2) + M + 1 for M in model_memory_length_list]

    results = dict(N_list=N_list, D_list=D_list)

    for algorithm_class in [L1AggregationAlgorithm, EntropicDualAveragingAlgorithm]:
        results[algorithm_class.__name__ + '_var_N'] = []
        results[algorithm_class.__name__ + '_var_D'] = []

        for N_idx, N in enumerate(N_list):
            print(f'N = {N}')
            M = model_memory_length_list[-1]
            kernels = [0, M]
            avg_execution_time = measure_estimation_time(x_est[:N], y_est[:N], kernels, algorithm_class, R,
                                                         num_of_experiments)
            results[algorithm_class.__name__ + '_var_N'].append(avg_execution_time)

        for M_idx, M in enumerate(model_memory_length_list):
            print(f'M = {M}')

            N = N_list[-1]
            kernels = [0, M]
            avg_execution_time = measure_estimation_time(x_est[:N], y_est[:N], kernels, algorithm_class, R,
                                                         num_of_experiments)

            results[algorithm_class.__name__ + '_var_D'].append(avg_execution_time)

    return results


def plot_results(results):
    plt.style.use('../common/style.mplstyle')

    N_list = results['N_list']
    D_list = results['D_list']

    results_eda_var_N = results[EntropicDualAveragingAlgorithm.__name__ + '_var_N']
    results_eda_var_D = results[EntropicDualAveragingAlgorithm.__name__ + '_var_D']
    results_aggr_var_N = results[L1AggregationAlgorithm.__name__ + '_var_N']
    results_aggr_var_D = results[L1AggregationAlgorithm.__name__ + '_var_D']

    plt.figure(figsize=(3.7, 2.0))
    plt.plot(N_list, results_eda_var_N, '.-')
    plt.plot(N_list, results_aggr_var_N, '.--')
    plt.xlabel('N')
    plt.ylabel('time of estimation [s]')
    plt.legend(['EDA', 'CA'])
    plt.grid()

    plt.savefig('time_var_N.pdf')

    plt.close()

    plt.figure(figsize=(3.7, 2.0))
    plt.plot(D_list, results_eda_var_D, '.-')
    plt.plot(D_list, results_aggr_var_D, '.--')
    plt.xlabel('D')
    plt.ylabel('time of estimation [s]')
    plt.legend(['EDA', 'CA'])
    plt.grid()

    plt.savefig('time_var_D.pdf')


if __name__ == "__main__":
    N_list = np.arange(200, 1001, 100)
    model_memory_length_list = np.arange(40, 101, 10)

    results_directory = os.path.join(os.path.dirname(__file__), 'results')
    filename = f'estimation_times.pz'
    fp = os.path.join(results_directory, filename)

    if os.path.isfile(fp):
        print(f"Loading results from file...")
        with open(fp, 'rb') as f:
            results = pickle.load(f)
    else:
        results = run_experiment(N_list, model_memory_length_list)
        with open(fp, 'wb') as f:
            pickle.dump(results, f)

    plot_results(results)

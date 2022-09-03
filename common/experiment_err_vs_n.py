import os
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from common.run_experiment import estimate_and_validate_DA, estimate_and_validate_l1_aggregation


def generate_results(load_data_function, N_range, lowest_cv_err_da_parameters, lowest_val_err_da_parameters,
                     lowest_cv_err_aggr_parameters, lowest_val_err_aggr_parameters):
    x_est, y_est, x_val, y_val = load_data_function()

    e_da_cv = {}
    e_da_val = {}
    e_aggr_cv = {}
    e_aggr_val = {}

    N_range = N_range.tolist()
    for N in tqdm(N_range):
        x_est_sliced = x_est[:N]
        y_est_sliced = y_est[:N]

        e_da_cv[int(N)], _, _ = estimate_and_validate_DA(x_est_sliced, y_est_sliced, x_val, y_val,
                                                    lowest_cv_err_da_parameters['kernels'],
                                                    lowest_cv_err_da_parameters['R'])

        e_da_val[int(N)], _, _ = estimate_and_validate_DA(x_est_sliced, y_est_sliced, x_val, y_val,
                                                     lowest_val_err_da_parameters['kernels'],
                                                     lowest_val_err_da_parameters['R'])

        e_aggr_cv[int(N)], _, _ = estimate_and_validate_l1_aggregation(x_est_sliced, y_est_sliced, x_val, y_val,
                                                                  lowest_cv_err_aggr_parameters['kernels'],
                                                                  lowest_cv_err_aggr_parameters['R'])

        e_aggr_val[int(N)], _, _ = estimate_and_validate_l1_aggregation(x_est_sliced, y_est_sliced, x_val, y_val,
                                                                   lowest_val_err_aggr_parameters['kernels'],
                                                                   lowest_val_err_aggr_parameters['R'])

    return dict(err_da_cv=e_da_cv, err_da_val=e_da_val, err_aggr_cv=e_aggr_cv, err_aggr_val=e_aggr_val, N_range=N_range)


def plot_results(results, results_directory):
    style_filepath = os.path.join(os.path.dirname(__file__), 'style.mplstyle')

    # plot algorithms' errors
    plt.close()
    plt.style.use(style_filepath)

    plt.figure(figsize=(3.7, 2.4))
    plt.plot(results['N_range'], results['err_da_cv'].values(), '.-')
    plt.plot(results['N_range'], results['err_aggr_cv'].values(), '.--')
    plt.plot(results['N_range'], results['err_da_val'].values(), 'o-')
    plt.plot(results['N_range'], results['err_aggr_val'].values(), 'o--')
    plt.xlabel('N')
    plt.ylabel('err')
    plt.legend(['Entropic DA ', '$\ell_{1}$ convex aggregation', 'Entropic DA', '$\ell_{1}$ convex aggregation'])
    plt.grid()

    plt.savefig(os.path.join(results_directory, 'err_vs_n.pdf'))


def run_experiment(load_data_function, N_range, lowest_cv_err_da_parameters, lowest_val_err_da_parameters,
                   lowest_cv_err_aggr_parameters, lowest_val_err_aggr_parameters, results_directory):
    results_filename = 'err_vs_n.json'
    results_fp = os.path.join(results_directory, results_filename)
    if os.path.isfile(results_fp):
        print(f"Loading results from file {results_directory}")
        with open(results_fp, 'r') as f:
            results = json.load(f)
    else:
        results = generate_results(load_data_function, N_range, lowest_cv_err_da_parameters, lowest_val_err_da_parameters,
                                   lowest_cv_err_aggr_parameters, lowest_val_err_aggr_parameters)
        os.makedirs(os.path.dirname(results_fp), exist_ok=True)
        with open(results_fp, 'w') as f:
            json.dump(results, f)

    return results

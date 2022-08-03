import os
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from common.run_experiment import estimate_and_validate_DA, estimate_and_validate_l1_aggregation


def generate_results(load_data_function, N_range, kernels_da, R_da, kernels_aggr, R_aggr):
    x_est, y_est, x_val, y_val = load_data_function()

    e_da = {}
    e_aggr = {}

    time_da = {}
    time_aggr = {}

    y_mod_da = None
    y_mod_aggr = None

    for N in tqdm(N_range):
        x_est_sliced = x_est[:N-1]
        y_est_sliced = y_est[:N-1]

        e_da[N], time_da[N], y_mod_da = estimate_and_validate_DA(x_est_sliced, y_est_sliced, x_val, y_val, kernels_da,
                                                                 R_da)
        e_aggr[N], time_aggr[N], y_mod_aggr = estimate_and_validate_l1_aggregation(x_est_sliced, y_est_sliced, x_val,
                                                                                   y_val, kernels_aggr, R_aggr)

    return dict(err_da=e_da, time_da=time_da, err_aggr=e_aggr, time_aggr=time_aggr, y_mod_da=list(y_mod_da),
                y_mod_aggr=list(y_mod_aggr), N_range=N_range, y_val=list(y_val))


def plot_results(results, results_directory):
    style_filepath = os.path.join(os.path.dirname(__file__), 'style.mplstyle')

    # plot models' outputs
    plt.close()
    plt.style.use(style_filepath)

    plt.figure(figsize=(3.8, 2.4))
    plt.plot(results['y_mod_da'])
    plt.plot(results['y_mod_aggr'], '--')
    plt.plot(results['y_val'], '-.')
    plt.xlabel('t')
    plt.ylabel('output')
    plt.legend(['Entropic DA', '$\ell_{1}$ convex aggregation', 'True system'])
    plt.grid()

    plt.savefig(os.path.join(results_directory, 'output.pdf'))

    # plot algorithms' errors
    plt.close()
    plt.style.use(style_filepath)

    plt.figure(figsize=(3.7, 2.4))
    plt.plot(results['N_range'], results['err_da'].values(), '.-')
    plt.plot(results['N_range'], results['err_aggr'].values(), '--.')
    plt.xlabel('N')
    plt.ylabel('err')
    plt.legend(['Entropic DA', '$\ell_{1}$ convex aggregation'])
    plt.grid()

    plt.savefig(os.path.join(results_directory, 'err.pdf'))

    # plot algorithms' times of execution
    t_da = sorted(results['time_da'].items())
    t_aggr = sorted(results['time_aggr'].items())

    t, t_da = zip(*t_da)
    t2, t_aggr = zip(*t_aggr)

    plt.close()
    plt.style.use(style_filepath)

    plt.figure(figsize=(3.7, 2.4))
    plt.plot(t, t_da, '.-')
    plt.plot(t2, t_aggr, '.--')
    plt.xlabel('N')
    plt.ylabel('time of estimation [s]')
    plt.legend(['Entropic DA', '$\ell_{1}$ convex aggregation'])
    plt.grid()

    plt.savefig(os.path.join(results_directory, 'time.pdf'))


def run_experiment(load_data_function, N_range, kernels_da, R_da, kernels_aggr, R_aggr, results_directory):
    results_filename = 'err_vs_n.json'
    results_fp = os.path.join(results_directory, results_filename)
    if os.path.isfile(results_fp):
        print(f"Loading results from file {results_directory}")
        with open(results_fp, 'r') as f:
            results = json.load(f)
    else:
        results = generate_results(load_data_function, N_range, kernels_da, R_da, kernels_aggr, R_aggr)
        os.makedirs(os.path.dirname(results_fp), exist_ok=True)
        with open(results_fp, 'w') as f:
            json.dump(results, f)

    plot_results(results, results_directory)

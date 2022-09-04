import os
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from common.run_experiment import estimate_and_validate_DA, estimate_and_validate_l1_aggregation


def generate_results(load_data_function, epochs_range, da_parameters):
    x_est, y_est, x_val, y_val = load_data_function()

    errors = []

    for num_of_epochs in tqdm(epochs_range):
        error, _, _ = estimate_and_validate_DA(x_est, y_est, x_val, y_val,
                                               da_parameters['kernels'],
                                               da_parameters['R'],
                                               num_of_epochs=num_of_epochs)
        errors.append(error)

    return dict(errors=errors, epochs_range=list(epochs_range))


def plot_results(results):
    style_filepath = os.path.join(os.path.dirname(__file__), 'style.mplstyle')

    plt.close()
    plt.style.use(style_filepath)

    # plt.figure(figsize=(3.7, 2.4))
    legend = []
    for err, epochs in zip(results['errors'], results['epochs_range']):
        plt.plot(err, epochs, '.-')
        legend.append(str(epochs))

    plt.xlabel('num of epochs')
    plt.ylabel('err')
    plt.legend(legend)
    plt.grid()

    plt.savefig(os.path.join('err_vs_num_of_epochs.pdf'))


def run_experiment(load_data_function, epochs_range, da_parameters, results_directory):
    results_filename = 'err_more_epochs.json'
    results_fp = os.path.join(results_directory, results_filename)
    if os.path.isfile(results_fp):
        print(f"Loading results from file {results_directory}")
        with open(results_fp, 'r') as f:
            results = json.load(f)
    else:
        results = generate_results(load_data_function, epochs_range, da_parameters)
        os.makedirs(os.path.dirname(results_fp), exist_ok=True)
        with open(results_fp, 'w') as f:
            json.dump(results, f)

    return results

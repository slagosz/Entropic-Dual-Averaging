import os

from common.experiment_err_vs_n import run_experiment
from load_data import load_data

# %% setup model parameters

best_kernels_da = (80, 30, 0)
best_R_da = 5

best_kernels_aggr = (10, 80, 0)
best_R_aggr = 5

# %% setup experiment parameters

N_range = [100, 200, 300, 400, 500]

results_directory = os.path.join(os.path.dirname(__file__), 'results')

run_experiment(load_data, N_range, best_kernels_da, best_R_da, best_kernels_aggr, best_R_aggr,
               results_directory)

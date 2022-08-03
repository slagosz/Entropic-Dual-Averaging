import os

from common.experiment_err_vs_n import run_experiment
from load_data import load_data

# %% setup model parameters

best_kernels_da = (10, 90, 0)
best_R_da = 35

best_kernels_aggr = (110, 10)
best_R_aggr = 35

# %% setup experiment parameters

N_range = [256, 384, 512, 640, 768, 896, 1024]

results_directory = os.path.join(os.path.dirname(__file__), 'results')

run_experiment(load_data, N_range, best_kernels_da, best_R_da, best_kernels_aggr, best_R_aggr,
               results_directory)

import itertools
import json
import time
from joblib import Parallel, delayed
from tqdm import tqdm

from common.run_experiment import estimate_and_validate_volterra_model


def helper_func(x_est, y_est, x_val, y_val, kernels, algorithm_class, R):
    results_key = "kernels=" + str(kernels) + "_R=" + str(R)
    error, _, _ = estimate_and_validate_volterra_model(x_est, y_est, x_val, y_val, kernels, algorithm_class, R)

    return error, results_key


def grid_search(algorithm_class, x_est, y_est, x_val, y_val, kernels_ranges, R_range):
    all_kernels = list(itertools.product(*kernels_ranges))
    results = Parallel(n_jobs=8)(delayed(helper_func)(x_est, y_est, x_val, y_val, kernels, algorithm_class, R)
                                                      for kernels in tqdm(all_kernels) for R in R_range)

    errors = dict((key, error) for error, key in results)

    best_kernels_r = min(errors, key=errors.get)
    print(f"Lowest error for {best_kernels_r}. Its value = {errors[best_kernels_r]}.")

    timestr = time.strftime("%Y%m%d-%H%M%S")
    with open(f'error_{algorithm_class.__name__}_{timestr}.json', 'w') as f:
        json.dump(dict(errors=errors, best_params=best_kernels_r, best_params_error=errors[best_kernels_r]), f)

    return best_kernels_r

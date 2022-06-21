import itertools
import json
import time

from common.run_experiment import estimate_and_validate_volterra_model


def grid_search(algorithm_class, x_est, y_est, x_val, y_val, kernels_ranges, R_range):
    errors = {}

    all_kernels = list(itertools.product(*kernels_ranges))
    for i, kernels in enumerate(all_kernels):
        print(f"kernels = {kernels}, [iteration {i+1}/{len(all_kernels)}]")
        for R in R_range:
            results_key = "kernels=" + str(kernels) + "_R=" + str(R)
            errors[results_key], _, _ = estimate_and_validate_volterra_model(x_est, y_est, x_val, y_val, kernels,
                                                                             algorithm_class, R)

    best_kernels_r = min(errors, key=errors.get)
    print(f"Lowest error for {best_kernels_r}. Its value = {errors[best_kernels_r]}.")

    timestr = time.strftime("%Y%m%d-%H%M%S")
    with open(f'error_{timestr}.json', 'w') as f:
        json.dump(dict(errors=errors, best_params=best_kernels_r, best_params_error=errors[best_kernels_r]), f)

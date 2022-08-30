import itertools
import pickle
import os.path

from joblib import Parallel, delayed
from tqdm import tqdm

from common.run_experiment import estimate_and_validate_volterra_model


def helper_func(outdir, x_est, y_est, x_val, y_val, kernels, algorithm_class, R):
    filename = f'val_error_{algorithm_class.__name__}_kernels={kernels}_R={R}.pz'
    fp = os.path.join(outdir, filename)

    if os.path.isfile(fp):
        print(f"Results for kernels={kernels},  R={R} already exist, skipping computations...")
        with open(fp, 'rb') as f:
            results = pickle.load(f)

        return results

    error, _, _ = estimate_and_validate_volterra_model(x_est, y_est, x_val, y_val, kernels, algorithm_class, R)
    result = dict(kernels=kernels, R=R, val_error=error)

    with open(fp, 'wb') as f:
        pickle.dump(result, f)

    return result


def grid_search(outdir, algorithm_class, x_est, y_est, x_val, y_val, kernels_ranges, R_range, n_jobs=-1):
    all_kernels = list(itertools.product(*kernels_ranges))
    results = Parallel(n_jobs=n_jobs)(delayed(helper_func)(outdir, x_est, y_est, x_val, y_val, kernels, algorithm_class, R)
                                      for kernels in tqdm(all_kernels) for R in R_range)

    best_result = min(results, key=lambda r: r['val_error'])

    print(f"Lowest validation error = {best_result['avg_error']} for kernels = {best_result['kernels']}, R = {best_result['R']}")

    return best_result

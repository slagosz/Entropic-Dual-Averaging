import itertools
import pickle
import os.path

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from common.run_experiment import estimate_and_validate_volterra_model_on_many_datasets


def split_dataset(x, y, model_memory_len, folds_num=5):
    offset = model_memory_len - 1
    N = len(y) - offset  # effective number of samples
    fold_len = N // folds_num

    assert fold_len >= model_memory_len

    folds_intervals = [(i * fold_len + offset, (i + 1) * fold_len + offset - 1) for i in range(0, folds_num - 1)]
    folds_intervals.append((folds_intervals[-1][1] + 1, len(y)))
    initial_condition_intervals = [(fold_start - offset, fold_start - 1) for fold_start, _ in folds_intervals]

    datasets = []
    for folds_interval, initial_condition_interval in zip(folds_intervals, initial_condition_intervals):
        x_fold = x[folds_interval[0]:folds_interval[1]+1]
        y_fold = y[folds_interval[0]:folds_interval[1]+1]
        x0_fold = x[initial_condition_interval[0]:initial_condition_interval[1]+1]
        ds = dict(x=x_fold, y=y_fold, x0=x0_fold)
        datasets.append(ds)

    return datasets


def helper_func(outdir, x_est, y_est, kernels, algorithm_class, R):
    errors_folds = []

    model_memory_len = np.max(kernels)
    datasets = split_dataset(x_est, y_est, model_memory_len)

    for validation_ds_index, validation_dataset in enumerate(datasets):
        training_datasets = [ds for ind, ds in enumerate(datasets) if ind != validation_ds_index]

        error, _, _ = estimate_and_validate_volterra_model_on_many_datasets(training_datasets, validation_dataset,
                                                                            kernels, algorithm_class, R)
        errors_folds.append(error)

    avg_error = np.average(errors_folds)
    result = dict(kernels=kernels, R=R, errors_folds=errors_folds, avg_error=avg_error)

    print(f"kernels = {kernels}, R = {R}, avg errr = {avg_error}")

    filename = f'error_{algorithm_class.__name__}_kernels={kernels}_R={R}.json'
    fp = os.path.join(outdir, filename)

    with open(fp, 'wb') as f:
        pickle.dump(result, f)

    return result


def grid_search(outdir, algorithm_class, x_est, y_est, kernels_ranges, R_range, n_jobs=-1):
    all_kernels = list(itertools.product(*kernels_ranges))
    results = Parallel(n_jobs=n_jobs)(delayed(helper_func)(outdir, x_est, y_est, kernels, algorithm_class, R)
                                                           for kernels in tqdm(all_kernels) for R in R_range)

    best_result = min(results, key=lambda r: r['avg_error'])

    print(f"Lowest avg error = {best_result['avg_error']} for kernels = {best_result['kernels']}, R = {best_result['R']}")

    return best_result

import copy
import numpy as np
import time

from common.aggregation_algorithm import L1AggregationAlgorithm
from common.dual_averaging_algorithm import EntropicDualAveragingAlgorithm
from common.volterra_model import VolterraModel


def preprocess_input_signals(x_est, x_val):
    scale_parameter = np.abs(np.max(x_est))
    x_est_scaled = x_est / scale_parameter
    x_val_scaled = x_val / scale_parameter

    return x_est_scaled, x_val_scaled


def preprocess_datasets(training_datasets, validation_dataset):
    training_input = np.concatenate([np.concatenate([ds['x'], ds['x0']]) for ds in training_datasets])
    scale_parameter = 1 / np.abs(np.max(training_input))

    training_datasets_scaled = copy.deepcopy(training_datasets)
    validation_dataset_scaled = copy.deepcopy(validation_dataset)

    for ds in training_datasets_scaled + [validation_dataset_scaled]:
        ds['x'] *= scale_parameter
        ds['x0'] *= scale_parameter

    return training_datasets_scaled, validation_dataset_scaled


def prepare_training_data(x, y, model_memory_len):
    x_sliced = x[model_memory_len - 1:]
    x0 = x[0: model_memory_len - 1]
    y_sliced = y[model_memory_len - 1:]

    return x_sliced, x0, y_sliced


def estimate_and_validate_volterra_model(x_est, y_est, x_val, y_val, kernels, algorithm_class, R, **kwargs):
    model_memory_len = np.max(kernels)
    model = VolterraModel(kernels=kernels)

    x_est, x_val = preprocess_input_signals(x_est, x_val)
    x, x0, y = prepare_training_data(x_est, y_est, model_memory_len)

    algorithm = algorithm_class(model.dictionary, R)

    # train model
    start = time.time()
    model_parameters = algorithm.run(x, y, x0=x0, **kwargs)
    end = time.time()
    execution_time = end - start

    # validate model
    model.set_parameters(model_parameters)
    y_mod = model.evaluate_output(x_val)
    error = 1 / len(x_val) * np.sum((y_mod - y_val) ** 2)

    return error, execution_time, y_mod


def estimate_and_validate_DA(x_est, y_est, x_val, y_val, kernels, R, **kwargs):
    algorithm = EntropicDualAveragingAlgorithm

    return estimate_and_validate_volterra_model(x_est, y_est, x_val, y_val, kernels, algorithm, R, **kwargs)


def estimate_and_validate_l1_aggregation(x_est, y_est, x_val, y_val, kernels, R):
    algorithm = L1AggregationAlgorithm

    return estimate_and_validate_volterra_model(x_est, y_est, x_val, y_val, kernels, algorithm, R)


def estimate_and_validate_volterra_model_on_many_datasets(training_datasets, validation_dataset, kernels,
                                                          algorithm_class, R, **kwargs):
    model = VolterraModel(kernels=kernels)
    algorithm = algorithm_class(model.dictionary, R)

    training_datasets, validation_dataset = preprocess_datasets(training_datasets, validation_dataset)
    x_val = validation_dataset['x']
    x0_val = validation_dataset['x0']
    y_val = validation_dataset['y']

    # train model
    start = time.time()
    model_parameters = algorithm.run_on_many_datasets(datasets=training_datasets, **kwargs)
    end = time.time()
    execution_time = end - start

    # validate model
    model.set_parameters(model_parameters)
    y_mod = model.evaluate_output(x_val, x0=x0_val)
    error = 1 / len(x_val) * np.sum((y_mod - y_val) ** 2)

    return error, execution_time, y_mod

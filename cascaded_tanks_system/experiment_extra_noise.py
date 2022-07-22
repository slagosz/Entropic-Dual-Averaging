import numpy as np

from common.run_experiment import estimate_and_validate_DA, estimate_and_validate_l1_aggregation
import matplotlib.pyplot as plt
from load_data import load_data

# %% load data

x_est, y_est, x_val, y_val = load_data()

# %% setup model parameters

kernels = (100, 100, 20)
R = 25

# %% run experiment

errors_da = {}
errors_aggr = {}

N = 1024
num_of_realizations = 10

# extra_noise_variance = [0, 0.5, 1, 1.5, 2, 2.5, 3]
extra_noise_variance = [0, 0.125, 0.25, 0.5, 1, 1.5, 2, 3]
np.random.seed(34)
# noise_signal = np.random.uniform(-1, 1, N)

for i in range(0, num_of_realizations):
    noise_signal = np.random.randn(N)
    for variance in extra_noise_variance:
        z = noise_signal * np.sqrt(variance)
        y_est_noisy = y_est + z

        err_da, _, _ = estimate_and_validate_DA(x_est, y_est_noisy, x_val, y_val, kernels, R)
        err_aggr, _, _ = estimate_and_validate_l1_aggregation(x_est, y_est_noisy, x_val, y_val, kernels, R)

        errors_da[variance] = errors_da.get(variance, 0) + err_da / num_of_realizations
        err_aggr[variance] = err_aggr.get(variance, 0) + err_aggr / num_of_realizations

import json
with open('extra_noise/extra_noise_experiment.json', 'w') as f:
    json.dump(dict(errors_da=errors_da, errors_aggr=errors_aggr), f)

# %% plot errors

plt.close()
plt.style.use('../common/style.mplstyle')

plt.figure(figsize=(3.7, 2.4))

err_da = sorted(errors_da.items())
sigma, err_da = zip(*err_da)
plt.plot(sigma, err_da, '.-')

err_aggr = sorted(errors_aggr.items())
sigma, err_aggr = zip(*err_aggr)
plt.plot(sigma, err_aggr, '.--')

plt.xlabel('$\sigma_Z^2$')
plt.ylabel('err')
plt.legend(['Entropic DA', '$\ell_{1}$ convex aggregation'])
plt.grid()

plt.savefig('extra_noise/err_extra_noise.pdf')

import numpy as np
from tqdm import tqdm

from common.run_experiment import estimate_and_validate_DA, estimate_and_validate_l1_aggregation
import matplotlib.pyplot as plt
from load_data import load_data

# %% load data

x_est, y_est, x_val, y_val = load_data()

# %% setup model parameters

kernels_da = (60, 80)
R_da = 30
kernels_aggr = (40, 90)
R_aggr = 25

# %% run experiment

errors_da = {}
errors_aggr = {}

N = 1024
num_of_realizations = 5

extra_noise_variance = [0, 0.125, 0.25, 0.5, 1, 1.5, 2, 3]
np.random.seed(34)

for i in tqdm(range(0, num_of_realizations)):
    noise_signal = np.random.randn(N)
    for variance in extra_noise_variance:
        print(variance)
        z = noise_signal * np.sqrt(variance)
        y_est_noisy = y_est + z

        err_da, _, _ = estimate_and_validate_DA(x_est, y_est_noisy, x_val, y_val, kernels_da, R_da)
        err_aggr, _, _ = estimate_and_validate_l1_aggregation(x_est, y_est_noisy, x_val, y_val, kernels_aggr, R_aggr)

        errors_da[variance] = errors_da.get(variance, 0) + err_da / num_of_realizations
        errors_aggr[variance] = errors_aggr.get(variance, 0) + err_aggr / num_of_realizations

import json, time
timestr = time.strftime("%m%d-%H%M")
with open(f'extra_noise_experiment_{timestr}.json', 'w') as f:
    json.dump(dict(errors_da=errors_da, errors_aggr=errors_aggr, kernels_da=kernels_da, R_da=R_da,
                   kernels_aggr=kernels_aggr, R_aggr=R_aggr), f)

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

plt.savefig(f'err_extra_noise_{timestr}.pdf')

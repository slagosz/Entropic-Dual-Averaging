import numpy as np

from common.run_experiment import estimate_and_validate_DA, estimate_and_validate_l1_aggregation
import matplotlib.pyplot as plt
from load_data import load_data

# %% load data

x_est, y_est, x_val, y_val = load_data()

# %% setup model parameters

kernels = (60, 60)
R = 25

# %% run experiment

e_da = {}
e_aggr = {}

N = 500

extra_noise_variance = [0, 0.05, 0.1, 0.25, 0.5, 1]
# np.random.seed(34)
noise_signal = np.random.randn(N)
# noise_signal = np.random.uniform(-1, 1, N)

for variance in extra_noise_variance:
    z = noise_signal * np.sqrt(variance)
    y_est_noisy = y_est + z

    e_da[variance], _, _ = estimate_and_validate_DA(x_est, y_est_noisy, x_val, y_val, kernels, R)
    e_aggr[variance], _, _ = estimate_and_validate_l1_aggregation(x_est, y_est_noisy, x_val, y_val, kernels, R)

# import json
# with open('extra_noise/variance_experiment.json', 'w') as f:
#     json.dump(dict(e_da=e_da), f)

# %% plot errors

plt.close()
plt.style.use('../common/style.mplstyle')

plt.figure(figsize=(3.7, 2.4))

err_da = sorted(e_da.items())
sigma, err_da = zip(*err_da)
plt.plot(sigma, err_da, '.-')

err_aggr = sorted(e_aggr.items())
sigma, err_aggr = zip(*err_aggr)
plt.plot(sigma, err_aggr, '.--')

plt.xlabel('$\sigma_Z^2$')
plt.ylabel('err')
plt.legend(['Entropic DA', '$\ell_{1}$ convex aggregation'])
plt.grid()

plt.savefig('err_extra_noise.pdf')

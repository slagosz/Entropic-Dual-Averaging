import numpy as np
from common.run_experiment import estimate_and_validate_DA
import matplotlib.pyplot as plt
from load_data import load_data

# %% load data

x_est, y_est, x_val, y_val = load_data()

# %% setup model parameters

kernels = (100, 100, 20)
R = 25

# %% setup experiment parameters

N = 1024
stepsize_scaling_tab = np.array([0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.2,
                                 1.4])

# %% DA constants calculation

U = np.max(np.abs(y_est))
sigma_z_sq = 0.1 * U
G_sq = R ** 2 * ((R + U) ** 2 + sigma_z_sq)

# %% run experiment

err_nonadaptive = {}

for stepsize_scaling in stepsize_scaling_tab:
    print("Scaling G_sq by : {0}".format(stepsize_scaling))
    
    G_sq_scaled = stepsize_scaling ** 2 * G_sq

    err_nonadaptive[stepsize_scaling], _, _ = estimate_and_validate_DA(x_est, y_est, x_val, y_val, kernels, R,
                                                                       G_sq=G_sq_scaled, adaptive_stepsize=False)

# %% plot models' errors

eda_adaptive_error = 1.422450866750514  # computed in the other experiment

err_nonadaptive_lists = sorted(err_nonadaptive.items())

x, y = zip(*err_nonadaptive_lists)

plt.close()
plt.style.use('../common/style.mplstyle')

fig, ax = plt.subplots(figsize=(3.8, 2.4))

ax.plot(x, y, '.-', color='tab:blue')
ax.axhline(y=eda_adaptive_error, color='tab:orange', linestyle='--')

axins = ax.inset_axes([0.54, 0.18, 0.45, 0.3])
axins.semilogx(x, y, '.-', color='tab:blue')

# sub region of the original image
axins.set_xlim(0.009, 0.18)
axins.set_ylim(1.15, 1.65)
axins.axhline(y=eda_adaptive_error, color='tab:orange', linestyle='--')
axins.grid()

ax.indicate_inset_zoom(axins, edgecolor='tab:red')
plt.legend(['nonadaptive Entropic DA', 'adaptive Entropic DA'])

plt.xlabel('$\\alpha$')
plt.ylabel('err')

plt.grid()
plt.savefig('err_scaling.pdf')

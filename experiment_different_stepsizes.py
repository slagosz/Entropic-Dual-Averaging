from scipy.io import loadmat
import numpy as np
from dual_averaging_algorithm import EntropicDualAveragingAlgorithm
from volterra_model import VolterraModel
import matplotlib.pyplot as plt

# %% load data

data = loadmat('dataBenchmark.mat')

u_est = np.squeeze(data['uEst'])
y_est = np.squeeze(data['yEst'])
u_val = np.squeeze(data['uVal'])
y_val = np.squeeze(data['yVal'])

# %% scale data (we want input to be in [-1, 1] interval)

scale_parameter = np.abs(np.max(u_est))
u_est /= scale_parameter
u_val /= scale_parameter

# %% setup model parameters

kernels = (100, 100, 20)
R = 25

model_memory_len = np.max(kernels)

# %% setup experiment parameters

N = 1024
stepsize_scaling_tab = np.array([0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.2,
                                 1.4])

# %% DA constants calculation

U = np.max(np.abs(y_est))
sigma_z_sq = 0.1 * U
G_sq = R ** 2 * ((R + U) ** 2 + sigma_z_sq)

# %% run experiment

err_adaptive = {}
err_nonadaptive = {}

for stepsize_scaling in stepsize_scaling_tab:
    print("Scaling G_sq by : {0}".format(stepsize_scaling))
    
    G_sq_scaled = stepsize_scaling ** 2 * G_sq
    
    x = u_est[model_memory_len-1:N-1]
    x0 = u_est[0:model_memory_len-1]
    y = y_est[model_memory_len-1:N-1]
    
    # DA adaptive
    m_da_adaptive = VolterraModel(kernels=kernels)
    alg = EntropicDualAveragingAlgorithm(m_da_adaptive.dictionary, R=R)
    da_parameters = alg.run(x, y, G_sq_scaled, x0=x0, adaptive_stepsize=True)
    
    # validate
    m_da_adaptive.set_parameters(da_parameters)
    y_mod_da_adaptive = m_da_adaptive.evaluate_output(u_val)
    err_adaptive[stepsize_scaling] = 1 / len(u_val) * np.sum((y_mod_da_adaptive - y_val) ** 2)

    # DA nonadaptive
    m_da_nonadaptive = VolterraModel(kernels=kernels)
    alg = EntropicDualAveragingAlgorithm(m_da_nonadaptive.dictionary, R=R)
    da_parameters = alg.run(x, y, G_sq_scaled, x0=x0, adaptive_stepsize=False)

    # validate
    m_da_nonadaptive.set_parameters(da_parameters)
    y_mod_da_nonadaptive = m_da_nonadaptive.evaluate_output(u_val)
    err_nonadaptive[stepsize_scaling] = 1 / len(u_val) * np.sum((y_mod_da_nonadaptive - y_val) ** 2)


# %% plot models' errors

aggregation_error = 1.101105958546409  # computed in the other experiment

err_adaptive_list = sorted(err_adaptive.items())
err_nonadaptive_lists = sorted(err_nonadaptive.items())

x, y = zip(*err_adaptive_list)
x2, y2 = zip(*err_nonadaptive_lists)

plt.clf()
plt.rcdefaults()
plt.rcParams['text.usetex'] = True
plt.rc('font', family='serif')
plt.rc('font', size=10)
plt.rc('axes', labelsize='small')
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')
plt.rc('legend', fontsize='small')

plt_scale = 0.65
fig, ax = plt.subplots(figsize=(plt_scale * 6.4, plt_scale * 4.8))

ax.plot(x2, y2, '.--', color='tab:orange')
ax.plot(x, y, '.-', color='tab:blue')

axins = ax.inset_axes([0.54, 0.18, 0.45, 0.3])
axins.semilogx(x2, y2, '.--', color='tab:orange')
axins.semilogx(x, y, '.-', color='tab:blue')

# sub region of the original image
axins.set_xlim(0.009, 0.18)
axins.set_ylim(1.15, 1.75)
axins.axhline(y=aggregation_error, color='tab:red', linestyle='--')
axins.grid()

ax.indicate_inset_zoom(axins, edgecolor='tab:red')
plt.legend(['nonadaptive Entropic DA', 'adaptive Entropic DA'])

plt.xlabel('$\\alpha$')
plt.ylabel('err')

plt.axhline(y=aggregation_error, color='tab:red', linestyle='--')
plt.grid()
plt.savefig('err_scaling.pdf', dpi=1200, transparent=False, bbox_inches='tight')
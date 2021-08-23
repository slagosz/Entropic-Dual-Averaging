from scipy.io import loadmat
import numpy as np
from dual_averaging_algorithm import EntropicDualAveragingAlgorithm
from aggregation_algorithm import aggregation_for_volterra
from volterra_model import VolterraModel
import time
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

N_tab = [256, 384, 512, 640, 768, 896, 1024]

# %% DA constants calculation

G_sq = (R * R * 2.1) ** 2

# %% run experiment

e_da = {}
e_aggr = {}

time_da = {}
time_aggr = {}

y_mod_da = 0
y_mod_aggr = 0

for N in N_tab:
    print("Number of measurements: {0}".format(N))

    x = u_est[model_memory_len - 1: N - 1]
    x0 = u_est[0: model_memory_len - 1]
    y = y_est[model_memory_len - 1: N - 1]

    # entropic dual averaging
    m_da = VolterraModel(kernels=kernels)
    alg = EntropicDualAveragingAlgorithm(m_da.dictionary, R=R)
    start = time.time()
    da_parameters = alg.run(x, y, G_sq, x0=x0)
    end = time.time()
    time_da[N] = end - start

    # validate model
    m_da.set_parameters(da_parameters)
    y_mod_da = m_da.evaluate_output(u_val)
    e_da[N] = 1 / len(u_val) * np.sum((y_mod_da - y_val) ** 2)

    # l1 convex aggregation
    m_aggr = VolterraModel(kernels=kernels)
    start = time.time()
    aggr_parameters = aggregation_for_volterra(m_aggr.dictionary, x, y, x0=x0, R=R)
    end = time.time()
    time_aggr[N] = end - start

    # validate model
    m_aggr.set_parameters(aggr_parameters)
    y_mod_aggr = m_aggr.evaluate_output(u_val)
    e_aggr[N] = 1 / len(u_val) * np.sum((y_mod_aggr - y_val) ** 2)


# %% plot models' outputs

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
plt.figure(figsize=(plt_scale * 6.4, plt_scale * 4.8))

plt.plot(y_mod_da)
plt.plot(y_mod_aggr, '--')
plt.plot(y_val, '-.')
plt.xlabel('t')
plt.ylabel('output')
plt.legend(['Entropic DA', '$\ell_{1}$ convex aggregation', 'True system'])
plt.grid()

plt.savefig('output.pdf', dpi=1200, transparent=False, bbox_inches='tight')


# %% plot algorithms' errors

plt.clf()
plt.rcdefaults()
plt.rcParams['text.usetex'] = True
plt.rc('font', family='serif')
plt.rc('font', size=10)
plt.rc('axes', labelsize='small')
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')
plt.rc('legend', fontsize='small')

plt_scale = 0.45
plt.figure(figsize=(plt_scale * 6.4, plt_scale * 4.8))

plt.plot(N_tab, e_da.values(), '.-')
#plt.plot(N_tab, e_aggr.values(), '--.')
plt.xlabel('N')
plt.ylabel('err')
plt.legend(['Entropic DA', '$\ell_{1}$ convex aggregation'])
plt.grid()

plt.savefig('err.pdf', dpi=1200, transparent=False, bbox_inches='tight')


# %% plot algorithms' times of execution

t_da = sorted(time_da.items())
t_aggr = sorted(time_aggr.items())

x, y = zip(*t_da)
x2, y2 = zip(*t_aggr)

plt.clf()
plt.rcdefaults()
plt.rcParams['text.usetex'] = True
plt.rc('font', family='serif')
plt.rc('font', size=10)
plt.rc('axes', labelsize='small')
plt.rc('xtick', labelsize='small')
plt.rc('ytick', labelsize='small')
plt.rc('legend', fontsize='small')

plt_scale = 0.45
plt.figure(figsize=(plt_scale * 6.4, plt_scale * 4.8))

plt.plot(x, y, '.-')
plt.plot(x2, y2, '.--')
plt.xlabel('N')
plt.ylabel('time of estimation [s]')
plt.legend(['Entropic DA', '$\ell_{1}$ convex aggregation'])
plt.grid()

plt.savefig('time.pdf', dpi=1200, transparent=False, bbox_inches='tight')

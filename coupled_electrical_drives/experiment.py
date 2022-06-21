import matplotlib.pyplot as plt

from common.run_experiment import estimate_and_validate_DA, estimate_and_validate_l1_aggregation
from load_data import load_data

# %% load data

x_est, y_est, x_val, y_val = load_data()

# %% setup model parameters

kernels = (5, 25)
R = 9.5

# %% setup experiment parameters

N_tab = [100, 200, 300, 400, 500]

# %% run experiment

e_da = {}
e_aggr = {}

time_da = {}
time_aggr = {}

for N in N_tab:
    print("Number of measurements: {0}".format(N))

    x_est_sliced = x_est[:N-1]
    y_est_sliced = y_est[:N-1]

    e_da[N], time_da[N], y_mod_da = estimate_and_validate_DA(x_est_sliced, y_est_sliced, x_val, y_val, kernels, R)
    e_aggr[N], time_aggr[N], y_mod_aggr = estimate_and_validate_l1_aggregation(x_est_sliced, y_est_sliced, x_val, y_val,
                                                                               kernels, R)

# %% plot models' outputs

plt.close()
plt.style.use('../common/style.mplstyle')

plt.figure(figsize=(3.8, 2.4))
plt.plot(y_mod_da)
plt.plot(y_mod_aggr, '--')
plt.plot(y_val, '-.')
plt.xlabel('t')
plt.ylabel('output')
plt.legend(['Entropic DA', '$\ell_{1}$ convex aggregation', 'True system'])
plt.grid()

plt.savefig('output.pdf')


# %% plot algorithms' errors

plt.close()
plt.style.use('../common/style.mplstyle')

plt.figure(figsize=(3.7, 2.4))
plt.plot(N_tab, e_da.values(), '.-')
plt.plot(N_tab, e_aggr.values(), '--.')
plt.xlabel('N')
plt.ylabel('err')
plt.legend(['Entropic DA', '$\ell_{1}$ convex aggregation'])
plt.grid()

plt.savefig(f'err_{str(kernels)}_{R}.pdf')


# %% plot algorithms' times of execution

t_da = sorted(time_da.items())
t_aggr = sorted(time_aggr.items())

x, y = zip(*t_da)
x2, y2 = zip(*t_aggr)

plt.close()
plt.style.use('../common/style.mplstyle')

plt.figure(figsize=(3.7, 2.4))
plt.plot(x, y, '.-')
plt.plot(x2, y2, '.--')
plt.xlabel('N')
plt.ylabel('time of estimation [s]')
plt.legend(['Entropic DA', '$\ell_{1}$ convex aggregation'])
plt.grid()

plt.savefig('time.pdf')

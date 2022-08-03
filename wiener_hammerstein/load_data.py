from scipy.io import loadmat
from random import randrange, seed
import numpy as np


def load_data():
    N_est = 1000
    N_val = 5000

    data = loadmat('data/WienerHammerBenchMark.mat')

    u, y = np.squeeze(data['uBenchMark']), np.squeeze(data['yBenchMark'])
    seed(1500100900)
    offset = randrange(1000, 100000 - N_est)
    u_est, y_est = u[offset:offset + N_est], y[offset:offset + N_est]

    offset = randrange(100000, 188000 - N_val)
    u_val, y_val = u[offset:offset + N_val], y[offset:offset + N_val]

    return u_est, y_est, u_val, y_val

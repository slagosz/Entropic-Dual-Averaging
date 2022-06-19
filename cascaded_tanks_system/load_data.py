from scipy.io import loadmat
import numpy as np


def load_data():
    data = loadmat('data/dataBenchmark.mat')

    u_est = np.squeeze(data['uEst'])
    y_est = np.squeeze(data['yEst'])
    u_val = np.squeeze(data['uVal'])
    y_val = np.squeeze(data['yVal'])

    return u_est, y_est, u_val, y_val

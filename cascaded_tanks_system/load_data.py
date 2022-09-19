import os
from scipy.io import loadmat
import numpy as np


def load_data():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'data/dataBenchmark.mat')
    data = loadmat(filename)

    u_est = np.squeeze(data['uEst'])
    y_est = np.squeeze(data['yEst'])
    u_val = np.squeeze(data['uVal'])
    y_val = np.squeeze(data['yVal'])

    return u_est, y_est, u_val, y_val

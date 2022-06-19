from scipy.io import loadmat
import numpy as np


def load_data():
    data = loadmat('data/DATAUNIF.MAT')

    u_est = np.squeeze(data['u11'])
    y_est = np.squeeze(data['z11'])
    u_val = np.squeeze(data['u12'])
    y_val = np.squeeze(data['z12'])

    return u_est, y_est, u_val, y_val

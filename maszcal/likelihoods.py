import numpy as np


def log_gaussian_shape(model, data, covariance):
    diff = model - data
    fisher = np.linalg.inv(covariance)
    return -(diff@fisher@diff.T) / 2

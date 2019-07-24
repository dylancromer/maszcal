import numpy as np


def cartesian_prod(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([la] + [len(a) for a in arrays], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[i, ...] = a
    return arr.reshape(la, -1).T


def combine_radii_with_params(rs, params):
    n_rs = rs.size
    n_params = params.shape[0]

    rs_to_concat = np.tile(rs, n_params)[None, :]
    rs_to_concat = rs_to_concat.reshape(n_rs, n_params, order='F').reshape(n_rs*n_params, 1)

    params_to_concat = np.tile(params.T, n_rs).T

    return np.concatenate((rs_to_concat, params_to_concat), axis=1)


def make_flat(array):
    try:
        return array.flatten()
    except AttributeError:
        return array.values.flatten()

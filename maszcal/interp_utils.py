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
    params_to_concat = np.tile(params, n_rs)

    return np.concatenate((rs_to_concat, params_to_concat), axis=0).T


def make_flat(array):
    try:
        return array.flatten()
    except AttributeError:
        return array.values.flatten()

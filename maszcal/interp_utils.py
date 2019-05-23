import numpy as np




def cartesian_prod(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([la] + [len(a) for a in arrays], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[i, ...] = a
    return arr.reshape(la, -1).T


def make_flat(array):
    try:
        return array.flatten()
    except AttributeError:
        return array.values.flatten()

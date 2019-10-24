import numpy as np


def atleast_kd(array, k, append_dims=True):
    array = np.asarray(array)

    if append_dims:
        new_shape = array.shape + (1,) * (k-array.ndim)
    else:
        new_shape = (1,) * (k-array.ndim) + array.shape

    return array.reshape(new_shape)


def trapz_(arr, axis, dx=None):
    arr = np.moveaxis(arr, axis, 0)

    if dx is None:
        dx_ = np.ones(arr.shape[0])
    dx = atleast_kd(dx, arr.ndim)

    arr = dx*arr

    return 0.5*(arr[0, ...] + 2*arr[1:-1, ...].sum(axis=0) + arr[-1, ...])

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
        dx = np.ones(arr.shape[0])
    dx = atleast_kd(dx, arr.ndim)

    arr = dx*arr

    return 0.5*(arr[0, ...] + 2*arr[1:-1, ...].sum(axis=0) + arr[-1, ...])


def expand_parameter_dims(*args):
    expanded_args = []
    num_params = len(args)
    for i, param in enumerate(args):
        expanded_args.append(
            atleast_kd(
                atleast_kd(param, i+1, append_dims=False),
                num_params,
                append_dims=True,
            )
        )
    return expanded_args

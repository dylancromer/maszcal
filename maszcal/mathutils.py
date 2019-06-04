import numpy as np




def atleast_kd(array, k):
    array = np.asarray(array)
    new_shape = array.shape + (1,) * (k - array.ndim)
    return array.reshape(new_shape)

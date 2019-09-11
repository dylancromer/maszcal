import os
import json
import pytest
import numpy as np
import maszcal.ioutils as ioutils


def describe_get_selection_func_interpolator():

    def it_can_load_a_selection_function():
        mus = np.linspace(1, 3, 10)
        zs = np.linspace(0, 2, 5)

        selection_function = lambda m,z: m[:, None]*z[None, :]
        sel_funcs = selection_function(mus, zs)
        sel_func_dict = {'zs':zs,
                         'mus':mus,
                         'selection_fs':sel_funcs}

        SAVED_SELFUNC = 'data/test/test_sel_func.json'
        with open(SAVED_SELFUNC, 'w') as outfile:
            json.dump(sel_func_dict, outfile, cls=ioutils.NumpyEncoder, ensure_ascii=False)

        sel_func_interp = ioutils.get_selection_func_interpolator(SAVED_SELFUNC)

        assert np.allclose(sel_func_interp(mus, zs), sel_funcs)

        os.remove(SAVED_SELFUNC)

def describe_get_lensing_weights_interpolator():

    def it_can_load_lensing_weights():
        mus = np.linspace(1, 3, 10)
        zs = np.linspace(0.1, 2, 5)

        weights = 1/zs**2

        weight_dict = {'zs':zs,
                       'weights':weights}

        SAVED_WEIGHTS = 'data/test/test_lensing_weights.json'
        with open(SAVED_WEIGHTS, 'w') as outfile:
            json.dump(weight_dict, outfile, cls=ioutils.NumpyEncoder, ensure_ascii=False)

        lensing_weights_interp = ioutils.get_lensing_weights_interpolator(SAVED_WEIGHTS)

        os.remove(SAVED_WEIGHTS)

        assert np.allclose(lensing_weights_interp(zs), weights)

import operator
import functools
import pytest
import numpy as np
import maszcal.mathutils


def describe_expand_parameter_dims():

    def it_expands_each_of_its_args_so_they_are_all_padded_with_unit_dimensions():
        param_1 = np.arange(2)
        param_2 = np.arange(3)
        with pytest.raises(ValueError):
            param_1*param_2

        new_param_1, new_param_2 = maszcal.mathutils.expand_parameter_dims(param_1, param_2)

        assert new_param_1.shape == (2, 1)
        assert new_param_2.shape == (1, 3)
        new_param_1*new_param_2

    def it_works_for_lots_of_params():
        params = [np.arange(i) for i in range(2, 5)]
        with pytest.raises(ValueError):
            functools.reduce(
                operator.mul,
                params,
            )

        num_params = len(params)
        new_params = maszcal.mathutils.expand_parameter_dims(*params)

        for i, new_param in enumerate(new_params):
            assert new_param.shape == i*(1,) + (params[i].size,) + (num_params-1-i)*(1,)

        functools.reduce(
            operator.mul,
            new_params,
        )

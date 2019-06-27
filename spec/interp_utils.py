import numpy as np
from maszcal.interp_utils import cartesian_prod, make_flat, combine_radii_with_params




def describe_cartesian_prod():

    def it_should_give_the_same_order_as_flattening():
        """
        The order of make_flat's output should always align with the ordering given
        by cartesian product, when the product is interpreted as coordinates for the
        values of the flattened array
        """
        xs = np.linspace(1, 10, 10)
        ys = np.linspace(1, 10, 10)
        zs = np.linspace(1, 10, 10)

        fs = np.random.rand(10, 10, 10)

        flat_func = make_flat(fs)

        coord_inds = []
        for x in (xs, ys, zs):
            coord_inds.append(np.linspace(0, x.size - 1, x.size, dtype=int))

        coord_inds = tuple(cartesian_prod(*coord_inds).T)
        what_flat_func_should_be = fs[coord_inds]

        assert np.all(flat_func == what_flat_func_should_be)


def describe_combine_radii_with_params():

    def it_should_give_the_same_order_as_flattening():
        rs = np.linspace(1, 10, 10)
        a = np.arange(1, 11)
        b = np.arange(2, 12)
        params = np.stack((a,b)).T

        fs = rs[:, None] * params[None, :, 0]

        flat_func = make_flat(fs)

        coords = combine_radii_with_params(rs, params)

        what_flat_func_should_be = coords.T[0] * coords.T[1]

        assert np.all(flat_func == what_flat_func_should_be)

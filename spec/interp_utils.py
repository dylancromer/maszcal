import numpy as np
from maszcal.interp_utils import cartesian_prod, make_flat




"""
    - The order of make_flat's output should always align with the ordering given
    by cartesian product, when the product is interpreted as coordinates for the
    values of the flattened array
"""
def test_orders_equal():
    xs = np.linspace(1, 10, 10)
    ys = np.linspace(1, 10, 10)
    zs = np.linspace(1, 10, 10)

    fs = np.random.rand(10, 10, 10)

    coords = cartesian_prod(xs, ys, zs)
    flat_func = make_flat(fs)

    coord_inds = []
    for x in (xs, ys, zs):
        coord_inds.append(np.linspace(0, x.size - 1, x.size, dtype=int))

    coord_inds = tuple(cartesian_prod(*coord_inds).T)
    flat_func_should_be = fs[coord_inds]

    assert np.all(flat_func == flat_func_should_be)

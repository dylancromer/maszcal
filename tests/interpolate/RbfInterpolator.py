"""
RbfInterpolator

Interpolates maszcal model grids
"""
import numpy as np
from maszcal.interpolate import RbfInterpolator




length = 5
xs = np.linspace(0, 3, length)
ys = np.linspace(0, 3, length)
zs = np.linspace(0, 3, length)
coords = (xs, ys, zs)
grid = np.ones((length, length, length))

interpolator = RbfInterpolator(coords, grid)


"""
    - it interpolates a constant to be a constant
"""
def test_interp_over_constant_input():
    interpolator.process()

    test_xs = np.linspace(0.1, 3, 10)
    test_ys = test_xs
    test_zs = test_xs

    test_coords = (test_xs, test_ys, test_zs)

    interpolated_grid = interpolator.interp(test_coords)

    assert np.allclose(interpolated_grid, np.ones((10, 10, 10)), rtol=1e-2)

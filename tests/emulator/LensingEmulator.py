import numpy as np
from maszcal.emulator import LensingEmulator




def test_lensing_emulator():
    xs = np.linspace(0, 1, 10)
    ys = np.linspace(0, 1, 10)
    coords = (xs, ys)
    grid = np.ones((10, 10))

    emulator = LensingEmulator()
    emulator.emulate(coords, grid)

    xs_subgrid = np.linspace(0, 0.5, 10)
    ys_suggrid = np.linspace(0.5, 1, 10)
    coords_subgrid = (xs_subgrid, ys_suggrid)

    values_subgrid,err = emulator.evaluate_on(coords_subgrid)

    assert values_subgrid.shape == (10, 10)
    assert np.all(err > 0)

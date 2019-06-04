import numpy as np
from maszcal.emulator import LensingEmulator




def test_emulator_generate_grid():
    emulator = LensingEmulator()

    rs = np.logspace(-1, 1, 10)
    cons = np.linspace(2, 5, 5)
    a_szs = np.linspace(-1, 1, 5)
    coords = (rs, cons, a_szs)

    grid = emulator.generate_grid(coords)
    assert grid.shape == (10, 5, 5)


def test_lensing_emulator():
    xs = np.linspace(0, 1, 10)
    ys = np.linspace(0, 1, 10)
    coords = (xs, ys)
    grid = np.ones((10, 10))

    emulator = LensingEmulator()
    emulator.emulate(coords, grid=grid, check_errs=False)

    xs_subgrid = np.linspace(0, 0.5, 10)
    ys_suggrid = np.linspace(0.5, 1, 10)
    coords_subgrid = (xs_subgrid, ys_suggrid)

    values_subgrid = emulator.evaluate_on(coords_subgrid)

    assert values_subgrid.shape == (10, 10)

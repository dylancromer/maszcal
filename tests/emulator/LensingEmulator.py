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
    ys_subgrid = np.linspace(0.5, 1, 10)
    coords_subgrid = (xs_subgrid, ys_subgrid)

    values_subgrid = emulator.evaluate_on(coords_subgrid)

    assert values_subgrid.shape == (10, 10)


def test_lensing_emulator_higher_dim():
    xs = np.linspace(0, 1, 4)
    ys = xs
    zs = xs
    ws = xs
    us = np.linspace(0, 1, 5)

    coords = (xs, ys, zs, ws, us)
    grid = np.ones((4, 4, 4, 4, 5))

    emulator = LensingEmulator()
    emulator.emulate(coords, grid)

    xs_subgrid = np.linspace(0, 0.5, 3)
    ys_subgrid = np.linspace(0.5, 1, 4)
    zs_subgrid = ys_subgrid
    ws_subgrid = xs_subgrid
    us_subgrid = xs_subgrid

    coords_subgrid = (xs_subgrid, ys_subgrid, zs_subgrid, ws_subgrid, us_subgrid)

    values_subgrid = emulator.evaluate_on(coords_subgrid)

    assert values_subgrid.shape == (3, 4, 4, 3, 3)

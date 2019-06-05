import numpy as np
from maszcal.likelihood import GaussianLikelihood




def test_emulator_setup():
    grid = np.load('data/tests/testgrid.npy')
    likelihood = GaussianLikelihood(grid=grid)

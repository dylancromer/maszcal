import numpy as np
import xarray as xa
from maszcal.interpolate import GaussInterpolator
from maszcal.model import StackedModel




class NoGrid:
    pass


class LensingEmulator:
    def __init__(self):
        self.ERRCHECK_NUM = 4

    def generate_grid(self, coords):
        pass

    def emulate(self, coords, grid=NoGrid()):
        if isinstance(grid, NoGrid):
            grid = self.generate_grid(coords)

        self.interpolator = GaussInterpolator(coords, grid)
        self.interpolator.process()

    def check_errors(self, coords):
        if isinstance(coords, xa.DataArray):
            coords = (c.values for c in coords)

        rand_coords = []
        for coord in coords:
            coord_length = coord.max() - coord.min()
            min_ = coord.min()
            rand_coord = coord_length*np.random.rand(self.ERRCHECK_NUM) + min_
            rand_coords.append(rand_coord)

    def evaluate_on(self, coords):
        return self.interpolator.interp(coords)

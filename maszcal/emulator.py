from maszcal.interpolate import GaussInterpolator




class LensingEmulator:
    def emulate(self, coords, grid):
        self.interpolator = GaussInterpolator(coords, grid)
        self.interpolator.process()

    def evaluate_on(self, coords):
        return self.interpolator.interp(coords)

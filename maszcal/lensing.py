import astropy.units as u
from maszcal.model import StackedModel


class LensingSignal:
    def __init__(self, mu_bins, redshift_bins, comoving=False):
        self.stacked_model = StackedModel(mu_bins, redshift_bins)
        self.stacked_model.comoving_radii = comoving

    def stacked_esd(self, rs, params, miscentered=False, units=u.Msun/u.Mpc**2):
        self.stacked_model.params = params

        return self.stacked_model.delta_sigma(rs, miscentered=miscentered, units=units)
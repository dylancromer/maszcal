import astropy.units as u
from maszcal.model import StackedModel
import maszcal.defaults as defaults


class LensingSignal:
    def __init__(
            self,
            mu_bins,
            redshift_bins,
            comoving=True,
            selection_func_file=defaults.DefaultSelectionFunc(),
            lensing_weights_file=defaults.DefaultLensingWeights(),
    ):
        self.stacked_model = StackedModel(
            mu_bins,
            redshift_bins,
            selection_func_file=selection_func_file,
            lensing_weights_file=lensing_weights_file
        )

        self.stacked_model.comoving_radii = comoving

    def stacked_esd(self, rs, params, miscentered=False, units=u.Msun/u.Mpc**2):
        self.stacked_model.params = params

        return self.stacked_model.delta_sigma(rs, miscentered=miscentered, units=units)

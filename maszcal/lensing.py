import astropy.units as u
import maszcal.nothing as nothing
from maszcal.model import StackedModel, SingleMassModel
import maszcal.defaults as defaults


class LensingSignal:
    def __init__(
            self,
            log_masses=nothing.NoMasses(),
            redshifts=nothing.NoRedshifts(),
            comoving=True,
            selection_func_file=defaults.DefaultSelectionFunc(),
            lensing_weights_file=defaults.DefaultLensingWeights(),
    ):

        self.comoving = comoving
        self.log_masses = log_masses

        if not isinstance(redshifts, nothing.NoRedshifts):
            self.redshifts = redshifts
        else:
            raise TypeError('redshifts are required to calculate a lensing signal')

        self.selection_func_file = selection_func_file
        self.lensing_weights_file = lensing_weights_file

    def _init_stacked_model(self):
        if isinstance(self.log_masses, nothing.NoMasses):
            raise TypeError('log_masses must be provided to calculate a stacked model')

        self.stacked_model = StackedModel(
            self.log_masses,
            self.redshifts,
            selection_func_file=self.selection_func_file,
            lensing_weights_file=self.lensing_weights_file
        )

        self.stacked_model.comoving_radii = self.comoving

    def stacked_esd(self, rs, params, miscentered=False, units=u.Msun/u.pc**2):
        try:
            self.stacked_model.params = params
            return self.stacked_model.delta_sigma(rs, miscentered=miscentered, units=units)
        except AttributeError:
            self._init_stacked_model()
            self.stacked_model.params = params
            return self.stacked_model.delta_sigma(rs, miscentered=miscentered, units=units)

    def _check_redshift_for_single_mass_model(self):
        if self.redshifts.size != 1:
            raise TypeError('redshifts must have length 1 to calculate a single-mass-bin model')

    def _init_single_mass_model(self):
        self._check_redshift_for_single_mass_model()

        self.single_mass_model = SingleMassModel(
            self.redshifts,
            comoving_radii=self.comoving,
        )

    def single_mass_esd(self, rs, params, units=u.Msun/u.pc**2):
        log_masses = params[:, 0]
        concentrations = params[:, 1]
        try:
            return self.single_mass_model.delta_sigma(rs, log_masses, concentrations, units=units)
        except AttributeError:
            self._init_single_mass_model()
            return self.single_mass_model.delta_sigma(rs, log_masses, concentrations, units=units)

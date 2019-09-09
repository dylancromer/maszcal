import astropy.units as u
import maszcal.nothing as nothing
from maszcal.model import StackedModel, SingleMassModel
import maszcal.defaults as defaults


class LensingSignal:
    def __init__(
            self,
            log_masses=nothing.NoMasses(),
            redshifts=nothing.NoRedshifts(),
            units=u.Msun/u.pc**2,
            comoving=True,
            delta=200,
            mass_definition='mean',
            cosmo_params=defaults.DefaultCosmology(),
            selection_func_file=defaults.DefaultSelectionFunc(),
            lensing_weights_file=defaults.DefaultLensingWeights(),
    ):
        self.log_masses = log_masses
        if not isinstance(redshifts, nothing.NoRedshifts):
            self.redshifts = redshifts
        else:
            raise TypeError('redshifts are required to calculate a lensing signal')

        self.units = units
        self.comoving = comoving
        self.delta = delta
        self.mass_definition = mass_definition

        self.cosmo_params = cosmo_params
        self.selection_func_file = selection_func_file
        self.lensing_weights_file = lensing_weights_file

    def _init_stacked_model(self):
        if isinstance(self.log_masses, nothing.NoMasses):
            raise TypeError('log_masses must be provided to calculate a stacked model')

        self.stacked_model = StackedModel(
            self.log_masses,
            self.redshifts,
            cosmo_params=self.cosmo_params,
            units=self.units,
            selection_func_file=self.selection_func_file,
            lensing_weights_file=self.lensing_weights_file,
            delta=self.delta,
            mass_definition=self.mass_definition,
        )

        self.stacked_model.comoving_radii = self.comoving

    def stacked_esd(self, rs, params):
        cons = params[:, 0]
        a_szs = params[:, 1]
        try:
            return self.stacked_model.delta_sigma(rs, cons, a_szs)
        except AttributeError:
            self._init_stacked_model()
            return self.stacked_model.delta_sigma(rs, cons, a_szs)

    def _check_redshift_for_single_mass_model(self):
        if self.redshifts.size != 1:
            raise TypeError('redshifts must have length 1 to calculate a single-mass-bin model')

    def _init_single_mass_model(self):
        self._check_redshift_for_single_mass_model()

        self.single_mass_model = SingleMassModel(
            self.redshifts,
            cosmo_params=self.cosmo_params,
            comoving_radii=self.comoving,
            delta=self.delta,
            mass_definition=self.mass_definition,
        )

    def single_mass_esd(self, rs, params):
        log_masses = params[:, 0]
        concentrations = params[:, 1]
        try:
            return self.single_mass_model.delta_sigma(rs, log_masses, concentrations)
        except AttributeError:
            self._init_single_mass_model()
            return self.single_mass_model.delta_sigma(rs, log_masses, concentrations)

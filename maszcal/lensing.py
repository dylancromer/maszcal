import numpy as np
import astropy.units as u
import maszcal.nothing as nothing
import maszcal.model as model
import maszcal.defaults as defaults


class SingleMassLensingSignal:
    def __init__(
            self,
            redshift=nothing.NoRedshifts(),
            units=u.Msun/u.pc**2,
            comoving=True,
            delta=200,
            mass_definition='mean',
            cosmo_params=defaults.DefaultCosmology(),
    ):
        if not isinstance(redshift, nothing.NoRedshifts):
            self.redshift = redshift
        else:
            raise TypeError('redshift is required to calculate a lensing signal')

        self._check_redshift_for_single_mass_model()

        self.units = units
        self.comoving = comoving
        self.delta = delta
        self.mass_definition = mass_definition

        self.cosmo_params = cosmo_params

    def _check_redshift_for_single_mass_model(self):
        if np.asarray(self.redshift).size != 1:
            raise TypeError('redshifts must have length 1 to calculate a single-mass-bin model')

    def _init_single_mass_model(self):
        self.single_mass_model = model.SingleMassModel(
            self.redshift,
            cosmo_params=self.cosmo_params,
            comoving_radii=self.comoving,
            delta=self.delta,
            mass_definition=self.mass_definition,
        )

    def esd(self, rs, params):
        log_masses = params[:, 0]
        concentrations = params[:, 1]
        try:
            return self.single_mass_model.delta_sigma(rs, log_masses, concentrations)
        except AttributeError:
            self._init_single_mass_model()
            return self.single_mass_model.delta_sigma(rs, log_masses, concentrations)


class StackedGbLensingSignal:
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
        if not isinstance(log_masses, nothing.NoMasses):
            self.log_masses = log_masses
        else:
            raise TypeError('log_masses must be provided to calculate a stacked model')

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

    def _init_gaussian_baryon_model(self):
        self.gaussian_baryon_model = model.GaussianBaryonModel(
            mu_bins=self.log_masses,
            redshift_bins=self.redshifts,
            selection_func_file=self.selection_func_file,
            lensing_weights_file=self.lensing_weights_file,
            cosmo_params=self.cosmo_params,
            units=self.units,
            comoving_radii=self.comoving,
            delta=self.delta,
            mass_definition=self.mass_definition,
        )

    def esd(self, rs, params):
        cons = params[:, 0]
        a_szs = params[:, 1]
        ln_bary_vars = params[:, 2]
        try:
            return self.gaussian_baryon_model.stacked_delta_sigma(rs, cons, a_szs, ln_bary_vars)
        except AttributeError:
            self._init_gaussian_baryon_model()
            return self.gaussian_baryon_model.stacked_delta_sigma(rs, cons, a_szs, ln_bary_vars)


class StackedLensingSignal:
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
        if not isinstance(log_masses, nothing.NoMasses):
            self.log_masses = log_masses
        else:
            raise TypeError('log_masses must be provided to calculate a stacked model')

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
        self.stacked_model = model.StackedModel(
            self.log_masses,
            self.redshifts,
            cosmo_params=self.cosmo_params,
            units=self.units,
            comoving_radii=self.comoving,
            selection_func_file=self.selection_func_file,
            lensing_weights_file=self.lensing_weights_file,
            delta=self.delta,
            mass_definition=self.mass_definition,
        )

    def stacked_esd(self, rs, params):
        cons = params[:, 0]
        a_szs = params[:, 1]
        try:
            return self.stacked_model.stacked_delta_sigma(rs, cons, a_szs)
        except AttributeError:
            self._init_stacked_model()
            return self.stacked_model.stacked_delta_sigma(rs, cons, a_szs)


class StackedBaryonLensingSignal:
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
        if not isinstance(log_masses, nothing.NoMasses):
            self.log_masses = log_masses
        else:
            raise TypeError('log_masses must be provided to calculate a stacked model')

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

    def _init_baryon_model(self):
        self.baryon_model = model.GnfwBaryonModel(
            self.log_masses,
            self.redshifts,
            cosmo_params=self.cosmo_params,
            units=self.units,
            comoving_radii=self.comoving,
            selection_func_file=self.selection_func_file,
            lensing_weights_file=self.lensing_weights_file,
            delta=self.delta,
            mass_definition=self.mass_definition,
        )

    def stacked_esd(self, rs, params):
        cons = params[:, 0]
        alphas = params[:, 1]
        betas = params[:, 2]
        gammas = params[:, 3]
        a_szs = params[:, 4]
        try:
            return self.baryon_model.stacked_delta_sigma(rs, cons, alphas, betas, gammas, a_szs)
        except AttributeError:
            self._init_baryon_model()
            return self.baryon_model.stacked_delta_sigma(rs, cons, alphas, betas, gammas, a_szs)

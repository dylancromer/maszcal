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
        self.single_mass_model = model.SingleMassNfwShearModel(
            self.redshift,
            cosmo_params=self.cosmo_params,
            comoving_radii=self.comoving,
            delta=self.delta,
            mass_definition=self.mass_definition,
        )

    def esd(self, rs, params):
        log_masses = params[:, 0].flatten()
        concentrations = params[:, 1].flatten()
        try:
            return self.single_mass_model.delta_sigma(rs, log_masses, concentrations)
        except AttributeError:
            self._init_single_mass_model()
            return self.single_mass_model.delta_sigma(rs, log_masses, concentrations)


class NfwLensingSignal:
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
            sz_scatter=0.2,
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

        self.sz_scatter = sz_scatter

    def _init_stacked_model(self):
        self.stacked_model = model.NfwShearModel(
            self.log_masses,
            self.redshifts,
            cosmo_params=self.cosmo_params,
            units=self.units,
            comoving_radii=self.comoving,
            selection_func_file=self.selection_func_file,
            lensing_weights_file=self.lensing_weights_file,
            delta=self.delta,
            mass_definition=self.mass_definition,
            sz_scatter=self.sz_scatter,
        )

    def stacked_esd(self, rs, params):
        cons = params[:, 0].flatten()
        a_szs = params[:, 1].flatten()
        try:
            return self.stacked_model.stacked_delta_sigma(rs, cons, a_szs)
        except AttributeError:
            self._init_stacked_model()
            return self.stacked_model.stacked_delta_sigma(rs, cons, a_szs)

    def avg_mass(self, a_szs):
        try:
            return self.stacked_model.weak_lensing_avg_mass(a_szs)
        except AttributeError:
            self._init_stacked_model()
            return self.stacked_model.weak_lensing_avg_mass(a_szs)


class NfwCmLensingSignal(NfwLensingSignal):
    def _init_stacked_model(self):
        self.stacked_model = model.NfwCmShearModel(
            self.log_masses,
            self.redshifts,
            cosmo_params=self.cosmo_params,
            units=self.units,
            comoving_radii=self.comoving,
            selection_func_file=self.selection_func_file,
            lensing_weights_file=self.lensing_weights_file,
            delta=self.delta,
            mass_definition=self.mass_definition,
            sz_scatter=self.sz_scatter,
        )

    def stacked_esd(self, rs, params):
        a_szs = params.flatten()
        try:
            return self.stacked_model.stacked_delta_sigma(rs, a_szs)
        except AttributeError:
            self._init_stacked_model()
            return self.stacked_model.stacked_delta_sigma(rs, a_szs)

    def avg_mass(self, a_szs):
        try:
            return self.stacked_model.weak_lensing_avg_mass(a_szs)
        except AttributeError:
            self._init_stacked_model()
            return self.stacked_model.weak_lensing_avg_mass(a_szs)


class MiyatakeLensingSignal(NfwLensingSignal):
    def _init_stacked_model(self):
        self.stacked_model = model.MiyatakeShearModel(
            self.log_masses,
            self.redshifts,
            cosmo_params=self.cosmo_params,
            units=self.units,
            comoving_radii=self.comoving,
            selection_func_file=self.selection_func_file,
            lensing_weights_file=self.lensing_weights_file,
            delta=self.delta,
            mass_definition=self.mass_definition,
            sz_scatter=self.sz_scatter,
        )

    def stacked_esd(self, rs, params):
        a_szs = params.flatten()
        try:
            return self.stacked_model.stacked_delta_sigma(rs, a_szs)
        except AttributeError:
            self._init_stacked_model()
            return self.stacked_model.stacked_delta_sigma(rs, a_szs)

    def avg_mass(self, a_szs):
        try:
            return self.stacked_model.weak_lensing_avg_mass(a_szs)
        except AttributeError:
            self._init_stacked_model()
            return self.stacked_model.weak_lensing_avg_mass(a_szs)


class BaryonLensingSignal:
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
            sz_scatter=0.2,
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

        self.sz_scatter = sz_scatter

    def _init_baryon_model(self):
        self.baryon_model = model.BaryonShearModel(
            self.log_masses,
            self.redshifts,
            cosmo_params=self.cosmo_params,
            units=self.units,
            comoving_radii=self.comoving,
            selection_func_file=self.selection_func_file,
            lensing_weights_file=self.lensing_weights_file,
            delta=self.delta,
            mass_definition=self.mass_definition,
            sz_scatter=self.sz_scatter,
        )

    def stacked_esd(self, rs, params):
        cons = params[:, 0].flatten()
        alphas = params[:, 1].flatten()
        betas = params[:, 2].flatten()
        gammas = params[:, 3].flatten()
        a_szs = params[:, 4].flatten()
        try:
            return self.baryon_model.stacked_delta_sigma(rs, cons, alphas, betas, gammas, a_szs)
        except AttributeError:
            self._init_baryon_model()
            return self.baryon_model.stacked_delta_sigma(rs, cons, alphas, betas, gammas, a_szs)

    def avg_mass(self, a_szs):
        try:
            return self.baryon_model.weak_lensing_avg_mass(a_szs)
        except AttributeError:
            self._init_baryon_model()
            return self.baryon_model.weak_lensing_avg_mass(a_szs)


class BaryonCmLensingSignal(BaryonLensingSignal):
    def _init_stacked_model(self):
        self.stacked_model = model.BaryonCmShearModel(
            self.log_masses,
            self.redshifts,
            cosmo_params=self.cosmo_params,
            units=self.units,
            comoving_radii=self.comoving,
            selection_func_file=self.selection_func_file,
            lensing_weights_file=self.lensing_weights_file,
            delta=self.delta,
            mass_definition=self.mass_definition,
            sz_scatter=self.sz_scatter,
        )

    def stacked_esd(self, rs, params):
        alphas = params[:, 0].flatten()
        betas = params[:, 1].flatten()
        gammas = params[:, 2].flatten()
        a_szs = params[:, 3].flatten()
        try:
            return self.stacked_model.stacked_delta_sigma(rs, alphas, betas, gammas, a_szs)
        except AttributeError:
            self._init_stacked_model()
            return self.stacked_model.stacked_delta_sigma(rs, alphas, betas, gammas, a_szs)

    def avg_mass(self, a_szs):
        try:
            return self.stacked_model.weak_lensing_avg_mass(a_szs)
        except AttributeError:
            self._init_stacked_model()
            return self.stacked_model.weak_lensing_avg_mass(a_szs)


class SingleBaryonLensingSignal:
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
        self.single_mass_model = model.SingleMassBaryonShearModel(
            self.redshift,
            cosmo_params=self.cosmo_params,
            comoving_radii=self.comoving,
            delta=self.delta,
            mass_definition=self.mass_definition,
        )

    def esd(self, rs, params):
        log_masses = params[:, 0].flatten()
        cons = params[:, 1].flatten()
        alphas = params[:, 2].flatten()
        betas = params[:, 3].flatten()
        gammas = params[:, 4].flatten()
        try:
            return self.single_mass_model.delta_sigma(rs, log_masses, cons, alphas, betas, gammas)
        except AttributeError:
            self._init_single_mass_model()
            return self.single_mass_model.delta_sigma(rs, log_masses, cons, alphas, betas, gammas)

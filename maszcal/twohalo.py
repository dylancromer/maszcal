from dataclasses import dataclass
import numpy as np
import scipy.integrate
import astropy.units as u
import projector
import maszcal.interpolate
import maszcal.interp_utils
import maszcal.matter
import maszcal.mathutils
import maszcal.cosmo_utils
import maszcal.tinker


@dataclass
class TwoHaloShearModel:
    MIN_K = 1e-5
    MAX_K = 1e2
    MAX_BIAS_K = 0.3
    NUM_BIAS_KS = 200
    MIN_REDSHIFT = 0
    MAX_REDSHIFT = 1
    NUM_INTERP_ZS = 40
    MIN_RADIUS = 1e-6
    MAX_RADIUS = 30
    NUM_INTERP_RADII = 200
    USE_NONLINEAR_MATTER_POWER = True
    QUAD_ITER_LIMIT = 2000

    cosmo_params: object
    units: object = u.Msun/u.pc**2
    delta: int = 200
    mass_definition: str = 'mean'
    comoving: bool = True
    matter_power_class: object = maszcal.matter.Power

    def __post_init__(self):
        self.astropy_cosmology = maszcal.cosmo_utils.get_astropy_cosmology(self.cosmo_params)

    def _init_tinker_bias(self):
        tinker_bias_model = maszcal.tinker.TinkerBias(
            delta=self.delta,
            mass_definition=self.mass_definition,
            astropy_cosmology=self.astropy_cosmology,
            comoving=self.comoving,
        )
        self.__bias = tinker_bias_model.bias

    def _bias(self, mus, zs):
        masses = np.exp(mus)
        ks = np.logspace(np.log10(self.MIN_K), np.log10(self.MAX_BIAS_K), self.NUM_BIAS_KS)

        try:
            power_spect = self._power_interpolator(ks, zs)
        except AttributeError:
            self._init_power_interpolator()
            power_spect = self._power_interpolator(ks, zs)

        try:
            return self.__bias(masses, zs, ks, power_spect)
        except AttributeError:
            self._init_tinker_bias()
            return self.__bias(masses, zs, ks, power_spect)

    def _init_power_interpolator(self):
        power = self.matter_power_class(cosmo_params=self.cosmo_params)

        dummy_ks = np.linspace(self.MIN_K, self.MAX_K, 2)
        dummy_zs = np.linspace(self.MIN_REDSHIFT, self.MAX_REDSHIFT, 6)

        self._power_interpolator = power.get_spectrum_interpolator(dummy_ks, dummy_zs, is_nonlinear=self.USE_NONLINEAR_MATTER_POWER)

    def _correlation_integrand(self, ln_k, rs, zs):
        k = np.exp(np.array([ln_k]))

        try:
            power_spect = self._power_interpolator(k, zs)[None, :, :]
        except AttributeError:
            self._init_power_interpolator()
            power_spect = self._power_interpolator(k, zs)[None, :, :]

        k = k[None, :, None]
        rs = rs[:, None, None]

        return k * k**2 * power_spect * np.sinc(k * rs) / (2 * np.pi**2) # extra k factor from log coordinates

    def _get_correlator_samples(self):
        sample_rs = np.logspace(np.log10(self.MIN_RADIUS), np.log10(self.MAX_RADIUS), self.NUM_INTERP_RADII)
        sample_zs = np.linspace(self.MIN_REDSHIFT, self.MAX_REDSHIFT, self.NUM_INTERP_ZS)
        xi_integral = scipy.integrate.quad_vec(
            lambda ln_k: self._correlation_integrand(ln_k, sample_rs, sample_zs),
            np.log(self.MIN_K),
            np.log(self.MAX_K),
            limit=self.QUAD_ITER_LIMIT,
        )

        sample_xis = np.squeeze(xi_integral[0])
        xi_errors = xi_integral[1]

        params = maszcal.interp_utils.cartesian_prod(np.log(sample_rs), sample_zs)
        return params, np.log(sample_xis)

    def _init_correlation_interpolator(self):
        sample_params, sample_ln_xis = self._get_correlator_samples()
        interpolator = maszcal.interpolate.RbfInterpolator(params=sample_params, func_vals=sample_ln_xis)
        interpolator.process()
        self._correlation_rbf = interpolator

    def _correlation_integral_interp(self, rs, zs):
        ln_rs = np.log(rs).flatten()
        params = maszcal.interp_utils.cartesian_prod(ln_rs, zs)
        return np.exp(self._correlation_rbf.interp(params).flatten()).reshape(rs.shape + zs.shape)

    def _correlation_interpolator(self, rs, zs):
        try:
            correlator = self._correlation_integral_interp(rs, zs)
        except AttributeError:
            self._init_correlation_interpolator()
            correlator = self._correlation_integral_interp(rs, zs)
        return correlator

    def density_interpolator(self, rs, zs):
        return self._correlation_interpolator(rs, zs)

    def _esd_radial_shape(self, rs, zs):
        return projector.esd_quad(rs, lambda radii: self.density_interpolator(radii, zs))

    def _esd(self, rs, mus, zs):
        total_dim = rs.ndim + mus.ndim + zs.ndim
        bias = maszcal.mathutils.atleast_kd(self._bias(mus, zs).T, total_dim, append_dims=False)
        esd_radial_shape = self._esd_radial_shape(rs, zs)
        esd_radial_shape = maszcal.mathutils.atleast_kd(esd_radial_shape, total_dim, append_dims=True)
        return np.swapaxes(
            bias * esd_radial_shape,
            0,
            2,
        )

    def matter_density(self, zs):
        return (self.astropy_cosmology.Om(zs)
                * self.astropy_cosmology.critical_density(zs)).to(u.Msun/u.Mpc**3).value

    def esd(self, rs, mus, zs):
        return self.matter_density(zs)[None, :, None] * self._esd(rs, mus, zs) * (u.Msun/u.Mpc**2).to(self.units)


@dataclass
class EmulatedTwoHaloShear:
    radii: np.ndarray
    redshifts: np.ndarray
    cosmo_params: object
    units: object = u.Msun/u.pc**2
    delta: int = 200
    mass_definition: str = 'mean'
    comoving: bool = True
    two_halo_shear_class: object = TwoHaloShearModel
    matter_power_class: object = maszcal.matter.Power

    def __post_init__(self):
        self.two_halo_shear_model = self.two_halo_shear_class(
            cosmo_params=self.cosmo_params,
            units=self.units,
            delta=self.delta,
            mass_definition=self.mass_definition,
            comoving=self.comoving,
            matter_power_class=self.matter_power_class,
        )

    def _get_data_grid(self, rs, zs):
        return rs[:, None] * self.two_halo_shear_model._esd_radial_shape(rs, zs)

    def process(self):
        data_grid = self._get_data_grid(self.radii, self.redshifts)
        params_grid = maszcal.interp_utils.cartesian_prod(self.radii, self.redshifts)
        interpolator = maszcal.interpolate.RbfInterpolator(params=params_grid, func_vals=data_grid)
        interpolator.process()
        self._radial_shape_interpolator = interpolator

    def _radial_shape(self, rs, zs):
        params = maszcal.interp_utils.cartesian_prod(rs, zs)
        return self._radial_shape_interpolator.interp(params)

    def esd(self, rs, mus, zs):
        total_dim = rs.ndim + mus.ndim + zs.ndim

        bias = self.two_halo_shear_model._bias(mus, zs).T
        bias = maszcal.mathutils.atleast_kd(bias, total_dim, append_dims=False)

        esd_radial_shape = self._radial_shape(rs, zs).reshape(rs.shape + zs.shape) / rs[:, None]
        esd_radial_shape = maszcal.mathutils.atleast_kd(esd_radial_shape, total_dim, append_dims=True)
        return np.swapaxes(
            bias * esd_radial_shape,
            0,
            2,
        )

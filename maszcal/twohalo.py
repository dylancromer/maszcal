from functools import partial
from dataclasses import dataclass
from types import MappingProxyType
import numpy as np
import astropy.units as u
import scipy.interpolate
import supercubos
import projector
import maszcal.interpolate
import maszcal.interp_utils
import maszcal.matter
import maszcal.mathutils
import maszcal.cosmo_utils
import maszcal.tinker
import maszcal.emulate


@dataclass
class TwoHaloModel:
    MIN_K = 1e-4
    MAX_K = 1e2
    NUM_KS = 800
    MAX_BIAS_K = 1e2
    NUM_BIAS_KS = 400
    MIN_REDSHIFT = 0
    MAX_REDSHIFT = 1
    NUM_INTERP_ZS = 40
    USE_NONLINEAR_MATTER_POWER_FOR_BIAS = False
    COMOVING = True

    cosmo_params: object
    units: object = u.Msun/u.pc**2
    delta: int = 200
    mass_definition: str = 'mean'
    is_nonlinear: bool = False
    matter_power_class: object = maszcal.matter.Power
    esd_kwargs: MappingProxyType = MappingProxyType({'radial_axis_to_broadcast': None})
    sd_kwargs: MappingProxyType = MappingProxyType({'radial_axis_to_broadcast': None})

    def __post_init__(self):
        self.astropy_cosmology = maszcal.cosmo_utils.get_astropy_cosmology(self.cosmo_params)

    def _init_tinker_bias(self):
        tinker_bias_model = maszcal.tinker.TinkerBias(
            delta=self.delta,
            mass_definition=self.mass_definition,
            astropy_cosmology=self.astropy_cosmology,
            comoving=self.COMOVING,
        )
        self.__bias = tinker_bias_model.bias

    def _init_power_interpolator(self):
        power = self.matter_power_class(cosmo_params=self.cosmo_params)
        dummy_ks = np.linspace(self.MIN_K, self.MAX_K, self.NUM_KS)
        dummy_zs = np.linspace(self.MIN_REDSHIFT, self.MAX_REDSHIFT, self.NUM_INTERP_ZS)
        self._power_interpolator = power.get_spectrum_interpolator(dummy_ks, dummy_zs, is_nonlinear=self.USE_NONLINEAR_MATTER_POWER_FOR_BIAS)

    def _bias(self, zs, mus):
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

    def _init_correlation_interpolator(self, zs):
        self.__correlation_interpolator = maszcal.matter.Correlations.from_cosmology(
            self.cosmo_params,
            zs,
            is_nonlinear=self.is_nonlinear,
        )

    def _correlation_interpolator(self, rs, zs):
        try:
            correlator = self.__correlation_interpolator(rs)
        except AttributeError:
            self._init_correlation_interpolator(zs)
            correlator = self.__correlation_interpolator(rs)
        return correlator

    def _correlation_interpolator_redshift_dep_radii(self, rs, zs):
        try:
            correlator = self.__correlation_interpolator.with_redshift_dependent_radii(rs)
        except AttributeError:
            self._init_correlation_interpolator(zs)
            correlator = self.__correlation_interpolator.with_redshift_dependent_radii(rs)
        return correlator

    def _density_shape_interpolator(self, rs, zs):
        corr = self._correlation_interpolator(rs.flatten(), zs).reshape(zs.shape + rs.shape)
        corr = np.moveaxis(corr, 0, -1)
        return corr

    def _density_shape_interpolator_theta_coords(self, rs, zs):
        corr = self._correlation_interpolator_redshift_dep_radii(rs, zs)
        corr = np.moveaxis(corr, 0, -1)
        return corr

    def matter_density(self, zs):
        if self.COMOVING:
            return np.ones_like(zs)*(self.astropy_cosmology.Om0
                    * self.astropy_cosmology.critical_density0).to(u.Msun/u.Mpc**3).value
        else:
            return (self.astropy_cosmology.Om(zs)
                    * self.astropy_cosmology.critical_density(zs)).to(u.Msun/u.Mpc**3).value

    def halo_matter_correlation(self, rs, zs, mus):
        bias = self._bias(zs, mus)[:, None]
        mm_corr = self._correlation_interpolator(rs, zs)
        return bias * mm_corr


class TwoHaloShearModel(TwoHaloModel):
    def _excess_surface_density_radial_shape(self, rs, zs):
        return projector.ExcessSurfaceDensity.calculate(rs, lambda radii: self._density_shape_interpolator(radii, zs), **self.esd_kwargs)

    def _excess_surface_density(self, rs, zs, mus):
        bias = self._bias(zs, mus)[:, None]
        excess_surface_density_radial_shape = self._excess_surface_density_radial_shape(rs, zs).T
        return bias * excess_surface_density_radial_shape

    def excess_surface_density(self, rs, zs, mus):
        return self.matter_density(zs)[:, None] * self._excess_surface_density(rs, zs, mus) * (u.Msun/u.Mpc**2).to(self.units)


class TwoHaloConvergenceModel(TwoHaloModel):
    CMB_REDSHIFT = 1100

    def __post_init__(self):
        self.astropy_cosmology = maszcal.cosmo_utils.get_astropy_cosmology(self.cosmo_params)
        self.sigma_crit = partial(
            maszcal.cosmology.SigmaCrit(self.cosmo_params, comoving=self.COMOVING, units=self.units).sdc,
            z_source=np.array([self.CMB_REDSHIFT]),
        )

    def _sd_radial_shape_for_theta(self, rs, zs):
        return projector.SurfaceDensity2.calculate(
            rs,
            lambda radii: self._density_shape_interpolator_theta_coords(radii, zs),
            **self.sd_kwargs,
        )

    def _sd_radial_shape(self, rs, zs):
        return projector.SurfaceDensity2.calculate(
            rs,
            lambda radii: self._density_shape_interpolator(radii, zs),
            **self.sd_kwargs,
        )

    def _radius_space_convergence_for_theta(self, rs, zs, mus):
        bias = self._bias(zs, mus)[:, None]
        sd_radial_shape = self._sd_radial_shape_for_theta(rs, zs).T
        return bias * sd_radial_shape / self.sigma_crit(z_lens=zs)[:, None]

    def _radius_space_convergence(self, rs, zs, mus):
        bias = self._bias(zs, mus)[:, None]
        sd_radial_shape = self._sd_radial_shape(rs, zs).T
        return bias * sd_radial_shape / self.sigma_crit(z_lens=zs)[:, None]

    def radius_space_convergence(self, rs, zs, mus):
        return self.matter_density(zs)[:, None] * self._radius_space_convergence(rs, zs, mus) * (u.Msun/u.Mpc**2).to(self.units)

    def radius_space_convergence_for_theta(self, rs, zs, mus):
        return self.matter_density(zs)[:, None] * self._radius_space_convergence_for_theta(rs, zs, mus) * (u.Msun/u.Mpc**2).to(self.units)

    def _comoving_distance(self, z):
        return self.astropy_cosmology.comoving_distance(z).to(u.Mpc).value

    def _angular_diameter_distance(self, z):
        return self.astropy_cosmology.angular_diameter_distance(z).to(u.Mpc).value

    def angle_scale_distance(self, z):
        if self.COMOVING:
            return self._comoving_distance(z)
        else:
            return self._angular_diameter_distance(z)

    def convergence(self, thetas, zs, mus):
        radii_of_z = thetas[:, None] * self.angle_scale_distance(zs)[None, :]
        return self.radius_space_convergence_for_theta(radii_of_z, zs, mus)


@dataclass
class TwoHaloEmulator:
    INTERPOLATOR_CLASS = maszcal.interpolate.RbfInterpolator
    NUM_PRINCIPAL_COMPONENTS = 6

    two_halo_samples: np.ndarray
    r_grid: np.ndarray
    z_mu_samples: np.ndarray
    separate_mu_and_z_axes: bool

    @classmethod
    def from_function(cls, two_halo_func, r_grid, z_lims, mu_lims, num_emulator_samples=600, separate_mu_and_z_axes=False):
        z_mu_samples = cls._get_z_mu_samples(z_lims, mu_lims, num_emulator_samples)
        sampled_two_halo_term = cls._get_two_halo_term_samples(r_grid, z_mu_samples, two_halo_func)
        return cls(
            two_halo_samples=sampled_two_halo_term,
            r_grid=r_grid,
            z_mu_samples=z_mu_samples,
            separate_mu_and_z_axes=separate_mu_and_z_axes,
        )

    @staticmethod
    def _get_z_mu_samples(z_lims, mu_lims, num_emulator_samples):
        param_mins = np.stack((z_lims, mu_lims))[:, 0]
        param_maxes = np.stack((z_lims, mu_lims))[:, 1]
        return supercubos.LatinSampler().get_lh_sample(
            param_mins=param_mins,
            param_maxes=param_maxes,
            num_samples=num_emulator_samples,
        )

    @staticmethod
    def _get_two_halo_term_samples(r_grid, z_mu_samples, two_halo_func):
        zs, mus = z_mu_samples.T
        sort_index = zs.argsort()
        inverse_index = sort_index.argsort()
        zs = zs[sort_index]
        mus = mus[sort_index]
        sampled_two_halo_term = two_halo_func(r_grid, zs, mus)
        return sampled_two_halo_term[inverse_index, :]

    def _wrap_emulator(self, unwrapped_emulator):
        if not self.separate_mu_and_z_axes:
            def wrapped_emulator(zs, mus): return unwrapped_emulator(np.stack((zs, mus)).T).T
        else:
            def wrapped_emulator(zs, mus):
                coords = maszcal.interp_utils.cartesian_prod(zs, mus)
                return np.moveaxis(
                    unwrapped_emulator(coords).reshape(-1, zs.size, mus.size),
                    (0, 1),
                    (2, 1),
                )
        return wrapped_emulator

    def _get_emulator(self, z_mu_samples, sampled_two_halo_term):
        emulator_ =  maszcal.emulate.PcaEmulator.create_from_data(
            z_mu_samples,
            sampled_two_halo_term.T,
            interpolator_class=self.INTERPOLATOR_CLASS,
            num_components=self.NUM_PRINCIPAL_COMPONENTS
        )
        return self._wrap_emulator(emulator_)

    def __post_init__(self):
        self._emulator = self._get_emulator(self.z_mu_samples, self.two_halo_samples)

    def _get_radial_interpolation(self, rs, zs, mus):
        return scipy.interpolate.interp1d(
            self.r_grid,
            self._emulator(zs, mus),
            kind='cubic',
            axis=-1,
        )(rs)

    def _get_redshift_dependent_radial_interpolation(self, rs, zs, mus):
        return np.array(
            [self._get_radial_interpolation(rs[:, i], zs[i:i+1], mus[i:i+1]) for i, _ in enumerate(zs)],
        ).squeeze(axis=1)

    def _get_redshift_dependent_radial_interpolation_sep_mu_z(self, rs, zs, mus):
        return np.swapaxes(
            np.array(
                [self._get_radial_interpolation(rs[:, i], zs[i:i+1], mus) for i, z in enumerate(zs)],
            ).squeeze(axis=2),
            0,
            1,
        )

    def _check_radii_are_flat(self, radii):
        if (radii.ndim > 1) and (radii.size != radii.shape[0]):
            raise ValueError('radii with ndim > 1 must use the with_redshift_dependent_radii method')

    def __call__(self, rs, zs, mus):
        self._check_radii_are_flat(rs)
        return self._get_radial_interpolation(rs.flatten(), zs, mus)

    def with_redshift_dependent_radii(self, rs, zs, mus):
        if not self.separate_mu_and_z_axes:
            return self._get_redshift_dependent_radial_interpolation(rs, zs, mus)
        else:
            return self._get_redshift_dependent_radial_interpolation_sep_mu_z(rs, zs, mus)

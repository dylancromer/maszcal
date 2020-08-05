from functools import partial
from dataclasses import dataclass
from types import MappingProxyType
import numpy as np
import astropy.units as u
import projector
import maszcal.interpolate
import maszcal.interp_utils
import maszcal.matter
import maszcal.mathutils
import maszcal.cosmo_utils
import maszcal.tinker


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
    USE_NONLINEAR_MATTER_POWER_FOR_BIAS = True

    cosmo_params: object
    units: object = u.Msun/u.pc**2
    delta: int = 200
    mass_definition: str = 'mean'
    comoving: bool = True
    is_nonlinear: bool = True
    matter_power_class: object = maszcal.matter.Power
    projector_kwargs: object = MappingProxyType({})

    def __post_init__(self):
        self.astropy_cosmology = maszcal.cosmo_utils.get_astropy_cosmology(self.cosmo_params)

        if not self.comoving:
            raise NotImplementedError('TwoHaloModel has not yet implemented a non-comoving option')

    def _init_tinker_bias(self):
        tinker_bias_model = maszcal.tinker.TinkerBias(
            delta=self.delta,
            mass_definition=self.mass_definition,
            astropy_cosmology=self.astropy_cosmology,
            comoving=self.comoving,
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

    def _density_shape_interpolator(self, rs, zs):
        corr = self._correlation_interpolator(rs.flatten(), zs).reshape(zs.shape + rs.shape)
        corr = np.moveaxis(corr, 0, -1)
        return corr

    def matter_density(self, zs):
        return (self.astropy_cosmology.Om(zs)
                * self.astropy_cosmology.critical_density(zs)).to(u.Msun/u.Mpc**3).value

    def halo_matter_correlation(self, rs, zs, mus):
        bias = self._bias(zs, mus)[:, None]
        mm_corr = self._correlation_interpolator(rs, zs)
        return bias * mm_corr


class TwoHaloShearModel(TwoHaloModel):
    def _esd_radial_shape(self, rs, zs):
        return projector.esd(rs, lambda radii: self._density_shape_interpolator(radii, zs), **self.projector_kwargs)

    def _esd(self, rs, zs, mus):
        bias = self._bias(zs, mus)[:, None]
        esd_radial_shape = self._esd_radial_shape(rs, zs).T
        return bias * esd_radial_shape

    def esd(self, rs, zs, mus):
        return self.matter_density(zs)[:, None] * self._esd(rs, zs, mus) * (u.Msun/u.Mpc**2).to(self.units)


class TwoHaloConvergenceModel(TwoHaloModel):
    CMB_REDSHIFT = 1100

    def __post_init__(self):
        self.astropy_cosmology = maszcal.cosmo_utils.get_astropy_cosmology(self.cosmo_params)
        if not self.comoving:
            raise NotImplementedError('TwoHaloModel has not yet implemented a non-comoving option')
        self.sigma_crit = partial(
            maszcal.cosmology.SigmaCrit(self.cosmo_params, units=self.units).sdc,
            z_source=np.array([self.CMB_REDSHIFT]),
        )

    def _sd_radial_shape(self, rs, zs):
        return projector.sd(rs, lambda radii: self._density_shape_interpolator(radii, zs), **self.projector_kwargs)

    def __radius_space_kappa(self, rs, zs, mus):
        bias = self._bias(zs, mus)[:, None]
        sd_radial_shape = self._sd_radial_shape(rs, zs).T
        return bias * sd_radial_shape / self.sigma_crit(z_lens=zs)[:, None]

    def _radius_space_kappa(self, rs, zs, mus):
        return self.matter_density(zs)[:, None] * self.__radius_space_kappa(rs, zs, mus) * (u.Msun/u.Mpc**2).to(self.units)

    def _comoving_distance(self, z):
        return self.astropy_cosmology.comoving_distance(z).to(u.Mpc).value

    def kappa(self, thetas, zs, mus):
        radii_of_z = [thetas * self._comoving_distance(z) for z in zs]
        return np.array([
            self._radius_space_kappa(rs, zs[i:i+1], mus[i:i+1])
            for i, rs in enumerate(radii_of_z)
        ]).squeeze()

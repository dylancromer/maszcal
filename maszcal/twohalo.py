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

    def _init_power_interpolator(self):
        power = self.matter_power_class(cosmo_params=self.cosmo_params)
        dummy_ks = np.linspace(self.MIN_K, self.MAX_K, self.NUM_KS)
        dummy_zs = np.linspace(self.MIN_REDSHIFT, self.MAX_REDSHIFT, self.NUM_INTERP_ZS)
        self._power_interpolator = power.get_spectrum_interpolator(dummy_ks, dummy_zs, is_nonlinear=self.USE_NONLINEAR_MATTER_POWER_FOR_BIAS)

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
        return 1 + corr

    def _esd_radial_shape(self, rs, zs):
        return projector.esd_quad(rs, lambda radii: self._density_shape_interpolator(radii, zs))

    def _esd(self, rs, mus, zs):
        bias = self._bias(mus, zs)[:, None, :]
        esd_radial_shape = self._esd_radial_shape(rs, zs)[None, :, :]
        return np.swapaxes(
            bias * esd_radial_shape,
            1,
            2,
        )

    def matter_density(self, zs):
        return (self.astropy_cosmology.Om(zs)
                * self.astropy_cosmology.critical_density(zs)).to(u.Msun/u.Mpc**3).value

    def halo_matter_correlation(self, rs, mus, zs):
        bias = self._bias(mus, zs)[:, None, :]
        mm_corr = self._correlation_interpolator(rs, zs).T[None, :, :]
        return np.swapaxes(
            bias * mm_corr,
            1,
            2,
        )

    def esd(self, rs, mus, zs):
        return self.matter_density(zs)[None, :, None] * self._esd(rs, mus, zs) * (u.Msun/u.Mpc**2).to(self.units)

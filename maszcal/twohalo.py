from dataclasses import dataclass
import numpy as np
import scipy.integrate
import projector
import maszcal.interpolate
import maszcal.interp_utils


@dataclass
class TwoHaloShearModel:
    MIN_K = 1e-5
    MAX_K = 1e2
    MIN_REDSHIFT = 0
    MAX_REDSHIFT = 1
    NUM_INTERP_ZS = 40
    MIN_RADIUS = 1e-6
    MAX_RADIUS = 30
    NUM_INTERP_RADII = 200

    cosmo_params: object

    def _bias(self, mus):
        return 1

    def _init_power_interpolator(self):
        pass

    def _correlation_integrand(self, ln_k, rs, zs):
        k = np.exp(np.array([ln_k]))

        try:
            power_spect = self._power_interpolator(k, zs)[None, :, :]
        except AttributeError:
            self._init_power_interpolator()
            power_spect = self._power_interpolator(k, zs)[None, :, :]

        k = k[None, :, None]
        rs = rs[:, None, None]

        return k * k**2 * power_spect * np.sinc(k * r) / (2 * np.pi**2) # extra k factor from log coordinates

    def _init_correlation_interpolator(self):
        sample_rs = np.logspace(np.log10(self.MIN_RADIUS), np.log10(self.MAX_RADIUS), self.NUM_INTERP_RADII)
        sample_zs = np.linspace(self.MIN_REDSHIFT, self.MAX_REDSHIFT, self.NUM_INTERP_ZS)
        sample_xis = scipy.integrate.quad_vec(
            lambda ln_k: self._correlation_integrand(ln_k, sample_rs, sample_zs),
            np.log(1e-5),
            np.log(100),
            limit=2000,
        )

    def _correlation_interpolator(self, rs, mus, zs):
        bias = self._bias(mus)
        try:
            correlator = self._correlation_integral_interp(rs, zs)
        except AttributeError:
            self._init_correlation_interpolator()
            correlator = self._correlation_integral_interp(rs, zs)
        return bias * correlator

    def density_interpolator(self, rs, mus, zs):
        return 1 + self._correlation_interpolator(rs, mus, zs)

    def esd(self, rs, mus, zs):
        return projector.esd_quad(rs, lambda radii: self.density_interpolator(radii, mus, zs))

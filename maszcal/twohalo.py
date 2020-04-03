from dataclasses import dataclass
import numpy as np
import scipy.integrate
import projector
import maszcal.interpolate
import maszcal.interp_utils
import maszcal.matter
import maszcal.mathutils


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
    USE_NONLINEAR_MATTER_POWER = True

    cosmo_params: object
    matter_power_class: object = maszcal.matter.Power

    def _bias(self, mus):
        return np.ones(mus.shape)

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
            np.log(1e-5),
            np.log(100),
            limit=2000,
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

    def _correlation_interpolator(self, rs, mus, zs):
        total_dim = rs.ndim + mus.ndim + zs.ndim
        bias = maszcal.mathutils.atleast_kd(self._bias(mus), total_dim, append_dims=False)
        try:
            correlator = self._correlation_integral_interp(rs, zs)
        except AttributeError:
            self._init_correlation_interpolator()
            correlator = self._correlation_integral_interp(rs, zs)
        correlator = maszcal.mathutils.atleast_kd(correlator, total_dim, append_dims=True)

        return bias * correlator

    def density_interpolator(self, rs, mus, zs):
        return 1 + self._correlation_interpolator(rs, mus, zs)

    def esd(self, rs, mus, zs):
        return np.swapaxes(
            projector.esd_quad(rs, lambda radii: self.density_interpolator(radii, mus, zs)),
            0,
            2,
        )

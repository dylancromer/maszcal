from dataclasses import dataclass
import numpy as np
import scipy.interpolate
import camb
import mcfit
import maszcal.cosmo_utils


@dataclass
class Power:
    cosmo_params: object
    MAX_K_OFFSET: float = 0.1

    def spectrum(self, ks, zs, is_nonlinear):
        interpolator = self.get_spectrum_interpolator(ks, zs, is_nonlinear)
        return interpolator(ks, zs)

    def get_spectrum_interpolator(self, ks, zs, is_nonlinear):
        max_k = ks.max() + self.MAX_K_OFFSET
        camb_params = maszcal.cosmo_utils.get_camb_params(self.cosmo_params, max_k, zs, is_nonlinear)

        camb_results = camb.get_results(camb_params)
        camb_results.calc_power_spectra()

        interpolator = camb_results.get_matter_power_interpolator(nonlinear=is_nonlinear, k_hunit=False, hubble_units=False)

        return lambda ks, zs: interpolator.P(zs, ks)

    def spectrum_nointerp(self, ks, zs, is_nonlinear):
        max_k = ks.max() + self.MAX_K_OFFSET
        camb_params = maszcal.cosmo_utils.get_camb_params(self.cosmo_params, max_k, zs, is_nonlinear)

        camb_results = camb.get_results(camb_params)
        camb_results.calc_power_spectra()

        ret_ks, ret_zs, ps = camb_results.get_matter_power_spectrum(maxkh=ks.max(), npoints=ks.size)

        return ret_ks, ps


class Correlations:
    SPLINE_DEGREE = 3
    MAX_LOG10_K = 2
    MIN_LOG10_k = -4
    NUM_KS = 800
    NUM_ZS = 40
    MIN_Z = 0
    MAX_Z = 1.1
    EXTRAPOLATE_OPTION = 1  # 0->extrapolate, 1->0-fill, 2->ValueError, 3->boundary_value

    def __init__(self, radius_samples, redshift_samples, xi_samples):
        num_zs = redshift_samples.size

        self.interpolators = [
            scipy.interpolate.InterpolatedUnivariateSpline(
                radius_samples,
                xi_samples[i, :],
                k=self.SPLINE_DEGREE,
                ext=self.EXTRAPOLATE_OPTION
            ) for i in range(num_zs)
        ]

    def __call__(self, rs):
        num_zs = len(self.interpolators)
        return np.array([self.interpolators[i](rs) for i in range(num_zs)])

    def with_redshift_dependent_radii(self, rs):
        num_zs = len(self.interpolators)
        return np.array([self.interpolators[i](rs[..., i]) for i in range(num_zs)])

    @classmethod
    def from_power_spectrum(cls, ks, zs, power_spectra):
        rs, xis = mcfit.P2xi(ks, lowring=True)(power_spectra, extrap=True)
        return cls(rs, zs, xis)

    @classmethod
    def _make_z_grid(cls):
        return np.linspace(cls.MIN_Z, cls.MAX_Z, cls.NUM_ZS)

    @classmethod
    def from_cosmology(cls, cosmo_params, zs, is_nonlinear):
        ks = np.logspace(cls.MIN_LOG10_k, cls.MAX_LOG10_K, cls.NUM_KS)
        power_spectra = Power(cosmo_params).get_spectrum_interpolator(
            ks,
            cls._make_z_grid(),
            is_nonlinear,
        )(ks, zs)
        return cls.from_power_spectrum(ks, zs, power_spectra)

from dataclasses import dataclass
import camb
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

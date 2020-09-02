from dataclasses import dataclass
import numpy as np
import astropy.units as u
import projector
import maszcal.density
import maszcal.cosmology
import maszcal.lensing
import maszcal.mathutils
import maszcal.interp_utils


@dataclass
class Matching2HaloCorrection:
    radii: np.ndarray
    one_halo_func: object
    two_halo_func: object

    def _combine_1_and_2_halo_terms(self, a_2hs, one_halo, two_halo):
        two_halo = two_halo[..., None, None] * a_2hs[None, None, None, :]
        return np.where(one_halo > two_halo, one_halo, two_halo)

    def _get_one_halo(self, zs, mus, *one_halo_params):
        return self.one_halo_func(self.radii, zs, mus, *one_halo_params)[..., None]

    def _get_two_halo(self, zs, mus):
        return np.moveaxis(
            self.two_halo_func(zs, mus),
            1,
            0,
        )

    def excess_surface_density(self, a_2hs, zs, mus, *one_halo_params):
        one_halo_excess_surface_densities = self._get_one_halo(zs, mus, *one_halo_params)
        two_halo_excess_surface_densities = self._get_two_halo(zs, mus)
        return self._combine_1_and_2_halo_terms(a_2hs, one_halo_excess_surface_densities, two_halo_excess_surface_densities)

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
    one_halo_func: object
    two_halo_func: object

    def _combine_1_and_2_halo_terms(self, one_halo, two_halo):
        return np.where(one_halo > two_halo, one_halo, two_halo)

    def _get_one_halo(self, rs, zs, mus, *one_halo_params):
        return self.one_halo_func(rs, zs, mus, *one_halo_params)

    def _get_two_halo(self, rs, zs, mus, a_2hs):
        return np.moveaxis(
            self.two_halo_func(rs, zs, mus)[..., None] * a_2hs[None, None, :],
            1,
            0,
        )

    def corrected_profile(self, rs, zs, mus, a_2hs, *one_halo_params):
        one_halo_excess_surface_densities = self._get_one_halo(rs, zs, mus, *one_halo_params)
        two_halo_excess_surface_densities = self._get_two_halo(rs, zs, mus, a_2hs)
        return self._combine_1_and_2_halo_terms(one_halo_excess_surface_densities, two_halo_excess_surface_densities)


@dataclass
class TwoHaloCorrection:
    one_halo_func: object
    two_halo_func: object

    def _combine_1_and_2_halo_terms(self, one_halo, two_halo):
        return np.where(one_halo > two_halo, one_halo, two_halo)

    def _get_one_halo(self, rs, zs, mus, *one_halo_params):
        return self.one_halo_func(rs, zs, mus, *one_halo_params)

    def _get_two_halo(self, rs, zs, mus, a_2hs):
        return np.moveaxis(
            self.two_halo_func(rs, zs, mus)[..., None] * a_2hs[None, None, None, :],
            2,
            0,
        )

    def corrected_profile(self, rs, zs, mus, a_2hs, *one_halo_params):
        one_halo_excess_surface_densities = self._get_one_halo(rs, zs, mus, *one_halo_params)
        two_halo_excess_surface_densities = self._get_two_halo(rs, zs, mus, a_2hs)
        return self._combine_1_and_2_halo_terms(one_halo_excess_surface_densities, two_halo_excess_surface_densities)

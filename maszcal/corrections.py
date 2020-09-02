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
class SingleMass2HaloCorrection:
    radii: np.ndarray
    one_halo_rho_func: object
    one_halo_shear_class: object
    two_halo_term_function: object
    mass_definition: str = 'mean'
    delta: int = 200
    units: u.Quantity = u.Msun/u.pc**2
    esd_func: object = projector.esd

    def __post_init__(self):
        self._one_halo_shear = self.one_halo_shear_class(
            rho_func=self.one_halo_rho_func,
            units=self.units,
            esd_func=self.esd_func,
        )

    def _one_halo_excess_surface_density(self, zs, mus, *args):
        return self._one_halo_shear.excess_surface_density(self.radii, zs, mus, *args)

    def _two_halo_excess_surface_density(self, zs, mus):
        zs_twohalo, mus_twohalo = maszcal.interp_utils.cartesian_prod(zs, mus).T
        return np.swapaxes(
            self.two_halo_term_function(zs_twohalo, mus_twohalo).reshape(mus.size, zs.size, -1),
            -1,
            0,
        )

    def _combine_1_and_2_halo_terms(self, a_2hs, one_halo, two_halo):
        two_halo = two_halo * a_2hs
        return np.where(one_halo > two_halo, one_halo, two_halo)

    def excess_surface_density(self, a_2hs, zs, mus, *one_halo_params):
        one_halo = self._one_halo_excess_surface_density(zs, mus, *one_halo_params)
        two_halo = self._two_halo_excess_surface_density(zs, mus)
        return self._combine_1_and_2_halo_terms(a_2hs, one_halo, two_halo)


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

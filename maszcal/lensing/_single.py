from dataclasses import dataclass
import numpy as np
from astropy import units as u
import projector
from maszcal.cosmo_utils import get_astropy_cosmology
from maszcal.cosmology import CosmoParams
import maszcal.density
import maszcal.matter
import maszcal.mathutils
import maszcal.ioutils
import maszcal.defaults
from . import _core


@dataclass
class SingleMassConvergenceModel:
    rho_func: object
    units: u.Quantity = u.Msun/u.pc**2
    comoving: bool = True
    convergence_class: object = _core.Convergence
    cosmo_params: maszcal.cosmology.CosmoParams = maszcal.cosmology.CosmoParams()
    sd_func: object = projector.sd

    def __post_init__(self):
        self._convergence = self.convergence_class(
            rho_func=self.rho_func,
            cosmo_params=self.cosmo_params,
            units=self.units,
            comoving=self.comoving,
            sd_func=self.sd_func,
        )
        self.astropy_cosmology = maszcal.cosmo_utils.get_astropy_cosmology(self.cosmo_params)

    def _radius_space_convergence(self, rs, zs, mus, *rho_params):
        return self._convergence.convergence(rs, zs, mus, *rho_params)

    def _comoving_distance(self, z):
        return self.astropy_cosmology.comoving_distance(z).to(u.Mpc).value

    def _angular_diameter_distance(self, z):
        return self.astropy_cosmology.angular_diameter_distance(z).to(u.Mpc).value

    def angle_scale_distance(self, z):
        if self.comoving:
            return self._comoving_distance(z)
        else:
            return self._angular_diameter_distance(z)

    def convergence(self, thetas, zs, mus, *rho_params):
        radii_of_z = [thetas * self.angle_scale_distance(z) for z in zs]
        convergences = np.array([
            self._radius_space_convergence(rs, zs[i:i+1], mus, *rho_params)
            for i, rs in enumerate(radii_of_z)
        ]).squeeze()
        return convergences.reshape(thetas.shape + zs.shape + (-1,))


@dataclass
class SingleMassShearModel:
    redshifts: np.ndarray
    rho_func: object
    units: u.Quantity = u.Msun/u.pc**2
    shear_class: object = _core.Shear
    esd_func: object = projector.esd

    def __post_init__(self):
        self._shear = self.shear_class(
            rho_func=self.rho_func,
            units=self.units,
            esd_func=self.esd_func,
        )

    def excess_surface_density(self, rs, mus, *rho_params):
        return self._shear.excess_surface_density(rs, self.redshifts, mus, *rho_params)


class SingleMassNfwShearModel:
    def __init__(
            self,
            redshifts,
            units=u.Msun/u.pc**2,
            comoving_radii=True,
            delta=200,
            mass_definition='mean',
            cosmo_params=maszcal.defaults.DefaultCosmology(),
    ):

        self.redshifts = redshifts
        self.units = units
        self.comoving_radii = comoving_radii
        self.delta = delta
        self.mass_definition = mass_definition

        if isinstance(cosmo_params, maszcal.defaults.DefaultCosmology):
            self.cosmo_params = CosmoParams()
        else:
            self.cosmo_params = cosmo_params

        self.astropy_cosmology = get_astropy_cosmology(self.cosmo_params)

    def _init_nfw(self):
        self.nfw_model = maszcal.density.SingleMassNfwModel(
            cosmo_params=self.cosmo_params,
            units=self.units,
            delta=self.delta,
            mass_definition=self.mass_definition,
            comoving=self.comoving_radii,
        )

    def mass(self, mu):
        return np.exp(mu)

    def excess_surface_density(self, rs, mus, concentrations):
        masses = self.mass(mus)

        try:
            return self.nfw_model.excess_surface_density(rs, self.redshifts, masses, concentrations)
        except AttributeError:
            self._init_nfw()
            return self.nfw_model.excess_surface_density(rs, self.redshifts, masses, concentrations)

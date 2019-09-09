import numpy as np
import astropy.units as u
from maszcal.defaults import DefaultCosmology
from maszcal.cosmology import CosmoParams
from maszcal.cosmo_utils import get_astropy_cosmology


class NfwModel:
    """
    SHAPE mass, z, r, cons
    """
    def __init__(
            self,
            cosmo_params=DefaultCosmology(),
            units=u.Msun/u.pc**2,
            delta=200,
            mass_definition='mean',
            comoving=True,
    ):
        self._delta = delta
        self._check_mass_def(mass_definition)
        self.mass_definition = mass_definition
        self.comoving = comoving

        if isinstance(cosmo_params, DefaultCosmology):
            self.cosmo_params = CosmoParams()
        else:
            self.cosmo_params = cosmo_params

        self._astropy_cosmology = get_astropy_cosmology(self.cosmo_params)

        self.units = units

    def _check_mass_def(self, mass_def):
        if mass_def not in ['mean', 'crit']:
            raise ValueError('Mass definition must be \'crit\' or \'mean\'')

    @property
    def mass_definition(self):
        return self._mass_definition

    @mass_definition.setter
    def mass_definition(self, new_mass_def):
        self._check_mass_def(new_mass_def)
        self._mass_definition = new_mass_def

    def _reference_density_comoving(self, zs):
        if self.mass_definition == 'mean':
            rho_mass_def = self._astropy_cosmology.critical_density0 * self._astropy_cosmology.Om0 * np.ones(zs.shape)
        elif self.mass_definition == 'crit':
            rho_mass_def = self._astropy_cosmology.critical_density0 * np.ones(zs.shape)

        return rho_mass_def

    def _reference_density_nocomoving(self, zs):
        if self.mass_definition == 'mean':
            rho_mass_def = self._astropy_cosmology.critical_density0 * self._astropy_cosmology.Om(zs)
        elif self.mass_definition == 'crit':
            rho_mass_def = self._astropy_cosmology.critical_density(zs)

        return rho_mass_def

    def reference_density(self, zs):
        """
        SHAPE z
        """

        if self.comoving:
            rho_mass_def = self._reference_density_comoving(zs)
        else:
            rho_mass_def = self._reference_density_nocomoving(zs)

        return rho_mass_def.to(u.Msun/u.Mpc**3).value

    def radius_delta(self, zs, masses):
        """
        SHAPE mass, z
        """
        pref = 3 / (4*np.pi)
        return (pref * masses[:, None] / (self._delta*self.reference_density(zs))[None, :])**(1/3)

    def scale_radius(self, zs, masses, cons):
        """
        SHAPE mass, z, cons
        """
        return self.radius_delta(zs, masses)[:, :, None]/cons[None, None, :]

    def delta_c(self, cons):
        """
        SHAPE cons
        """
        return (self._delta * cons**3)/(3 * (np.log(1+cons) - cons/(1+cons)))

    def _less_than_func(self, x):
        return (
            8 * np.arctanh(np.sqrt((1-x) / (1+x))) / (x**2 * np.sqrt(1 - x**2))
            + 4 * np.log(x/2) / x**2
            - 2 / (x**2 - 1)
            + 4 * np.arctanh(np.sqrt((1-x) / (1+x))) / ((x**2 - 1) * np.sqrt(1 - x**2))
        )

    def _equal_func(self, x):
        return 10/3 + 4*np.log(1/2)

    def _greater_than_func(self, x):
        return (
            8 * np.arctan(np.sqrt((x-1) / (x+1))) / (x**2 * np.sqrt(x**2 - 1))
            + 4 * np.log(x/2) / x**2
            - 2 / (x**2 - 1)
            + 4 * np.arctan(np.sqrt((x-1) / (x+1))) / ((x**2 - 1)**(3/2))
        )

    def _inequality_func(self, xs):
        full_func_vals = np.zeros(xs.shape)

        full_func_vals[xs < 1] = self._less_than_func(xs[xs < 1])
        full_func_vals[xs == 1] = self._equal_func(xs[xs == 1])
        full_func_vals[xs > 1] = self._greater_than_func(xs[xs > 1])

        return full_func_vals

    def delta_sigma(self, rs, zs, masses, cons):
        """
        SHAPE mass, z, r, cons
        """
        scale_radii = self.scale_radius(zs, masses, cons)
        prefactor = scale_radii * self.delta_c(cons)[None, None, :] * self.reference_density(zs)[None, :, None]
        prefactor = prefactor * (u.Msun/u.Mpc**2).to(self.units)

        xs = rs[None, None, :, None]/scale_radii[:, :, None, :]

        postfactor = self._inequality_func(xs)

        return prefactor[:, :, None, :] * postfactor

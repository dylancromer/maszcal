from dataclasses import dataclass
import numpy as np
import scipy.interpolate
import colossus.cosmology.cosmology as colossus_cosmo
from colossus.halo.mass_adv import changeMassDefinitionCModel as _change_mass_def_cmodel
from colossus.halo.mass_defs import changeMassDefinition as _change_mass_def
from maszcal.cosmology import CosmoParams
from maszcal.cosmo_utils import get_colossus_params
import maszcal.defaults as defaults


@dataclass
class ConInterpolator:
    '''
    Interpolates the concentration-mass relation.

    Used for quick calculations of concentration in c-m models.

    Parameters
    ----------
    mass_samples : ndarray
        Array of masses over which to interpolate.
    redshift_samples : ndarray
        Array of redshifts over which to interpolate.
    mass_definition : str
        String identifying which mass definition to use. Must be in the form `"<int>m"` or `"<int>c"`,
        e.g. `"500c"` for a 500-critical mass definition.
    cosmo_params : CosmoParams
        Cosmology parameters to be used for the c-m relation.
    '''
    mass_samples: np.ndarray
    redshift_samples: np.ndarray
    mass_definition: str
    cosmo_params: object

    def __post_init__(self):
        c_samples = ConModel(mass_def=self.mass_definition, cosmology=self.cosmo_params).c(
            self.mass_samples,
            self.redshift_samples,
            self.mass_definition,
        )

        self._interpolator = scipy.interpolate.interp2d(
            self.redshift_samples,
            self.mass_samples,
            c_samples,
            kind='cubic',
        )

    def __call__(self, masses, redshifts):
        return self._interpolator(masses, redshifts).T


class ConModel:
    def _init_colossus(self, cosmology):
        params = get_colossus_params(cosmology)
        colossus_cosmo.setCosmology('mycosmo', params)

    def __init__(self, mass_def, cosmology=defaults.DefaultCosmology()):
        self.mass_def = mass_def

        if isinstance(cosmology, defaults.DefaultCosmology):
            cosmology_ = CosmoParams()
        else:
            cosmology_ = cosmology

        self._init_colossus(cosmology_)

    def c(self, masses, redshifts, output_mass_def):
        _, cons = self.get_masses_and_cons(masses, redshifts, self.mass_def, output_mass_def)
        return cons

    def convert_mass_def(self, masses, redshifts, old_def, new_def):
        new_masses, _ = self.get_masses_and_cons(masses, redshifts, old_def, new_def)

        return new_masses

    def get_masses_and_cons(self, masses, redshifts, old_def, new_def):
        MODEL = 'diemer19'

        converted_ms = np.empty((masses.size, redshifts.size))
        cons = np.empty((masses.size, redshifts.size))
        for i, z in enumerate(redshifts):
            converted_ms[:, i], _, cons[:, i] = _change_mass_def_cmodel(masses, z, old_def, new_def, c_model=MODEL)

        return converted_ms, cons

    def convert_c_mass_pair(self, masses, concentrations, redshifts, old_def, new_def):
        converted_ms = np.empty((masses.size, redshifts.size))
        converted_cons = np.empty((concentrations.size, redshifts.size))
        for i, z in enumerate(redshifts):
            converted_ms[:, i], _, converted_cons[:, i] = _change_mass_def(masses, concentrations, z, old_def, new_def)

        return converted_ms, converted_cons


class MatchingConModel(ConModel):
    def _check_masses_and_redshifts_match(self, masses, redshifts):
        if masses.shape != redshifts.shape:
            raise ValueError('Masses and redshifts must have same shape')

    def get_masses_and_cons(self, masses, redshifts, old_def, new_def):
        MODEL = 'diemer19'

        self._check_masses_and_redshifts_match(masses, redshifts)

        converted_ms = np.empty((masses.size))
        cons = np.empty((masses.size))
        for i, z in enumerate(redshifts):
            converted_ms[i], _, cons[i] = _change_mass_def_cmodel(masses[i], z, old_def, new_def, c_model=MODEL)

        return converted_ms, cons

    def convert_c_mass_pair(self, masses, concentrations, redshifts, old_def, new_def):
        self._check_masses_and_redshifts_match(masses, redshifts)

        converted_ms = np.empty((masses.size))
        converted_cons = np.empty((concentrations.size))
        for i, z in enumerate(redshifts):
            converted_ms[i], _, converted_cons[i] = _change_mass_def(masses[i], concentrations, z, old_def, new_def)

        return converted_ms, converted_cons

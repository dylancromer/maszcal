import numpy as np
import colossus.cosmology.cosmology as colossus_cosmo
from colossus.halo.mass_adv import changeMassDefinitionCModel as _change_mass_def_cmodel
from colossus.halo.mass_defs import changeMassDefinition as _change_mass_def
from maszcal.cosmology import CosmoParams
from maszcal.cosmo_utils import get_colossus_params
import maszcal.defaults as defaults


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


import numpy as np
import colossus.cosmology.cosmology as colossus_cosmo
import colossus.halo.concentration as colossus_con
from colossus.halo.mass_adv import changeMassDefinitionCModel as _change_mass_def
from maszcal.cosmology import CosmoParams
from maszcal.cosmo_utils import get_colossus_params


class DefaultCosmology:
    pass


class ConModel:
    def _init_colossus(self, cosmology):
        params = get_colossus_params(cosmology)
        colossus_cosmo.setCosmology('mycosmo', params)

    def __init__(self, mass_def, cosmology=DefaultCosmology()):
        self.mass_def = mass_def

        if isinstance(cosmology, DefaultCosmology):
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
            converted_ms[:, i], _, cons[:, i] = _change_mass_def(masses, z, old_def, new_def, c_model=MODEL)

        return converted_ms, cons

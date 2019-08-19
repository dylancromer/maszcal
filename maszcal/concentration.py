import numpy as np
import colossus.cosmology.cosmology as colossus_cosmo
import colossus.halo.concentration as colossus_con
from colossus.halo.mass_adv import changeMassDefinitionCModel as _change_mass_def


class ConModel:
    def _init_colossus(self):
        params = {'flat': True, 'H0': 67.2, 'Om0': 0.31, 'Ob0': 0.049, 'sigma8': 0.81, 'ns': 0.95}
        colossus_cosmo.setCosmology('mycosmo', params)

    def __init__(self, mass_def='200m'):
        self.mass_def = mass_def
        self._init_colossus()

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

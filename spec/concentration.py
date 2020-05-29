import pytest
import numpy as np
from maszcal.concentration import ConModel, MatchingConModel
from maszcal.cosmology import CosmoParams


def describe_ConModel():

    def describe_c():

        def it_converts_the_masses_appropriately():
            con_model = ConModel(mass_def='500c')
            masses = np.logspace(14, 16, 3)
            redshifts = np.linspace(0, 2, 4)
            cons = con_model.c(masses, redshifts, '200c')
            assert cons.shape == (3, 4)
            assert np.all(cons > 0)

    def describe_convert_mass_def():

        def it_does_nothing_if_in_def_and_out_def_are_the_same():
            con_model = ConModel(mass_def='500c')
            masses = np.logspace(14, 16, 3)
            redshifts = np.zeros(1)
            new_masses = con_model.convert_mass_def(masses, redshifts, '500c', '500c')
            assert np.allclose(masses, new_masses.flatten())

        def the_conversion_makes_sense():
            con_model = ConModel(mass_def='500m')
            masses = np.logspace(14, 16, 3)
            redshifts = np.zeros(1)
            new_masses = con_model.convert_mass_def(masses, redshifts, '500m', '200m')
            assert np.all(masses < new_masses.flatten())

    def describe_convert_c_mass_pair():

        def it_does_nothing_if_in_def_and_out_def_are_the_same():
            con_model = ConModel(mass_def='500c')
            masses = np.logspace(14, 16, 3)
            cons = np.linspace(2, 4, 3)
            redshifts = np.zeros(1)
            new_masses, new_cons = con_model.convert_c_mass_pair(masses, cons, redshifts, '500c', '500c')
            assert np.allclose(masses, new_masses.flatten())
            assert np.allclose(cons, new_cons.flatten())


def describe_MatchingConModel():

    def describe_c():

        def it_converts_the_masses_appropriately():
            con_model = MatchingConModel(mass_def='500c')
            masses = np.logspace(14, 16, 3)
            redshifts = np.linspace(0, 2, 3)
            cons = con_model.c(masses, redshifts, '200c')
            assert cons.shape == (3,)
            assert np.all(cons > 0)

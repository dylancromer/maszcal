import pytest
import numpy as np
from maszcal.concentration import ConModel
from maszcal.cosmology import CosmoParams


def describe_con_model():

    def describe_init():

        @pytest.fixture
        def custom_cosmo():
            return CosmoParams(hubble_constant=70.0)

        def it_can_accept_a_custom_cosmology(custom_cosmo):
            con_model = ConModel(mass_def='200m', cosmology=custom_cosmo)

    def describe_c():

        def it_calculates_a_concentration_from_a_mass():
            con_model = ConModel(mass_def='200m')

            masses = np.linspace(np.log(1e14), np.log(1e16), 3)

            redshifts = np.linspace(0, 2, 3)

            cons = con_model.c(masses, redshifts, '200c')

            assert np.all(cons > 0)

        def it_converts_the_masses_appropriately():
            con_model = ConModel(mass_def='500c')

            masses = np.linspace(np.log(1e14), np.log(1e16), 3)

            redshifts = np.linspace(0, 2, 4)

            cons = con_model.c(masses, redshifts, '200c')

            assert cons.shape == (3, 4)
            assert np.all(cons > 0)

    def describe_convert_mass_def():

        def it_does_nothing_if_in_def_and_out_def_are_the_same():
            con_model = ConModel(mass_def='500c')

            masses = np.linspace(np.log(1e14), np.log(1e16), 3)

            redshifts = np.zeros(1)

            new_masses = con_model.convert_mass_def(masses, redshifts, '500c', '500c')

            assert np.allclose(masses, new_masses.flatten())

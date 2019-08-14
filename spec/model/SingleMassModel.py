import pytest
import numpy as np
import astropy.units as u
from maszcal.model import SingleMassModel


def describe_single_mass_model():

    def describe_math():

        @pytest.fixture
        def single_mass_model():
            redshift = 0.4
            model = SingleMassModel(redshift=redshift)
            return model

        def it_can_calculate_delta_sigma_of_mass(single_mass_model):
            mu = np.array([np.log(1e15)])
            con = np.array([3])
            rs = np.logspace(-1, 1, 5)

            delta_sigs = single_mass_model.delta_sigma(rs, mu, con)

            assert np.all(delta_sigs > 0)

        def it_can_use_different_units(single_mass_model):
            mu = np.array([np.log(1e15)])
            con = np.array([3])
            rs = np.logspace(-1, 1, 5)

            delta_sigs = single_mass_model.delta_sigma(rs, mu, con, units=u.Msun/u.Mpc**2)/1e12

            assert np.all(rs*delta_sigs < 1e6)

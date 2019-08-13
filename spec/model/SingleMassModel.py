import pytest
import numpy as np
from maszcal.model import SingleMassModel


def describe_single_mass_model():

    def describe_math():

        @pytest.fixture
        def single_mass_model():
            redshift = 0.4
            model = SingleMassModel(redshift=redshift)
            return model

        def it_can_calculate_delta_sigma_of_mass(single_mass_model):
            mass = 1e15
            con = 3
            params = np.array([mass, con])
            rs = np.logspace(-1, 1, 5)

            delta_sigs = single_mass_model.delta_sigma(rs, params)

            assert np.all(delta_sigs > 0)

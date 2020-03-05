import pytest
import numpy as np
import astropy.units as u
from maszcal.lensing.shear import SingleMassNfwShearModel


def describe_single_mass_model():

    def describe_math():

        @pytest.fixture
        def single_mass_model():
            redshift = 0.4*np.ones(1)
            model = SingleMassNfwShearModel(redshift=redshift)
            return model

        def it_can_calculate_delta_sigma_of_mass(single_mass_model):
            mu = np.array([np.log(1e15)])
            con = np.array([3])
            rs = np.logspace(-1, 1, 5)

            delta_sigs = single_mass_model.delta_sigma(rs, mu, con)

            assert np.all(delta_sigs > 0)
            assert delta_sigs.shape == (1, 1, 5, 1)

        def it_can_use_different_units():
            redshift = 0.4*np.ones(1)
            model = SingleMassNfwShearModel(redshift=redshift, units=u.Msun/u.Mpc**2)

            mu = np.array([np.log(1e15)])
            con = np.array([3])
            rs = np.logspace(-1, 1, 5)

            delta_sigs = model.delta_sigma(rs, mu, con)/1e12

            assert np.all(rs*delta_sigs < 1e6)

        def it_can_use_different_mass_definitions():
            mu = np.array([np.log(1e15)])
            con = np.array([3])
            rs = np.logspace(-1, 1, 5)

            redshift = 0.4*np.ones(1)
            delta = 500
            mass_def = 'crit'
            model = SingleMassNfwShearModel(redshift=redshift, delta=delta, mass_definition=mass_def)

            delta_sigs_500c = model.delta_sigma(rs, mu, con)

            delta = 200
            kind = 'mean'
            model = SingleMassNfwShearModel(redshift=redshift, delta=delta, mass_definition=mass_def)

            delta_sigs_200m = model.delta_sigma(rs, mu, con)

            assert np.all(delta_sigs_200m < delta_sigs_500c)

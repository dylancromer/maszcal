import pytest
import numpy as np
import astropy.units as u
from maszcal.lensing.shear import SingleMassBaryonShearModel


def describe_single_mass_model():

    def describe_math():

        @pytest.fixture
        def single_mass_model():
            redshifts = 0.4*np.ones(2)
            model = SingleMassBaryonShearModel(redshifts=redshifts)
            return model

        def it_can_calculate_delta_sigma_of_mass(single_mass_model):
            base = np.ones(11)

            mus = np.log(1e15)*base
            cons = 3*base
            alphas = 0.88*base
            betas = 3.8*base
            gammas = 0.2*base
            rs = np.logspace(-1, 1, 5)

            delta_sigs = single_mass_model.delta_sigma(rs, mus, cons, alphas, betas, gammas)

            assert np.all(delta_sigs > 0)

        # def it_can_use_different_units():
        #     redshifts = 0.4*np.ones(1)
        #     model = SingleMassBaryonShearModel(redshifts=redshifts, units=u.Msun/u.Mpc**2)

        #     mu = np.array([np.log(1e15)])
        #     con = np.array([3])
        #     alpha = np.array([0.88])
        #     beta = np.array([3.8])
        #     gamma = np.array([0.2])
        #     rs = np.logspace(-1, 1, 5)

        #     delta_sigs = model.delta_sigma(rs, mu, con, alpha, beta, gamma)/1e12

        #     assert np.all(rs*delta_sigs < 1e6)

        # def it_can_use_different_mass_definitions():
        #     mu = np.array([np.log(1e15)])
        #     con = np.array([3])
        #     alpha = np.array([0.88])
        #     beta = np.array([3.8])
        #     gamma = np.array([0.2])
        #     rs = np.logspace(-1, 1, 5)

        #     redshifts = 0.4*np.ones(1)
        #     delta = 500
        #     mass_def = 'crit'
        #     model = SingleMassBaryonShearModel(redshifts=redshifts, delta=delta, mass_definition=mass_def)

        #     delta_sigs_500c = model.delta_sigma(rs, mu, con, alpha, beta, gamma)

        #     delta = 200
        #     kind = 'mean'
        #     model = SingleMassBaryonShearModel(redshifts=redshifts, delta=delta, mass_definition=mass_def)

        #     delta_sigs_200m = model.delta_sigma(rs, mu, con, alpha, beta, gamma)

        #     assert np.all(delta_sigs_200m < delta_sigs_500c)

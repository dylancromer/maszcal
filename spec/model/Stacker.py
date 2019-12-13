import os
import json
import pytest
import numpy as np
import astropy.units as u
from maszcal.model import Stacker
from maszcal.ioutils import NumpyEncoder


def describe_stacker():

    def describe_math():

        @pytest.fixture
        def stacker():
            mus = np.linspace(np.log(1e14), np.log(1e15), 10)
            redshifts = np.linspace(0, 1, 8)
            return Stacker(
                mus,
                redshifts,
                units=u.Msun/u.pc**2,
                delta=200,
                sz_scatter=0.2,
                comoving=True,
                mass_definition='mean',
            )

        def prob_musz_given_mu_is_not_negative(stacker):
            mu_szs = np.linspace(np.log(1e12), np.log(1e16), 10)
            mus = np.linspace(np.log(1e12), np.log(1e16), 20)

            a_szs = np.linspace(-1, 1, 5)

            prob_sz = stacker.prob_musz_given_mu(mu_szs, mus, a_szs)
            assert np.all(prob_sz >= 0)

        def prob_musz_given_mu_integrates_to_1(stacker):
            mu_szs = np.linspace(np.log(1e11), np.log(1e17), 100)
            mus = np.array([np.log(1e15)])

            a_szs = np.linspace(-1, 1, 5)

            prob_sz = stacker.prob_musz_given_mu(mu_szs, mus, a_szs)
            integ = np.trapz(prob_sz, x=mu_szs, axis=0)
            assert np.allclose(integ, 1)

        def delta_sigma_of_r_divided_by_nsz_always_one(stacker):
            """
            This test functions by setting delta_sigma_of_mass to be constant,
            resulting in it being identical to the normalization. Thus this test should
            always return 1s, rather than a true precomputed value
            """
            zs = np.linspace(0, 2, 8)

            stacker.dnumber_dlogmass = lambda : np.ones(
                (stacker.mus.size, stacker.zs.size)
            )

            rs = np.logspace(-1, 1, 21)
            cons = np.linspace(2, 4, 1)
            a_szs = np.linspace(-1, 1, 1)

            delta_sigmas_of_mass = np.ones((stacker.mus.size, zs.size, rs.size, cons.size))

            delta_sigmas = stacker.stacked_delta_sigma(delta_sigmas_of_mass, rs, a_szs)

            precomp_delta_sigmas = np.ones((rs.size, 1))

            np.testing.assert_allclose(delta_sigmas, precomp_delta_sigmas)

        def it_can_handle_delta_sigmas_of_mass_with_different_params(stacker):
            N_PARAMS = 3
            rs = np.logspace(-1, 1, 21)
            a_szs = np.linspace(-1, 1, N_PARAMS)

            stacker.dnumber_dlogmass = lambda : np.ones(
                (stacker.mus.size, stacker.zs.size)
            )

            delta_sigmas_of_mass = np.ones((stacker.mus.size, stacker.zs.size, rs.size, N_PARAMS))

            delta_sigmas = stacker.stacked_delta_sigma(delta_sigmas_of_mass, rs, a_szs)

        def it_complains_about_nans(stacker):
            zs = np.linspace(0, 2, 8)
            stacker.dnumber_dlogmass = lambda : np.full(
                (stacker.mus.size, stacker.zs.size),
                np.nan,
            )

            rs = np.logspace(-1, 1, 10)
            cons = np.linspace(2, 4, 1)
            a_szs = np.linspace(-1, 1, 1)
            delta_sigmas_of_mass = np.ones((stacker.mus.size, zs.size, rs.size, cons.size))

            with pytest.raises(ValueError):
                stacker.stacked_delta_sigma(delta_sigmas_of_mass, rs, a_szs)

            with pytest.raises(ValueError):
                stacker.weak_lensing_avg_mass(a_szs)

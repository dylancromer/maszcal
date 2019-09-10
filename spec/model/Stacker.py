import os
import json
import pytest
import numpy as np
import astropy.units as u
from maszcal.model import Stacker
from maszcal.ioutils import NumpyEncoder


def describe_stacker():

    def describe_data_handling():

        def it_can_load_a_selection_function():
            mus = np.linspace(1, 3, 10)
            zs = np.linspace(0, 2, 5)

            selection_function = lambda m,z: m[:, None]*z[None, :]
            sel_funcs = selection_function(mus, zs)
            sel_func_dict = {'zs':zs,
                             'mus':mus,
                             'selection_fs':sel_funcs}

            SAVED_SELFUNC = 'data/test/test_sel_func.json'
            with open(SAVED_SELFUNC, 'w') as outfile:
                json.dump(sel_func_dict, outfile, cls=NumpyEncoder, ensure_ascii=False)

            stacker = Stacker(mus, zs, selection_func_file=SAVED_SELFUNC, delta=200, units=u.Msun/u.pc**2)

            assert np.allclose(stacker.selection_func(mus, zs), sel_funcs)

            os.remove(SAVED_SELFUNC)

        def it_can_load_lensing_weights():
            mus = np.linspace(1, 3, 10)
            zs = np.linspace(0.1, 2, 5)

            weights = 1/zs**2

            weight_dict = {'zs':zs,
                           'weights':weights}

            SAVED_WEIGHTS = 'data/test/test_lensing_weights.json'
            with open(SAVED_WEIGHTS, 'w') as outfile:
                json.dump(weight_dict, outfile, cls=NumpyEncoder, ensure_ascii=False)

            stacker = Stacker(mus, zs, lensing_weights_file=SAVED_WEIGHTS, delta=200, units=u.Msun/u.pc**2)

            os.remove(SAVED_WEIGHTS)

            assert np.allclose(stacker.lensing_weights(zs), weights)


    def describe_math():

        @pytest.fixture
        def stacker():
            mus = np.linspace(np.log(1e14), np.log(1e15), 10)
            redshifts = np.linspace(0, 1, 8)
            return Stacker(mus, redshifts, units=u.Msun/u.pc**2, delta=200)

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

            delta_sigmas = stacker.delta_sigma(delta_sigmas_of_mass, rs, a_szs)

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

            delta_sigmas = stacker.delta_sigma(delta_sigmas_of_mass, rs, a_szs)

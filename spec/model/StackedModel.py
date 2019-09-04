import os
import json
import pytest
import numpy as np
from maszcal.model import StackedModel
from maszcal.ioutils import NumpyEncoder


def describe_stacked_model():

    def describe_init():

        def it_requires_you_to_provide_mass_and_redshift():
            with pytest.raises(TypeError):
                StackedModel()

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

            stacked_model = StackedModel(mus, zs, selection_func_file=SAVED_SELFUNC)

            assert np.allclose(stacked_model.selection_func(mus, zs), sel_funcs)

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

            stacked_model = StackedModel(mus, zs, lensing_weights_file=SAVED_WEIGHTS)

            os.remove(SAVED_WEIGHTS)

            assert np.allclose(stacked_model.lensing_weights(zs), weights)

    def describe_math_functions():

        @pytest.fixture
        def stacked_model():
            mus = np.linspace(np.log(1e12), np.log(1e16), 20)
            zs = np.linspace(0, 2, 8)

            model = StackedModel(mus, zs)

            return model

        def prob_musz_given_mu_is_not_negative(stacked_model):
            mu_szs = np.linspace(np.log(1e12), np.log(1e16), 10)
            mus = np.linspace(np.log(1e12), np.log(1e16), 20)

            a_szs = np.linspace(-1, 1, 5)

            prob_sz = stacked_model.prob_musz_given_mu(mu_szs, mus, a_szs)
            assert np.all(prob_sz >= 0)

        def prob_musz_given_mu_integrates_to_1(stacked_model):
            mu_szs = np.linspace(np.log(1e11), np.log(1e17), 100)
            mus = np.array([np.log(1e15)])

            a_szs = np.linspace(-1, 1, 5)

            prob_sz = stacked_model.prob_musz_given_mu(mu_szs, mus, a_szs)
            integ = np.trapz(prob_sz, x=mu_szs, axis=0)
            assert np.allclose(integ, 1)

        def delta_sigma_of_r_divided_by_nsz_always_one(stacked_model):
            """
            This test functions by setting delta_sigma_of_mass to be constant,
            resulting in it being identical to the normalization. Thus this test should
            always return 1s, rather than a true precomputed value
            """
            zs = np.linspace(0, 2, 8)

            stacked_model.dnumber_dlogmass = lambda : np.ones(
                (stacked_model.mus.size, stacked_model.zs.size)
            )

            rs = np.logspace(-1, 1, 21)

            stacked_model.delta_sigma_of_mass = lambda rs, mus, cons, units: np.ones(
                (stacked_model.mus.size, zs.size, rs.size, cons.size)
            )

            cons = np.linspace(2, 4, 1)
            a_szs = np.linspace(-1, 1, 1)

            delta_sigmas = stacked_model.delta_sigma(rs, cons, a_szs)

            precomp_delta_sigmas = np.ones((rs.size, 1))

            np.testing.assert_allclose(delta_sigmas, precomp_delta_sigmas)


        def it_computes_weak_lensing_avg_mass(stacked_model):

            stacked_model.dnumber_dlogmass = lambda : np.ones(
                (stacked_model.mus.size, stacked_model.zs.size)
            )

            stacked_model.delta_sigma_of_mass = lambda rs,mus,cons,units: np.ones(
                (stacked_model.mus.size, rs.size)
            )

            a_szs = np.linspace(-1, 1, 1)

            avg_wl_mass = stacked_model.weak_lensing_avg_mass(a_szs)

            assert avg_wl_mass.shape == (1,)

        def it_can_use_different_mass_definitions():
            mu = np.array([np.log(1e15)])
            con = np.array([3])
            rs = np.logspace(-1, 1, 10)

            mus = np.linspace(np.log(1e12), np.log(1e16), 20)
            zs = np.linspace(0, 2, 8)

            delta = 500
            mass_def = 'crit'
            model = StackedModel(mus, zs, delta=delta, mass_definition=mass_def)

            delta_sigs_500c = model.delta_sigma_of_mass(rs, mu, con)

            delta = 200
            kind = 'mean'
            model = StackedModel(mus, zs, delta=delta, mass_definition=mass_def)

            delta_sigs_200m = model.delta_sigma_of_mass(rs, mu, con)

            assert np.all(delta_sigs_200m < delta_sigs_500c)

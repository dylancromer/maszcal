import os
import json
import pytest
import numpy as np
from maszcal.model import StackedModel
from maszcal.ioutils import NumpyEncoder




def describe_stacked_model():

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

            stacked_model = StackedModel(selection_func_file=SAVED_SELFUNC)

            assert np.allclose(stacked_model.selection_func(mus, zs), sel_funcs)

            os.remove(SAVED_SELFUNC)

        def it_can_load_lensing_weights():
            zs = np.linspace(0.1, 2, 5)
            weights = 1/zs**2

            weight_dict = {'zs':zs,
                           'weights':weights}

            SAVED_WEIGHTS = 'data/test/test_lensing_weights.json'
            with open(SAVED_WEIGHTS, 'w') as outfile:
                json.dump(weight_dict, outfile, cls=NumpyEncoder, ensure_ascii=False)

            stacked_model = StackedModel(lensing_weights_file=SAVED_WEIGHTS)

            os.remove(SAVED_WEIGHTS)

            assert np.allclose(stacked_model.lensing_weights(zs), weights)

    def describe_math_functions():

        @pytest.fixture
        def stacked_model():
            params = 2*np.ones((1,2))
            model = StackedModel(params=params)

            model.mu_szs = np.linspace(12, 16, 10)
            model.mus = np.linspace(12, 16, 20)
            model.zs = np.linspace(0, 2, 8)

            rs = np.logspace(-1, 1, 40)
            return model

        def prob_musz_given_mu_is_not_negative(stacked_model):
            mu_szs = np.linspace(12, 16, 10)
            mus = np.linspace(12, 16, 20)

            prob_sz = stacked_model.prob_musz_given_mu(mu_szs, mus)
            assert np.all(prob_sz > 0)


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

            rs = np.logspace(-1, 1, 40)

            stacked_model.delta_sigma_of_mass = lambda rs,mus,cons,units,miscentered: np.ones(
                (stacked_model.mus.size, zs.size, rs.size, stacked_model.concentrations.size)
            )

            delta_sigmas = stacked_model.delta_sigma(rs)

            precomp_delta_sigmas = np.ones((rs.size, 1))

            np.testing.assert_allclose(delta_sigmas, precomp_delta_sigmas)


        def it_computes_weak_lensing_avg_mass(stacked_model):

            stacked_model.dnumber_dlogmass = lambda : np.ones(
                (stacked_model.mus.size, stacked_model.zs.size)
            )

            stacked_model.delta_sigma_of_mass = lambda rs,mus,cons,units,miscentered: np.ones(
                (stacked_model.mus.size, rs.size)
            )

            avg_wl_mass = stacked_model.weak_lensing_avg_mass()

            assert avg_wl_mass.shape == (1,)


        def it_computes_miscentered_delta_sigma(stacked_model):
            zs = stacked_model.zs
            mus = np.array([15])
            stacked_model.mus = mus
            rs = np.logspace(-1, 1, 21)

            params = np.array([[2, 3, 0.5, 1e-2],
                               [2, 3, 0.7, 1e-1]])

            cons = params[:, 1]
            frac = params[:, 2]
            r_misc = params[:, 3]

            stacked_model.params = params

            stacked_model.sigma_of_mass = lambda rs,mus,cons,units: np.ones((mus.size, zs.size, rs.size, cons.size))

            miscentered_sigmas = stacked_model.misc_sigma(rs, mus, cons, frac, r_misc)

            assert miscentered_sigmas.shape == (1, 8, 21, 2)

        def it_computes_stacked_miscentered_delta_sigma(stacked_model):
            zs = stacked_model.zs
            mus = stacked_model.mus
            rs = np.logspace(-1, 1, 21)

            params = np.array([[2, 3, 0.5, 1e-2],
                               [2, 3, 0.7, 1e-1],
                               [2, 3, 0.8, 2e-2]])

            cons = params[:, 1]
            frac = params[:, 2]
            r_misc = params[:, 3]

            stacked_model.params = params

            stacked_model.sigma_of_mass = lambda rs,mus,cons,units: np.ones((mus.size, zs.size, rs.size, cons.size))
            delta_sigmas = stacked_model.delta_sigma(rs, miscentered=True)

            assert delta_sigmas.shape == (21, 3)

        def it_computes_stacked_delta_sigma(stacked_model):
            zs = stacked_model.zs
            mus = stacked_model.mus
            rs = np.logspace(-1, 1, 21)

            params = np.array([[2, 3.0],
                               [2, 3.1],
                               [2, 3.2]])

            cons = params[:, 1]

            stacked_model.params = params

            stacked_model.sigma_of_mass = lambda rs,mus,cons,units: np.ones((mus.size, zs.size, rs.size, cons.size))
            delta_sigmas = stacked_model.delta_sigma(rs, miscentered=False)

            assert delta_sigmas.shape == (21, 3)

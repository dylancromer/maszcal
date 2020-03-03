import pytest
import numpy as np
from maszcal.lensing.shear import NfwShearModel


def describe_stacked_model():

    def describe_math_functions():

        @pytest.fixture
        def stacked_model():
            mus = np.linspace(np.log(1e12), np.log(1e16), 20)
            zs = np.linspace(0, 2, 8)

            model = NfwShearModel(mus, zs)

            return model

        def it_computes_weak_lensing_avg_mass():
            mus = np.linspace(np.log(1e12), np.log(1e16), 20)
            zs = np.linspace(0, 2, 8)
            stacked_model = NfwShearModel(mus, zs)
            stacked_model._init_stacker()

            stacked_model.stacker.dnumber_dlogmass = lambda : np.ones(
                (stacked_model.mus.size, stacked_model.zs.size)
            )

            a_szs = np.linspace(-1, 1, 1)

            avg_wl_mass = stacked_model.weak_lensing_avg_mass(a_szs)

            assert avg_wl_mass.shape == (1,)

        def it_can_use_different_mass_definitions():
            cons = np.array([2, 3, 4])
            rs = np.logspace(-1, 1, 10)

            mus = np.linspace(np.log(1e12), np.log(1e15), 20)
            zs = np.linspace(0, 2, 7)

            delta = 500
            mass_def = 'crit'
            model = NfwShearModel(mus, zs, delta=delta, mass_definition=mass_def)

            delta_sigs_500c = model.delta_sigma(rs, mus, cons)

            delta = 200
            kind = 'mean'
            model = NfwShearModel(mus, zs, delta=delta, mass_definition=mass_def)

            delta_sigs_200m = model.delta_sigma(rs, mus, cons)

            assert np.all(delta_sigs_200m < delta_sigs_500c)

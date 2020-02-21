import pytest
import numpy as np
from maszcal.lensing.shear import MiyatakeShearModel


class FakeConModel:
    def __init__(self, mass_def, cosmology=None):
        pass

    def c(self, masses, redshifts, mass_def):
        return np.ones((masses.size, redshifts.size))

    def convert_mass_def(self, masses, redshifts, old_def, new_def):
        return np.ones((masses.size, redshifts.size))


def describe_stacked_model():

    def describe_init():

        def it_requires_you_to_provide_mass_and_redshift():
            with pytest.raises(TypeError):
                MiyatakeShearModel()

    def describe_math_functions():

        @pytest.fixture
        def stacked_model(mocker):
            mus = np.linspace(np.log(1e12), np.log(1e16), 20)
            zs = np.linspace(0, 2, 8)
            mocker.patch('maszcal.lensing.shear.ConModel', new=FakeConModel)

            model = MiyatakeShearModel(mus, zs)
            model._init_stacker()

            return model

        def it_computes_weak_lensing_avg_mass(stacked_model):
            stacked_model.stacker.dnumber_dlogmass = lambda : np.ones(
                (stacked_model.mus.size, stacked_model.zs.size)
            )

            a_szs = np.linspace(-1, 1, 1)

            avg_wl_mass = stacked_model.weak_lensing_avg_mass(a_szs)

            assert avg_wl_mass.shape == (1,)
            assert avg_wl_mass > 0

        def it_can_use_different_mass_definitions(mocker):
            mocker.patch('maszcal.lensing.shear.ConModel', new=FakeConModel)
            rs = np.logspace(-1, 1, 10)

            mus = np.linspace(np.log(1e12), np.log(1e15), 20)
            zs = np.linspace(0, 2, 7)

            delta = 500
            mass_def = 'crit'
            model = MiyatakeShearModel(mus, zs, delta=delta, mass_definition=mass_def)

            delta_sigs_500c = model.delta_sigma(rs, mus)

            delta = 200
            kind = 'mean'
            model = MiyatakeShearModel(mus, zs, delta=delta, mass_definition=mass_def)

            delta_sigs_200m = model.delta_sigma(rs, mus)

            assert np.all(delta_sigs_200m < delta_sigs_500c)

        def it_computes_stacked_delta_sigmas_with_the_right_shape(stacked_model):
            stacked_model.stacker.dnumber_dlogmass = lambda : np.ones(
                (stacked_model.mus.size, stacked_model.zs.size)
            )

            a_szs = np.linspace(-1, 1, 1)

            rs = np.logspace(-1, 1, 4)

            delta_sigs_stacked = stacked_model.stacked_delta_sigma(rs, a_szs)

            assert delta_sigs_stacked.shape == (4, 1)

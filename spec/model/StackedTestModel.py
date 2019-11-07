import pytest
import numpy as np
from maszcal.model import StackedTestModel


class FakeConModel:
    def __init__(self, mass_def, cosmology=None):
        pass

    def c(self, masses, redshifts, mass_def):
        return np.ones((masses.size, redshifts.size))


def describe_stacked_model():

    def describe_init():

        def it_requires_you_to_provide_mass_and_redshift():
            with pytest.raises(TypeError):
                StackedTestModel()

    def describe_math_functions():

        @pytest.fixture
        def stacked_model(mocker):
            mus = np.linspace(np.log(1e12), np.log(1e16), 20)
            zs = np.linspace(0, 2, 8)
            mocker.patch('maszcal.model.ConModel', new=FakeConModel)

            model = StackedTestModel(mus, zs)

            return model

        def it_computes_weak_lensing_avg_mass(mocker):
            mus = np.linspace(np.log(1e12), np.log(1e16), 20)
            zs = np.linspace(0, 2, 8)
            stacked_model = StackedTestModel(mus, zs)
            stacked_model._init_stacker()
            mocker.patch('maszcal.model.ConModel', new=FakeConModel)

            stacked_model.stacker.dnumber_dlogmass = lambda : np.ones(
                (stacked_model.mus.size, stacked_model.zs.size)
            )

            a_szs = np.linspace(-1, 1, 1)

            avg_wl_mass = stacked_model.weak_lensing_avg_mass(a_szs)

            assert avg_wl_mass.shape == (1,)

        def it_can_use_different_mass_definitions(mocker):
            mocker.patch('maszcal.model.ConModel', new=FakeConModel)
            rs = np.logspace(-1, 1, 10)

            mus = np.linspace(np.log(1e12), np.log(1e15), 20)
            zs = np.linspace(0, 2, 7)

            delta = 500
            mass_def = 'crit'
            model = StackedTestModel(mus, zs, delta=delta, mass_definition=mass_def)

            delta_sigs_500c = model.delta_sigma(rs, mus)

            delta = 200
            kind = 'mean'
            model = StackedTestModel(mus, zs, delta=delta, mass_definition=mass_def)

            delta_sigs_200m = model.delta_sigma(rs, mus)

            assert np.all(delta_sigs_200m < delta_sigs_500c)

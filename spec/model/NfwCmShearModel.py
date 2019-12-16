from dataclasses import dataclass
import pytest
import numpy as np
from maszcal.model import NfwCmShearModel


@dataclass
class FakeConModel:
    mass_def: str
    cosmology: object = 'blah'

    def c(self, masses, zs, mass_def):
        return np.ones((masses.size, zs.size))


def describe_stacked_model():

    def describe_init():

        def it_requires_you_to_provide_mass_and_redshift():
            with pytest.raises(TypeError):
                NfwCmShearModel()

    def describe_math_functions():

        @pytest.fixture
        def stacked_model():
            mus = np.linspace(np.log(1e12), np.log(1e16), 20)
            zs = np.linspace(0, 2, 8)

            model = NfwCmShearModel(mus, zs)

            return model

        def it_computes_stacked_delta_sigma(mocker):
            mocker.patch('maszcal.model.ConModel', new=FakeConModel)

            mus = np.linspace(np.log(1e12), np.log(1e16), 20)
            zs = np.linspace(0, 2, 8)
            stacked_model = NfwCmShearModel(mus, zs)
            stacked_model._init_stacker()

            stacked_model.stacker.dnumber_dlogmass = lambda : np.ones(
                (stacked_model.mus.size, stacked_model.zs.size)
            )

            rs = np.logspace(-1, 1, 4)
            a_szs = np.linspace(-1, 1, 3)

            avg_wl_mass = stacked_model.stacked_delta_sigma(rs, a_szs)

            assert avg_wl_mass.shape == (rs.size, a_szs.size)

        def it_computes_weak_lensing_avg_mass(mocker):
            mocker.patch('maszcal.model.ConModel', new=FakeConModel)

            mus = np.linspace(np.log(1e12), np.log(1e16), 20)
            zs = np.linspace(0, 2, 8)
            stacked_model = NfwCmShearModel(mus, zs)
            stacked_model._init_stacker()

            stacked_model.stacker.dnumber_dlogmass = lambda : np.ones(
                (stacked_model.mus.size, stacked_model.zs.size)
            )

            a_szs = np.linspace(-1, 1, 3)

            avg_wl_mass = stacked_model.weak_lensing_avg_mass(a_szs)

            assert avg_wl_mass.shape == (3,)

        def it_can_use_different_mass_definitions(mocker):
            mocker.patch('maszcal.model.ConModel', new=FakeConModel)

            rs = np.logspace(-1, 1, 10)

            mus = np.linspace(np.log(1e12), np.log(1e15), 20)
            zs = np.linspace(0, 2, 7)

            delta = 500
            mass_def = 'crit'
            model = NfwCmShearModel(mus, zs, delta=delta, mass_definition=mass_def)

            delta_sigs_500c = model.delta_sigma(rs, mus)

            assert delta_sigs_500c.shape == (20, 7, 10)

            delta = 200
            kind = 'mean'
            model = NfwCmShearModel(mus, zs, delta=delta, mass_definition=mass_def)

            delta_sigs_200m = model.delta_sigma(rs, mus)

            assert np.all(delta_sigs_200m < delta_sigs_500c)

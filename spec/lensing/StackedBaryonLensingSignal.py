import numpy as np
import pytest
from maszcal.lensing import StackedBaryonLensingModel
from maszcal.cosmology import CosmoParams
import maszcal.defaults as defaults


class FakeGnfwBaryonModel:
    def __init__(
            self,
            mu_bins,
            redshift_bins,
            units=1,
            cosmo_params=defaults.DefaultCosmology(),
            selection_func_file=defaults.DefaultSelectionFunc(),
            lensing_weights_file=defaults.DefaultLensingWeights(),
            delta=200,
            mass_definition='mean',
    ):
        pass

    def stacked_delta_sigma(self, rs, cons, a_szs):
        return np.ones(12)

def describe_lensing_signal():

    def describe_init():

        def it_requires_redshifts():
            with pytest.raises(TypeError):
                StackedLensingSignal()

        def it_accepts_a_selection_func_file(mocker):
            mus = np.ones(10)
            zs = np.ones(5)
            mocker.patch('maszcal.lensing.model.StackedModel', new=FakeStackedModel)
            sel_func_file = 'test/file/here'
            StackedLensingSignal(mus, zs, selection_func_file=sel_func_file)

        def it_accepts_a_weights_file(mocker):
            mus = np.ones(10)
            zs = np.ones(5)
            mocker.patch('maszcal.lensing.model.StackedModel', new=FakeStackedModel)
            weights_file = 'test/file/here'
            StackedLensingSignal(mus, zs, lensing_weights_file=weights_file)

        def it_allows_a_different_mass_definition(mocker):
            mus = np.ones(10)
            zs = np.ones(5)
            mocker.patch('maszcal.lensing.model.StackedModel', new=FakeStackedModel)

            delta = 500
            mass_definition = 'crit'

            StackedLensingSignal(mus, zs, delta=delta, mass_definition=mass_definition)

        def it_can_use_a_different_cosmology(mocker):
            mus = np.ones(10)
            zs = np.ones(5)
            mocker.patch('maszcal.lensing.model.StackedModel', new=FakeStackedModel)

            cosmo = CosmoParams(neutrino_mass_sum=1)
            StackedLensingSignal(mus, zs, cosmo_params=cosmo)

    def describe_stacked_esd():

        @pytest.fixture
        def lensing_signal(mocker):
            mocker.patch('maszcal.lensing.model.StackedModel', new=FakeStackedModel)

            mus = np.linspace(32, 34, 10)
            zs = np.linspace(0, 2, 5)
            return StackedLensingSignal(log_masses=mus, redshifts=zs)

        def it_requires_masses():
            zs = np.ones(2)
            with pytest.raises(TypeError):
                lensing_signal = StackedLensingSignal(redshifts=zs)

        def it_gives_a_stacked_model_for_the_esd(lensing_signal):
            rs = np.logspace(-1, 1, 10)
            params = np.array([[3.01, 0],
                               [3.02, 0]])

            esd = lensing_signal.stacked_esd(rs, params)

            assert np.all(esd == np.ones(12))

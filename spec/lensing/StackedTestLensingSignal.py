
import numpy as np
import pytest
from maszcal.lensing import StackedTestLensingSignal
from maszcal.cosmology import CosmoParams
import maszcal.defaults as defaults


class FakeStackedTestModel:
    def __init__(
            self,
            mu_bins,
            redshift_bins,
            selection_func_file=defaults.DefaultSelectionFunc(),
            lensing_weights_file=defaults.DefaultLensingWeights(),
            cosmo_params=defaults.DefaultCosmology(),
            units=1,
            comoving_radii=True,
            delta=200,
            mass_definition='mean',

    ):
        pass

    def stacked_delta_sigma(self, rs, a_szs):
        return np.ones(12)

def describe_lensing_signal():

    def describe_init():

        def it_requires_redshifts():
            with pytest.raises(TypeError):
                StackedTestLensingSignal()

        def it_accepts_a_selection_func_file(mocker):
            mus = np.ones(10)
            zs = np.ones(5)
            mocker.patch('maszcal.lensing.model.StackedTestModel', new=FakeStackedTestModel)
            sel_func_file = 'test/file/here'
            StackedTestLensingSignal(mus, zs, selection_func_file=sel_func_file)

        def it_accepts_a_weights_file(mocker):
            mus = np.ones(10)
            zs = np.ones(5)
            mocker.patch('maszcal.lensing.model.StackedTestModel', new=FakeStackedTestModel)
            weights_file = 'test/file/here'
            StackedTestLensingSignal(mus, zs, lensing_weights_file=weights_file)

        def it_allows_a_different_mass_definition(mocker):
            mus = np.ones(10)
            zs = np.ones(5)
            mocker.patch('maszcal.lensing.model.StackedTestModel', new=FakeStackedTestModel)

            delta = 500
            mass_definition = 'crit'

            StackedTestLensingSignal(mus, zs, delta=delta, mass_definition=mass_definition)

        def it_can_use_a_different_cosmology(mocker):
            mus = np.ones(10)
            zs = np.ones(5)
            mocker.patch('maszcal.lensing.model.StackedTestModel', new=FakeStackedTestModel)

            cosmo = CosmoParams(neutrino_mass_sum=1)
            StackedTestLensingSignal(mus, zs, cosmo_params=cosmo)

    def describe_stacked_esd():

        @pytest.fixture
        def lensing_signal(mocker):
            mocker.patch('maszcal.lensing.model.StackedTestModel', new=FakeStackedTestModel)

            mus = np.linspace(32, 34, 10)
            zs = np.linspace(0, 2, 5)
            return StackedTestLensingSignal(log_masses=mus, redshifts=zs)

        def it_requires_masses():
            zs = np.ones(2)
            with pytest.raises(TypeError):
                lensing_signal = StackedTestLensingSignal(redshifts=zs)

        def it_gives_a_stacked_model_for_the_esd(lensing_signal):
            rs = np.logspace(-1, 1, 10)
            params = np.array([[0.01], [0.02]])

            esd = lensing_signal.stacked_esd(rs, params)

            assert np.all(esd == np.ones(12))

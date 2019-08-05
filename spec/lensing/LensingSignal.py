import numpy as np
import pytest
from maszcal.lensing import LensingSignal
import maszcal.defaults as defaults


class FakeStackedModel:
    def __init__(
            self,
            mu_bins,
            redshift_bins,
            selection_func_file=defaults.DefaultSelectionFunc,
            lensing_weights_file=defaults.DefaultLensingWeights,
    ):
        pass

    def delta_sigma(self, rs, miscentered=False, units=1):
        return np.ones(12)


def describe_lensing_signal():

    def describe_init():

        def it_requires_mass_and_redshift():
            with pytest.raises(TypeError):
                LensingSignal()

            mus = np.ones(10)
            zs = np.ones(5)

            LensingSignal(mus, zs)

        def it_accepts_a_selection_func_file(mocker):
            mus = np.ones(10)
            zs = np.ones(5)
            mocker.patch('maszcal.lensing.StackedModel', new=FakeStackedModel)
            sel_func_file = 'test/file/here'
            LensingSignal(mus, zs, selection_func_file=sel_func_file)

        def it_accepts_a_weights_file(mocker):
            mus = np.ones(10)
            zs = np.ones(5)
            mocker.patch('maszcal.lensing.StackedModel', new=FakeStackedModel)
            weights_file = 'test/file/here'
            LensingSignal(mus, zs, lensing_weights_file=weights_file)


    def describe_stacked_esd():

        @pytest.fixture
        def lensing_signal(mocker):
            mocker.patch('maszcal.lensing.StackedModel', new=FakeStackedModel)

            mus = np.ones(10)
            zs = np.ones(5)
            return LensingSignal(mus, zs)

        def it_gives_a_stacked_model_for_the_esd(lensing_signal):
            rs = np.logspace(-1, 1, 10)
            params = np.array([[0, 3.01],
                               [0, 3.02]])

            esd = lensing_signal.stacked_esd(rs, params)

            assert np.all(esd == np.ones(12))

        def it_allows_miscentered_profiles(lensing_signal):
            rs = np.logspace(-1, 1, 10)
            params = np.array([[0, 3.01],
                               [0, 3.02]])

            esd = lensing_signal.stacked_esd(rs, params, miscentered=True)
            assert np.all(esd == np.ones(12))

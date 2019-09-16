import numpy as np
import pytest
from maszcal.lensing import LensingSignal
from maszcal.cosmology import CosmoParams
import maszcal.defaults as defaults


class FakeStackedModel:
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

    def delta_sigma(self, rs, cons, a_szs):
        return np.ones(12)

class FakeSingleMassModel:
    def __init__(
            self,
            redshift,
            comoving_radii=True,
            cosmo_params=defaults.DefaultCosmology(),
            delta=200,
            mass_definition='mean',
    ):
        pass

    def delta_sigma(self, rs, mus, concentrations):
        return np.ones(13)


class FakeGaussianBaryonModel:
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

    def delta_sigma(self, rs, cons, a_szs, ln_bary_vars):
        return np.ones(7)


def describe_lensing_signal():

    def describe_init():

        def it_requires_redshifts():
            with pytest.raises(TypeError):
                LensingSignal()

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

        def it_allows_a_different_mass_definition(mocker):
            mus = np.ones(10)
            zs = np.ones(5)
            mocker.patch('maszcal.lensing.StackedModel', new=FakeStackedModel)

            delta = 500
            mass_definition = 'crit'

            LensingSignal(mus, zs, delta=delta, mass_definition=mass_definition)

        def it_can_use_a_different_cosmology(mocker):
            mus = np.ones(10)
            zs = np.ones(5)
            mocker.patch('maszcal.lensing.StackedModel', new=FakeStackedModel)

            cosmo = CosmoParams(neutrino_mass_sum=1)
            LensingSignal(mus, zs, cosmo_params=cosmo)

    def describe_stacked_esd():

        @pytest.fixture
        def lensing_signal(mocker):
            mocker.patch('maszcal.lensing.StackedModel', new=FakeStackedModel)

            mus = np.linspace(32, 34, 10)
            zs = np.linspace(0, 2, 5)
            return LensingSignal(log_masses=mus, redshifts=zs)

        def it_requires_masses():
            zs = np.ones(2)
            lensing_signal = LensingSignal(redshifts=zs)

            rs = np.logspace(-1, 1, 10)
            params = np.array([[3.01, 0],
                               [3.02, 0]])

            with pytest.raises(TypeError):
                esd = lensing_signal.stacked_esd(rs, params)

        def it_gives_a_stacked_model_for_the_esd(lensing_signal):
            rs = np.logspace(-1, 1, 10)
            params = np.array([[3.01, 0],
                               [3.02, 0]])

            esd = lensing_signal.stacked_esd(rs, params)

            assert np.all(esd == np.ones(12))

    def describe_single_mass_esd():

        def it_fails_if_there_are_too_many_redshifts():
            redshifts = np.ones(10)
            lensing_signal = LensingSignal(redshifts=redshifts)

            rs = np.logspace(-1, 1, 3)
            params = np.array([[1, 1]])

            with pytest.raises(TypeError):
                esd = lensing_signal.single_mass_esd(rs, params)

        @pytest.fixture
        def lensing_signal(mocker):
            mocker.patch('maszcal.lensing.SingleMassModel', new=FakeSingleMassModel)
            redshift = np.array([0])
            return LensingSignal(redshifts=redshift)

        def it_gives_a_single_mass_model_for_the_esd(lensing_signal):
            rs = np.logspace(-1, 1, 3)
            params = np.array([[1, 1]])

            esd = lensing_signal.single_mass_esd(rs, params)
            assert np.all(esd == np.ones(13))

    def describe_gaussian_baryon_esd():

        @pytest.fixture
        def lensing_signal(mocker):
            mocker.patch('maszcal.lensing.BaryonGaussianModel', new=FakeGaussianBaryonModel)
            mus = np.ones(10)
            zs = np.ones(5)
            return LensingSignal(log_masses=mus, redshifts=zs)

        def it_can_compute_a_stacked_esd_with_baryons(lensing_signal):
            pass

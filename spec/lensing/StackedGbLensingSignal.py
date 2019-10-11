
import numpy as np
import pytest
from maszcal.lensing import StackedGbLensingSignal
from maszcal.cosmology import CosmoParams
import maszcal.defaults as defaults


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

    def stacked_delta_sigma(self, rs, cons, a_szs, ln_bary_vars):
        return np.ones(7)


def describe_lensing_signal():

    def describe_init():

        def it_requires_redshifts():
            with pytest.raises(TypeError):
                StackedGbLensingSignal()

        def it_accepts_a_selection_func_file(mocker):
            mus = np.ones(10)
            zs = np.ones(5)
            mocker.patch('maszcal.lensing.model.GaussianBaryonModel', new=FakeGaussianBaryonModel)
            sel_func_file = 'test/file/here'
            StackedGbLensingSignal(mus, zs, selection_func_file=sel_func_file)

        def it_accepts_a_weights_file(mocker):
            mus = np.ones(10)
            zs = np.ones(5)
            mocker.patch('maszcal.lensing.model.GaussianBaryonModel', new=FakeGaussianBaryonModel)
            weights_file = 'test/file/here'
            StackedGbLensingSignal(mus, zs, lensing_weights_file=weights_file)

        def it_allows_a_different_mass_definition(mocker):
            mus = np.ones(10)
            zs = np.ones(5)
            mocker.patch('maszcal.lensing.model.GaussianBaryonModel', new=FakeGaussianBaryonModel)

            delta = 500
            mass_definition = 'crit'

            StackedGbLensingSignal(mus, zs, delta=delta, mass_definition=mass_definition)

        def it_can_use_a_different_cosmology(mocker):
            mus = np.ones(10)
            zs = np.ones(5)
            mocker.patch('maszcal.lensing.model.GaussianBaryonModel', new=FakeGaussianBaryonModel)

            cosmo = CosmoParams(neutrino_mass_sum=1)
            StackedGbLensingSignal(mus, zs, cosmo_params=cosmo)

    def describe_gaussian_baryon_esd():

        @pytest.fixture
        def lensing_signal(mocker):
            mocker.patch('maszcal.lensing.model.GaussianBaryonModel', new=FakeGaussianBaryonModel)
            mus = np.ones(10)
            zs = np.ones(5)
            return StackedGbLensingSignal(log_masses=mus, redshifts=zs)

        def it_can_compute_a_stacked_esd_with_baryons(lensing_signal):
            rs = np.logspace(-1, 1, 10)
            params = np.array([[3.01, 0, 1e-2],
                               [3.02, 0, 2e-2]])

            esd = lensing_signal.esd(rs, params)

            assert np.all(esd == np.ones(7))

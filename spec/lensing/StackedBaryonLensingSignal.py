import numpy as np
import pytest
from maszcal.lensing import StackedBaryonLensingSignal
from maszcal.cosmology import CosmoParams
import maszcal.defaults as defaults


class FakeGnfwBaryonModel:
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
            sz_scatter=0.2,
    ):
        pass

    def stacked_delta_sigma(self, rs, cons, alphas, betas, gammas, a_szs):
        return np.ones(12)

    def weak_lensing_avg_mass(self, a_szs):
        return np.ones(4)

def describe_lensing_signal():

    def describe_init():

        def it_requires_redshifts():
            with pytest.raises(TypeError):
                StackedBaryonLensingSignal()

        def it_accepts_a_selection_func_file(mocker):
            mus = np.ones(10)
            zs = np.ones(5)
            mocker.patch('maszcal.lensing.model.GnfwBaryonModel', new=FakeGnfwBaryonModel)
            sel_func_file = 'test/file/here'
            StackedBaryonLensingSignal(mus, zs, selection_func_file=sel_func_file)

        def it_accepts_a_weights_file(mocker):
            mus = np.ones(10)
            zs = np.ones(5)
            mocker.patch('maszcal.lensing.model.GnfwBaryonModel', new=FakeGnfwBaryonModel)
            weights_file = 'test/file/here'
            StackedBaryonLensingSignal(mus, zs, lensing_weights_file=weights_file)

        def it_allows_a_different_mass_definition(mocker):
            mus = np.ones(10)
            zs = np.ones(5)
            mocker.patch('maszcal.lensing.model.GnfwBaryonModel', new=FakeGnfwBaryonModel)

            delta = 500
            mass_definition = 'crit'

            StackedBaryonLensingSignal(mus, zs, delta=delta, mass_definition=mass_definition)

        def it_can_use_a_different_cosmology(mocker):
            mus = np.ones(10)
            zs = np.ones(5)
            mocker.patch('maszcal.lensing.model.GnfwBaryonModel', new=FakeGnfwBaryonModel)

            cosmo = CosmoParams(neutrino_mass_sum=1)
            StackedBaryonLensingSignal(mus, zs, cosmo_params=cosmo)

    def describe_stacked_esd():

        @pytest.fixture
        def lensing_signal(mocker):
            mocker.patch('maszcal.lensing.model.GnfwBaryonModel', new=FakeGnfwBaryonModel)

            mus = np.linspace(32, 34, 10)
            zs = np.linspace(0, 2, 5)
            return StackedBaryonLensingSignal(log_masses=mus, redshifts=zs)

        def it_requires_masses():
            zs = np.ones(2)
            with pytest.raises(TypeError):
                lensing_signal = StackedBaryonLensingSignal(redshifts=zs)

        def it_gives_a_stacked_model_for_the_esd(lensing_signal):
            rs = np.logspace(-1, 1, 10)
            params = np.array([[3.01, 1, 3, 1, 0],
                               [3.02, 1, 3, 1, 0]])

            esd = lensing_signal.stacked_esd(rs, params)

            assert np.all(esd == np.ones(12))

    def describe_avg_mass():

        @pytest.fixture
        def lensing_signal(mocker):
            mocker.patch('maszcal.lensing.model.GnfwBaryonModel', new=FakeGnfwBaryonModel)

            mus = np.linspace(32, 34, 10)
            zs = np.linspace(0, 2, 5)
            return StackedBaryonLensingSignal(log_masses=mus, redshifts=zs)

        def it_gives_the_avg_wl_mass_for_the_stack(lensing_signal):
            a_szs = np.linspace(-1, 1, 4)
            avg_masses = lensing_signal.avg_mass(a_szs)
            assert np.all(avg_masses == np.ones(4))

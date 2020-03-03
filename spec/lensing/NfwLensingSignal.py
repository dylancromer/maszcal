import numpy as np
import pytest
from maszcal.lensing import NfwLensingSignal
from maszcal.cosmology import CosmoParams
import maszcal.defaults as defaults


class FakeNfwShearModel:
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

    def stacked_delta_sigma(self, rs, cons, a_szs):
        return np.ones(12)

    def weak_lensing_avg_mass(self, a_szs):
        return np.ones(4)

def describe_lensing_signal():

    def describe_stacked_esd():

        @pytest.fixture
        def lensing_signal(mocker):
            mocker.patch('maszcal.lensing.maszcal.lensing.shear.NfwShearModel', new=FakeNfwShearModel)

            mus = np.linspace(32, 34, 10)
            zs = np.linspace(0, 2, 5)
            return NfwLensingSignal(log_masses=mus, redshifts=zs)

        def it_requires_masses():
            zs = np.ones(2)
            with pytest.raises(TypeError):
                lensing_signal = NfwLensingSignal(redshifts=zs)

        def it_gives_a_stacked_model_for_the_esd(lensing_signal):
            rs = np.logspace(-1, 1, 10)
            params = np.array([[3.01, 0],
                               [3.02, 0]])

            esd = lensing_signal.stacked_esd(rs, params)

            assert np.all(esd == np.ones(12))

    def describe_avg_mass():

        @pytest.fixture
        def lensing_signal(mocker):
            mocker.patch('maszcal.lensing.maszcal.lensing.shear.NfwShearModel', new=FakeNfwShearModel)

            mus = np.linspace(32, 34, 10)
            zs = np.linspace(0, 2, 5)
            return NfwLensingSignal(log_masses=mus, redshifts=zs)

        def it_gives_the_avg_wl_mass_for_the_stack(lensing_signal):
            a_szs = np.linspace(-1, 1, 4)
            avg_masses = lensing_signal.avg_mass(a_szs)
            assert np.all(avg_masses == np.ones(4))

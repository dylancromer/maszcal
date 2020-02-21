import numpy as np
import pytest
from maszcal.lensing import SingleMassNfwLensingSignal
from maszcal.cosmology import CosmoParams
import maszcal.defaults as defaults


class FakeSingleMassNfwShearModel:
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


def describe_single_mass_lensing_signal():

    def describe_init():

        def it_requires_redshifts():
            with pytest.raises(TypeError):
                SingleMassNfwLensingSignal()

        def it_allows_a_different_mass_definition(mocker):
            zs = np.ones(1)
            delta = 500
            mass_definition = 'crit'

            SingleMassNfwLensingSignal(zs, delta=delta, mass_definition=mass_definition)

        def it_can_use_a_different_cosmology(mocker):
            zs = np.ones(1)
            cosmo = CosmoParams(neutrino_mass_sum=1)
            SingleMassNfwLensingSignal(zs, cosmo_params=cosmo)

    def describe_esd():

        def it_fails_if_there_are_too_many_redshifts():
            redshifts = np.ones(10)
            with pytest.raises(TypeError):
                lensing_signal = SingleMassNfwLensingSignal(redshift=redshifts)

        @pytest.fixture
        def lensing_signal(mocker):
            mocker.patch('maszcal.lensing.maszcal.shear.SingleMassNfwShearModel', new=FakeSingleMassNfwShearModel)
            redshift = np.array([0])
            return SingleMassNfwLensingSignal(redshift=redshift)

        def it_gives_a_single_mass_model_for_the_esd(lensing_signal):
            rs = np.logspace(-1, 1, 3)
            params = np.array([[1, 1]])

            esd = lensing_signal.esd(rs, params)
            assert np.all(esd == np.ones(13))

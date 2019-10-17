import numpy as np
import pytest
from maszcal.lensing import SingleMassLensingSignal
from maszcal.cosmology import CosmoParams
import maszcal.defaults as defaults


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


def describe_single_mass_lensing_signal():

    def describe_init():

        def it_requires_redshifts():
            with pytest.raises(TypeError):
                SingleMassLensingSignal()

        def it_allows_a_different_mass_definition(mocker):
            zs = np.ones(1)
            delta = 500
            mass_definition = 'crit'

            SingleMassLensingSignal(zs, delta=delta, mass_definition=mass_definition)

        def it_can_use_a_different_cosmology(mocker):
            zs = np.ones(1)
            cosmo = CosmoParams(neutrino_mass_sum=1)
            SingleMassLensingSignal(zs, cosmo_params=cosmo)

    def describe_esd():

        def it_fails_if_there_are_too_many_redshifts():
            redshifts = np.ones(10)
            with pytest.raises(TypeError):
                lensing_signal = SingleMassLensingSignal(redshift=redshifts)

        @pytest.fixture
        def lensing_signal(mocker):
            mocker.patch('maszcal.lensing.model.SingleMassModel', new=FakeSingleMassModel)
            redshift = np.array([0])
            return SingleMassLensingSignal(redshift=redshift)

        def it_gives_a_single_mass_model_for_the_esd(lensing_signal):
            rs = np.logspace(-1, 1, 3)
            params = np.array([[1, 1]])

            esd = lensing_signal.esd(rs, params)
            assert np.all(esd == np.ones(13))
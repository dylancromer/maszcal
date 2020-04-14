import numpy as np
import pytest
import maszcal.lensing
from maszcal.cosmology import CosmoParams
import maszcal.defaults as defaults


class FakeBaryonCmShearModel:
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

    def stacked_delta_sigma(self, rs, alphas, betas, gammas, a_szs):
        return np.ones(12)

    def weak_lensing_avg_mass(self, a_szs):
        return np.ones(4)


def describe_BaryonCmLensingSignal():

    def describe_init():

        def it_requires_redshifts():
            with pytest.raises(TypeError):
                maszcal.lensing.BaryonCmLensingSignal()

        def it_accepts_a_selection_func_file(mocker):
            mus = np.ones(10)
            zs = np.ones(5)
            mocker.patch('maszcal.lensing.maszcal.lensing.shear.BaryonCmShearModel', new=FakeBaryonCmShearModel)
            sel_func_file = 'test/file/here'
            maszcal.lensing.BaryonCmLensingSignal(mus, zs, selection_func_file=sel_func_file)

        def it_accepts_a_weights_file(mocker):
            mus = np.ones(10)
            zs = np.ones(5)
            mocker.patch('maszcal.lensing.maszcal.lensing.shear.BaryonCmShearModel', new=FakeBaryonCmShearModel)
            weights_file = 'test/file/here'
            maszcal.lensing.BaryonCmLensingSignal(mus, zs, lensing_weights_file=weights_file)

        def it_allows_a_different_mass_definition(mocker):
            mus = np.ones(10)
            zs = np.ones(5)
            mocker.patch('maszcal.lensing.maszcal.lensing.shear.BaryonCmShearModel', new=FakeBaryonCmShearModel)

            delta = 500
            mass_definition = 'crit'

            maszcal.lensing.BaryonCmLensingSignal(mus, zs, delta=delta, mass_definition=mass_definition)

        def it_can_use_a_different_cosmology(mocker):
            mus = np.ones(10)
            zs = np.ones(5)
            mocker.patch('maszcal.lensing.maszcal.lensing.shear.BaryonCmShearModel', new=FakeBaryonCmShearModel)

            cosmo = CosmoParams(neutrino_mass_sum=1)
            maszcal.lensing.BaryonCmLensingSignal(mus, zs, cosmo_params=cosmo)

    def describe_stacked_esd():

        @pytest.fixture
        def lensing_signal(mocker):
            mocker.patch('maszcal.lensing.maszcal.lensing.shear.BaryonCmShearModel', new=FakeBaryonCmShearModel)

            mus = np.linspace(32, 34, 10)
            zs = np.linspace(0, 2, 5)
            return maszcal.lensing.BaryonCmLensingSignal(log_masses=mus, redshifts=zs)

        def it_requires_masses():
            zs = np.ones(2)
            with pytest.raises(TypeError):
                lensing_signal = maszcal.lensing.BaryonCmLensingSignal(redshifts=zs)

        def it_gives_a_stacked_model_for_the_esd(lensing_signal):
            rs = np.logspace(-1, 1, 10)
            params = np.array([[1, 3, 1, 0],
                               [1, 3, 1, 0]])

            esd = lensing_signal.stacked_esd(rs, params)

            assert np.all(esd == np.ones(12))

    def describe_avg_mass():

        @pytest.fixture
        def lensing_signal(mocker):
            mocker.patch('maszcal.lensing.maszcal.lensing.shear.BaryonCmShearModel', new=FakeBaryonCmShearModel)

            mus = np.linspace(32, 34, 10)
            zs = np.linspace(0, 2, 5)
            return maszcal.lensing.BaryonCmLensingSignal(log_masses=mus, redshifts=zs)

        def it_gives_the_avg_wl_mass_for_the_stack(lensing_signal):
            a_szs = np.linspace(-1, 1, 4)
            avg_masses = lensing_signal.avg_mass(a_szs)
            assert np.all(avg_masses == np.ones(4))


class FakeBaryonShearModel:
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


def describe_BaryonLensingSignal():

    def describe_stacked_esd():

        @pytest.fixture
        def lensing_signal(mocker):
            mocker.patch('maszcal.lensing.maszcal.lensing.shear.BaryonShearModel', new=FakeBaryonShearModel)

            mus = np.linspace(32, 34, 10)
            zs = np.linspace(0, 2, 5)
            return maszcal.lensing.BaryonLensingSignal(log_masses=mus, redshifts=zs)

        def it_requires_masses():
            zs = np.ones(2)
            with pytest.raises(TypeError):
                lensing_signal = maszcal.lensing.BaryonLensingSignal(redshifts=zs)

        def it_gives_a_stacked_model_for_the_esd(lensing_signal):
            rs = np.logspace(-1, 1, 10)
            params = np.array([[3.01, 1, 3, 1, 0],
                               [3.02, 1, 3, 1, 0]])

            esd = lensing_signal.stacked_esd(rs, params)

            assert np.all(esd == np.ones(12))

    def describe_avg_mass():

        @pytest.fixture
        def lensing_signal(mocker):
            mocker.patch('maszcal.lensing.maszcal.lensing.shear.BaryonShearModel', new=FakeBaryonShearModel)

            mus = np.linspace(32, 34, 10)
            zs = np.linspace(0, 2, 5)
            return maszcal.lensing.BaryonLensingSignal(log_masses=mus, redshifts=zs)

        def it_gives_the_avg_wl_mass_for_the_stack(lensing_signal):
            a_szs = np.linspace(-1, 1, 4)
            avg_masses = lensing_signal.avg_mass(a_szs)
            assert np.all(avg_masses == np.ones(4))


class FakeMiyatakeShearModel:
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

    def stacked_delta_sigma(self, rs, a_szs):
        return np.ones(12)

    def weak_lensing_avg_mass(self, a_szs):
        return np.ones(4)


def describe_MiyatakeLensingSignal():

    def describe_stacked_esd():

        @pytest.fixture
        def lensing_signal(mocker):
            mocker.patch('maszcal.lensing.maszcal.lensing.shear.MiyatakeShearModel', new=FakeMiyatakeShearModel)

            mus = np.linspace(32, 34, 10)
            zs = np.linspace(0, 2, 5)
            return maszcal.lensing.MiyatakeLensingSignal(log_masses=mus, redshifts=zs)

        def it_requires_masses():
            zs = np.ones(2)
            with pytest.raises(TypeError):
                lensing_signal = maszcal.lensing.MiyatakeLensingSignal(redshifts=zs)

        def it_gives_a_stacked_model_for_the_esd(lensing_signal):
            rs = np.logspace(-1, 1, 10)
            params = np.array([[0.01], [0.02]])

            esd = lensing_signal.stacked_esd(rs, params)

            assert np.all(esd == np.ones(12))

    def describe_avg_mass():

        @pytest.fixture
        def lensing_signal(mocker):
            mocker.patch('maszcal.lensing.maszcal.lensing.shear.MiyatakeShearModel', new=FakeMiyatakeShearModel)

            mus = np.linspace(32, 34, 10)
            zs = np.linspace(0, 2, 5)
            return maszcal.lensing.MiyatakeLensingSignal(log_masses=mus, redshifts=zs)

        def it_gives_the_avg_wl_mass_for_the_stack(lensing_signal):
            a_szs = np.linspace(-1, 1, 4)
            avg_masses = lensing_signal.avg_mass(a_szs)
            assert np.all(avg_masses == np.ones(4))


class FakeNfwCmShearModel:
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

    def stacked_delta_sigma(self, rs, a_szs):
        return np.ones(12)

    def weak_lensing_avg_mass(self, a_szs):
        return np.ones(4)


def describe_NfwCmLensingSignal():

    def describe_stacked_esd():

        @pytest.fixture
        def lensing_signal(mocker):
            mocker.patch('maszcal.lensing.maszcal.lensing.shear.NfwCmShearModel', new=FakeNfwCmShearModel)

            mus = np.linspace(32, 34, 10)
            zs = np.linspace(0, 2, 5)
            return maszcal.lensing.NfwCmLensingSignal(log_masses=mus, redshifts=zs)

        def it_gives_a_stacked_model_for_the_esd(lensing_signal):
            rs = np.logspace(-1, 1, 10)
            params = np.array([[0], [0.1]])

            esd = lensing_signal.stacked_esd(rs, params)

            assert np.all(esd == np.ones(12))

    def describe_avg_mass():

        @pytest.fixture
        def lensing_signal(mocker):
            mocker.patch('maszcal.lensing.maszcal.lensing.shear.NfwCmShearModel', new=FakeNfwCmShearModel)

            mus = np.linspace(32, 34, 10)
            zs = np.linspace(0, 2, 5)
            return maszcal.lensing.NfwCmLensingSignal(log_masses=mus, redshifts=zs)

        def it_gives_the_avg_wl_mass_for_the_stack(lensing_signal):
            a_szs = np.linspace(-1, 1, 4)
            avg_masses = lensing_signal.avg_mass(a_szs)
            assert np.all(avg_masses == np.ones(4))


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


def describe_NfwLensingSignal():

    def describe_stacked_esd():

        @pytest.fixture
        def lensing_signal(mocker):
            mocker.patch('maszcal.lensing.maszcal.lensing.shear.NfwShearModel', new=FakeNfwShearModel)

            mus = np.linspace(32, 34, 10)
            zs = np.linspace(0, 2, 5)
            return maszcal.lensing.NfwLensingSignal(log_masses=mus, redshifts=zs)

        def it_requires_masses():
            zs = np.ones(2)
            with pytest.raises(TypeError):
                lensing_signal = maszcal.lensing.NfwLensingSignal(redshifts=zs)

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
            return maszcal.lensing.NfwLensingSignal(log_masses=mus, redshifts=zs)

        def it_gives_the_avg_wl_mass_for_the_stack(lensing_signal):
            a_szs = np.linspace(-1, 1, 4)
            avg_masses = lensing_signal.avg_mass(a_szs)
            assert np.all(avg_masses == np.ones(4))


class FakeSingleMassBaryonShearModel:
    def __init__(
            self,
            redshifts,
            units=1,
            comoving_radii=True,
            delta=200,
            mass_definition='mean',
            cosmo_params=defaults.DefaultCosmology(),
    ):
        pass

    def delta_sigma(self, rs, mus, cons, alphas, betas, gammas):
        return np.ones(13)


def describe_SingleMassBaryonLensingSignal():

    def describe_esd():

        @pytest.fixture
        def lensing_signal(mocker):
            mocker.patch('maszcal.lensing.maszcal.lensing.shear.SingleMassBaryonShearModel', new=FakeSingleMassBaryonShearModel)
            redshift = np.array([0])
            return maszcal.lensing.SingleBaryonLensingSignal(redshifts=redshift)

        def it_gives_a_single_mass_model_for_the_esd(lensing_signal):
            rs = np.logspace(-1, 1, 3)
            params = np.array([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2]])

            esd = lensing_signal.esd(rs, params)
            assert np.all(esd == np.ones(13))


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


def describe_SingleMassNfwLensingSignal():

    def describe_esd():

        def it_fails_if_there_are_too_many_redshifts():
            redshifts = np.ones(10)
            with pytest.raises(TypeError):
                lensing_signal = maszcal.lensing.SingleMassNfwLensingSignal(redshift=redshifts)

        @pytest.fixture
        def lensing_signal(mocker):
            mocker.patch('maszcal.lensing.maszcal.lensing.shear.SingleMassNfwShearModel', new=FakeSingleMassNfwShearModel)
            redshift = np.array([0])
            return maszcal.lensing.SingleMassNfwLensingSignal(redshift=redshift)

        def it_gives_a_single_mass_model_for_the_esd(lensing_signal):
            rs = np.logspace(-1, 1, 3)
            params = np.array([[1, 1]])

            esd = lensing_signal.esd(rs, params)
            assert np.all(esd == np.ones(13))

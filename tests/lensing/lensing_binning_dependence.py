import pytest
import numpy as np
import maszcal.lensing as lensing
import maszcal.model as model
from maszcal.interp_utils import cartesian_prod


def describe_lensing():

    def describe_stacked_lensing_signal():

        @pytest.fixture
        def coarse_mu_signal():
            coarse_mus = np.linspace(np.log(5e13), np.log(5e15), 40)
            redshifts = np.linspace(0, 1.2, 16)
            return lensing.NfwLensingSignal(log_masses=coarse_mus, redshifts=redshifts)

        @pytest.fixture
        def fine_mu_signal():
            fine_mus = np.linspace(np.log(5e13), np.log(5e15), 100)
            redshifts = np.linspace(0, 1.2, 16)
            return lensing.NfwLensingSignal(log_masses=fine_mus, redshifts=redshifts)

        def it_is_stable_at_40_bins_of_mu(coarse_mu_signal, fine_mu_signal):
            rs = np.logspace(-1, 1, 10)

            cons = np.linspace(1, 4, 5)
            a_szs = np.linspace(-2, 2, 5)
            params = cartesian_prod(cons, a_szs)

            coarse_mu_esd = coarse_mu_signal.stacked_esd(rs, params)
            fine_mu_esd = fine_mu_signal.stacked_esd(rs, params)

            assert np.allclose(coarse_mu_esd, fine_mu_esd, rtol=1e-2)

        @pytest.fixture
        def coarse_z_signal():
            mus = np.linspace(np.log(5e13), np.log(5e15), 40)
            coarse_zs = np.linspace(0, 1.2, 10)
            return lensing.NfwLensingSignal(log_masses=mus, redshifts=coarse_zs)

        @pytest.fixture
        def fine_z_signal():
            mus = np.linspace(np.log(5e13), np.log(5e15), 40)
            fine_zs = np.linspace(0, 1.2, 100)
            return lensing.NfwLensingSignal(log_masses=mus, redshifts=fine_zs)

        def it_is_stable_at_10_bins_of_z(coarse_z_signal, fine_z_signal):
            rs = np.logspace(-1, 1, 10)

            cons = np.linspace(1, 4, 5)
            a_szs = np.linspace(-2, 2, 5)
            params = cartesian_prod(cons, a_szs)

            coarse_z_esd = coarse_z_signal.stacked_esd(rs, params)
            fine_z_esd = fine_z_signal.stacked_esd(rs, params)

            assert np.allclose(coarse_z_esd, fine_z_esd, rtol=1e-2)


    def describe_stacked_baryon_lensing_signal():

        @pytest.fixture
        def coarse_mu_signal():
            coarse_mus = np.linspace(np.log(5e13), np.log(5e15), 40)
            redshifts = np.linspace(0, 1.2, 12)
            return lensing.BaryonLensingSignal(log_masses=coarse_mus, redshifts=redshifts)

        @pytest.fixture
        def fine_mu_signal():
            fine_mus = np.linspace(np.log(5e13), np.log(5e15), 100)
            redshifts = np.linspace(0, 1.2, 12)
            return lensing.BaryonLensingSignal(log_masses=fine_mus, redshifts=redshifts)

        def it_is_stable_at_40_bins_of_mu(coarse_mu_signal, fine_mu_signal):
            rs = np.logspace(-1, 1, 8)

            cons = np.linspace(2, 3, 2)
            alphas = np.linspace(0.7, 0.9, 2)
            betas = np.linspace(3.6, 4, 2)
            gammas = np.linspace(0.1, 0.3, 2)
            a_szs = np.linspace(-2, 2, 2)
            params = cartesian_prod(cons, alphas, betas, gammas, a_szs)

            coarse_mu_esd = coarse_mu_signal.stacked_esd(rs, params)
            fine_mu_esd = fine_mu_signal.stacked_esd(rs, params)

            assert np.allclose(coarse_mu_esd, fine_mu_esd, rtol=1e-2)

        @pytest.fixture
        def coarse_z_signal():
            mus = np.linspace(np.log(5e13), np.log(5e15), 10)
            coarse_zs = np.linspace(0, 1.2, 10)
            return lensing.BaryonLensingSignal(log_masses=mus, redshifts=coarse_zs)

        @pytest.fixture
        def fine_z_signal():
            mus = np.linspace(np.log(5e13), np.log(5e15), 10)
            fine_zs = np.linspace(0, 1.2, 100)
            return lensing.BaryonLensingSignal(log_masses=mus, redshifts=fine_zs)

        def it_is_stable_at_10_bins_of_z(coarse_z_signal, fine_z_signal):
            rs = np.logspace(-1, 1, 8)

            cons = np.linspace(2, 3, 2)
            alphas = np.linspace(0.7, 0.9, 2)
            betas = np.linspace(3.6, 4, 2)
            gammas = np.linspace(0.1, 0.3, 2)
            a_szs = np.linspace(-2, 2, 2)

            params = cartesian_prod(cons, alphas, betas, gammas, a_szs)

            coarse_z_esd = coarse_z_signal.stacked_esd(rs, params)
            fine_z_esd = fine_z_signal.stacked_esd(rs, params)

            assert np.allclose(coarse_z_esd, fine_z_esd, rtol=1e-2)

    def describe_miyatake_lensing_signal():

        @pytest.fixture
        def coarse_mu_signal():
            coarse_mus = np.linspace(np.log(5e13), np.log(5e15), 40)
            redshifts = np.linspace(0, 1.2, 12)
            return lensing.MiyatakeLensingSignal(log_masses=coarse_mus, redshifts=redshifts)

        @pytest.fixture
        def fine_mu_signal():
            fine_mus = np.linspace(np.log(5e13), np.log(5e15), 100)
            redshifts = np.linspace(0, 1.2, 12)
            return lensing.MiyatakeLensingSignal(log_masses=fine_mus, redshifts=redshifts)

        def it_is_stable_at_40_bins_of_mu(coarse_mu_signal, fine_mu_signal):
            rs = np.logspace(-1, 1, 10)

            a_szs = np.linspace(-2, 2, 30)

            coarse_mu_esd = coarse_mu_signal.stacked_esd(rs, a_szs)
            fine_mu_esd = fine_mu_signal.stacked_esd(rs, a_szs)

            assert np.allclose(coarse_mu_esd, fine_mu_esd, rtol=1e-2)

        @pytest.fixture
        def coarse_z_signal():
            mus = np.linspace(np.log(5e13), np.log(5e15), 20)
            coarse_zs = np.linspace(0, 1.2, 10)
            return lensing.MiyatakeLensingSignal(log_masses=mus, redshifts=coarse_zs)

        @pytest.fixture
        def fine_z_signal():
            mus = np.linspace(np.log(5e13), np.log(5e15), 20)
            fine_zs = np.linspace(0, 1.2, 100)
            return lensing.MiyatakeLensingSignal(log_masses=mus, redshifts=fine_zs)

        def it_is_stable_at_10_bins_of_z(coarse_z_signal, fine_z_signal):
            rs = np.logspace(-1, 1, 10)
            a_szs = np.linspace(-2, 2, 20)

            coarse_z_esd = coarse_z_signal.stacked_esd(rs, a_szs)
            fine_z_esd = fine_z_signal.stacked_esd(rs, a_szs)

            assert np.allclose(coarse_z_esd, fine_z_esd, rtol=1e-2)

    def describe_nfw_cm_lensing_signal():

        @pytest.fixture
        def coarse_mu_signal():
            coarse_mus = np.linspace(np.log(5e13), np.log(5e15), 40)
            redshifts = np.linspace(0, 1.2, 12)
            return lensing.NfwCmLensingSignal(log_masses=coarse_mus, redshifts=redshifts)

        @pytest.fixture
        def fine_mu_signal():
            fine_mus = np.linspace(np.log(5e13), np.log(5e15), 100)
            redshifts = np.linspace(0, 1.2, 12)
            return lensing.NfwCmLensingSignal(log_masses=fine_mus, redshifts=redshifts)

        def it_is_stable_at_40_bins_of_mu(coarse_mu_signal, fine_mu_signal):
            rs = np.logspace(-1, 1, 10)

            a_szs = np.linspace(-2, 2, 30)

            coarse_mu_esd = coarse_mu_signal.stacked_esd(rs, a_szs)
            fine_mu_esd = fine_mu_signal.stacked_esd(rs, a_szs)

            assert np.allclose(coarse_mu_esd, fine_mu_esd, rtol=1e-2)

        @pytest.fixture
        def coarse_z_signal():
            mus = np.linspace(np.log(5e13), np.log(5e15), 20)
            coarse_zs = np.linspace(0, 1.2, 10)
            return lensing.NfwCmLensingSignal(log_masses=mus, redshifts=coarse_zs)

        @pytest.fixture
        def fine_z_signal():
            mus = np.linspace(np.log(5e13), np.log(5e15), 20)
            fine_zs = np.linspace(0, 1.2, 100)
            return lensing.NfwCmLensingSignal(log_masses=mus, redshifts=fine_zs)

        def it_is_stable_at_10_bins_of_z(coarse_z_signal, fine_z_signal):
            rs = np.logspace(-1, 1, 10)
            a_szs = np.linspace(-2, 2, 20)

            coarse_z_esd = coarse_z_signal.stacked_esd(rs, a_szs)
            fine_z_esd = fine_z_signal.stacked_esd(rs, a_szs)

            assert np.allclose(coarse_z_esd, fine_z_esd, rtol=1e-2)

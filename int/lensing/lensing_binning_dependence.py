import pytest
import numpy as np
import maszcal.lensing as lensing


def describe_lensing():

    def describe_stacked_lensing_signal():

        @pytest.fixture
        def coarse_mu_signal():
            coarse_mus = np.linspace(np.log(5e13), np.log(5e15), 40)
            redshift_bins = np.linspace(0, 1.2, 16)
            return lensing.NfwShearModel(mu_bins=coarse_mus, redshift_bins=redshift_bins)

        @pytest.fixture
        def fine_mu_signal():
            fine_mus = np.linspace(np.log(5e13), np.log(5e15), 100)
            redshift_bins = np.linspace(0, 1.2, 16)
            return lensing.NfwShearModel(mu_bins=fine_mus, redshift_bins=redshift_bins)

        def it_is_stable_at_40_bins_of_mu(coarse_mu_signal, fine_mu_signal):
            rs = np.logspace(-1, 1, 10)

            cons = np.linspace(1, 4, 5)
            a_szs = np.linspace(-2, 2, 5)

            coarse_mu_esd = coarse_mu_signal.stacked_delta_sigma(rs, cons, a_szs)
            fine_mu_esd = fine_mu_signal.stacked_delta_sigma(rs, cons, a_szs)

            assert np.allclose(coarse_mu_esd, fine_mu_esd, rtol=1e-2)

        @pytest.fixture
        def coarse_z_signal():
            mus = np.linspace(np.log(5e13), np.log(5e15), 40)
            coarse_zs = np.linspace(0, 1.2, 10)
            return lensing.NfwShearModel(mu_bins=mus, redshift_bins=coarse_zs)

        @pytest.fixture
        def fine_z_signal():
            mus = np.linspace(np.log(5e13), np.log(5e15), 40)
            fine_zs = np.linspace(0, 1.2, 100)
            return lensing.NfwShearModel(mu_bins=mus, redshift_bins=fine_zs)

        def it_is_stable_at_10_bins_of_z(coarse_z_signal, fine_z_signal):
            rs = np.logspace(-1, 1, 10)

            cons = np.linspace(1, 4, 5)
            a_szs = np.linspace(-2, 2, 5)

            coarse_z_esd = coarse_z_signal.stacked_delta_sigma(rs, cons, a_szs)
            fine_z_esd = fine_z_signal.stacked_delta_sigma(rs, cons, a_szs)

            assert np.allclose(coarse_z_esd, fine_z_esd, rtol=1e-2)


    def describe_stacked_baryon_lensing_signal():

        @pytest.fixture
        def coarse_mu_signal():
            coarse_mus = np.linspace(np.log(5e13), np.log(5e15), 40)
            redshift_bins = np.linspace(0, 1.2, 12)
            return lensing.BaryonShearModel(mu_bins=coarse_mus, redshift_bins=redshift_bins)

        @pytest.fixture
        def fine_mu_signal():
            fine_mus = np.linspace(np.log(5e13), np.log(5e15), 100)
            redshift_bins = np.linspace(0, 1.2, 12)
            return lensing.BaryonShearModel(mu_bins=fine_mus, redshift_bins=redshift_bins)

        def it_is_stable_at_40_bins_of_mu(coarse_mu_signal, fine_mu_signal):
            rs = np.logspace(-1, 1, 8)

            cons = np.linspace(2, 3, 2)
            alphas = np.linspace(0.7, 0.9, 2)
            betas = np.linspace(3.6, 4, 2)
            gammas = np.linspace(0.1, 0.3, 2)
            a_szs = np.linspace(-2, 2, 2)

            coarse_mu_esd = coarse_mu_signal.stacked_delta_sigma(rs, cons, alphas, betas, gammas, a_szs)
            fine_mu_esd = fine_mu_signal.stacked_delta_sigma(rs, cons, alphas, betas, gammas, a_szs)

            assert np.allclose(coarse_mu_esd, fine_mu_esd, rtol=1e-2)

        @pytest.fixture
        def coarse_z_signal():
            mus = np.linspace(np.log(5e13), np.log(5e15), 10)
            coarse_zs = np.linspace(0, 1.2, 10)
            return lensing.BaryonShearModel(mu_bins=mus, redshift_bins=coarse_zs)

        @pytest.fixture
        def fine_z_signal():
            mus = np.linspace(np.log(5e13), np.log(5e15), 10)
            fine_zs = np.linspace(0, 1.2, 100)
            return lensing.BaryonShearModel(mu_bins=mus, redshift_bins=fine_zs)

        def it_is_stable_at_10_bins_of_z(coarse_z_signal, fine_z_signal):
            rs = np.logspace(-1, 1, 8)

            cons = np.linspace(2, 3, 2)
            alphas = np.linspace(0.7, 0.9, 2)
            betas = np.linspace(3.6, 4, 2)
            gammas = np.linspace(0.1, 0.3, 2)
            a_szs = np.linspace(-2, 2, 2)

            coarse_z_esd = coarse_z_signal.stacked_delta_sigma(rs, cons, alphas, betas, gammas, a_szs)
            fine_z_esd = fine_z_signal.stacked_delta_sigma(rs, cons, alphas, betas, gammas, a_szs)

            assert np.allclose(coarse_z_esd, fine_z_esd, rtol=1e-2)

    def describe_miyatake_lensing_signal():

        @pytest.fixture
        def coarse_mu_signal():
            coarse_mus = np.linspace(np.log(5e13), np.log(5e15), 40)
            redshift_bins = np.linspace(0, 1.2, 12)
            return lensing.MiyatakeShearModel(mu_bins=coarse_mus, redshift_bins=redshift_bins)

        @pytest.fixture
        def fine_mu_signal():
            fine_mus = np.linspace(np.log(5e13), np.log(5e15), 100)
            redshift_bins = np.linspace(0, 1.2, 12)
            return lensing.MiyatakeShearModel(mu_bins=fine_mus, redshift_bins=redshift_bins)

        def it_is_stable_at_40_bins_of_mu(coarse_mu_signal, fine_mu_signal):
            rs = np.logspace(-1, 1, 10)

            a_szs = np.linspace(-2, 2, 30)

            coarse_mu_esd = coarse_mu_signal.stacked_delta_sigma(rs, a_szs)
            fine_mu_esd = fine_mu_signal.stacked_delta_sigma(rs, a_szs)

            assert np.allclose(coarse_mu_esd, fine_mu_esd, rtol=1e-2)

        @pytest.fixture
        def coarse_z_signal():
            mus = np.linspace(np.log(5e13), np.log(5e15), 20)
            coarse_zs = np.linspace(0, 1.2, 10)
            return lensing.MiyatakeShearModel(mu_bins=mus, redshift_bins=coarse_zs)

        @pytest.fixture
        def fine_z_signal():
            mus = np.linspace(np.log(5e13), np.log(5e15), 20)
            fine_zs = np.linspace(0, 1.2, 100)
            return lensing.MiyatakeShearModel(mu_bins=mus, redshift_bins=fine_zs)

        def it_is_stable_at_10_bins_of_z(coarse_z_signal, fine_z_signal):
            rs = np.logspace(-1, 1, 10)
            a_szs = np.linspace(-2, 2, 20)

            coarse_z_esd = coarse_z_signal.stacked_delta_sigma(rs, a_szs)
            fine_z_esd = fine_z_signal.stacked_delta_sigma(rs, a_szs)

            assert np.allclose(coarse_z_esd, fine_z_esd, rtol=1e-2)

    def describe_nfw_cm_lensing_signal():

        @pytest.fixture
        def coarse_mu_signal():
            coarse_mus = np.linspace(np.log(5e13), np.log(5e15), 40)
            redshift_bins = np.linspace(0, 1.2, 12)
            return lensing.NfwCmShearModel(mu_bins=coarse_mus, redshift_bins=redshift_bins)

        @pytest.fixture
        def fine_mu_signal():
            fine_mus = np.linspace(np.log(5e13), np.log(5e15), 100)
            redshift_bins = np.linspace(0, 1.2, 12)
            return lensing.NfwCmShearModel(mu_bins=fine_mus, redshift_bins=redshift_bins)

        def it_is_stable_at_40_bins_of_mu(coarse_mu_signal, fine_mu_signal):
            rs = np.logspace(-1, 1, 10)

            a_szs = np.linspace(-2, 2, 30)

            coarse_mu_esd = coarse_mu_signal.stacked_delta_sigma(rs, a_szs)
            fine_mu_esd = fine_mu_signal.stacked_delta_sigma(rs, a_szs)

            assert np.allclose(coarse_mu_esd, fine_mu_esd, rtol=1e-2)

        @pytest.fixture
        def coarse_z_signal():
            mus = np.linspace(np.log(5e13), np.log(5e15), 20)
            coarse_zs = np.linspace(0, 1.2, 10)
            return lensing.NfwCmShearModel(mu_bins=mus, redshift_bins=coarse_zs)

        @pytest.fixture
        def fine_z_signal():
            mus = np.linspace(np.log(5e13), np.log(5e15), 20)
            fine_zs = np.linspace(0, 1.2, 100)
            return lensing.NfwCmShearModel(mu_bins=mus, redshift_bins=fine_zs)

        def it_is_stable_at_10_bins_of_z(coarse_z_signal, fine_z_signal):
            rs = np.logspace(-1, 1, 10)
            a_szs = np.linspace(-2, 2, 20)

            coarse_z_esd = coarse_z_signal.stacked_delta_sigma(rs, a_szs)
            fine_z_esd = fine_z_signal.stacked_delta_sigma(rs, a_szs)

            assert np.allclose(coarse_z_esd, fine_z_esd, rtol=1e-2)

    def describe_baryon_cm_lensing_signal():

        @pytest.fixture
        def coarse_mu_signal():
            coarse_mus = np.linspace(np.log(5e13), np.log(5e15), 40)
            redshift_bins = np.linspace(0, 1.2, 12)
            return lensing.BaryonCmShearModel(mu_bins=coarse_mus, redshift_bins=redshift_bins)

        @pytest.fixture
        def fine_mu_signal():
            fine_mus = np.linspace(np.log(5e13), np.log(5e15), 100)
            redshift_bins = np.linspace(0, 1.2, 12)
            return lensing.BaryonCmShearModel(mu_bins=fine_mus, redshift_bins=redshift_bins)

        def it_is_stable_at_40_bins_of_mu(coarse_mu_signal, fine_mu_signal):
            rs = np.logspace(-1, 1, 8)

            alphas = np.linspace(0.7, 0.9, 2)
            betas = np.linspace(3.6, 4, 2)
            gammas = np.linspace(0.1, 0.3, 2)
            a_szs = np.linspace(-2, 2, 2)

            coarse_mu_esd = coarse_mu_signal.stacked_delta_sigma(rs, alphas, betas, gammas, a_szs)
            fine_mu_esd = fine_mu_signal.stacked_delta_sigma(rs, alphas, betas, gammas, a_szs)

            assert np.allclose(coarse_mu_esd, fine_mu_esd, rtol=1e-2)

        @pytest.fixture
        def coarse_z_signal():
            mus = np.linspace(np.log(5e13), np.log(5e15), 10)
            coarse_zs = np.linspace(0, 1.2, 10)
            return lensing.BaryonCmShearModel(mu_bins=mus, redshift_bins=coarse_zs)

        @pytest.fixture
        def fine_z_signal():
            mus = np.linspace(np.log(5e13), np.log(5e15), 10)
            fine_zs = np.linspace(0, 1.2, 100)
            return lensing.BaryonCmShearModel(mu_bins=mus, redshift_bins=fine_zs)

        def it_is_stable_at_10_bins_of_z(coarse_z_signal, fine_z_signal):
            rs = np.logspace(-1, 1, 8)

            alphas = np.linspace(0.7, 0.9, 2)
            betas = np.linspace(3.6, 4, 2)
            gammas = np.linspace(0.1, 0.3, 2)
            a_szs = np.linspace(-2, 2, 2)

            coarse_z_esd = coarse_z_signal.stacked_delta_sigma(rs, alphas, betas, gammas, a_szs)
            fine_z_esd = fine_z_signal.stacked_delta_sigma(rs, alphas, betas, gammas, a_szs)

            assert np.allclose(coarse_z_esd, fine_z_esd, rtol=1e-2)

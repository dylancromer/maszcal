from dataclasses import dataclass
import pytest
import numpy as np
import astropy.units as u
import maszcal.lensing
import maszcal.cosmology


def fake_projector_esd(rs, rho_func):
    rhos = rho_func(rs)
    return np.ones(rhos.shape)


def fake_projector_sd(rs, rho_func):
    rhos = rho_func(rs)
    return np.ones(rhos.shape)


class FakeConModel:
    def __init__(self, mass_def, cosmology=None):
        pass

    def c(self, masses, redshifts, mass_def):
        return np.ones((masses.size, redshifts.size))

    def convert_mass_def(self, masses, redshifts, old_def, new_def):
        return np.ones((masses.size, redshifts.size))


def fake_rho_total(rs, zs, mus, *params):
    return np.ones(rs.shape + mus.shape + zs.shape + (params[0].size,))


def describe_IntegratedShearModel():

    def describe_math():

        @pytest.fixture
        def baryon_model():
            mus = np.linspace(np.log(1e14), np.log(1e16), 9)
            zs = np.linspace(0, 1, 8)
            return maszcal.lensing.IntegratedShearModel(
                mus,
                zs,
                rho_func=fake_rho_total,
                esd_func=fake_projector_esd,
            )

        def it_can_calculate_a_total_delta_sigma(baryon_model):
            radii = np.logspace(-1, 1, 10)
            zs = np.linspace(0, 1, 8)
            mus = np.log(1e14)*np.ones(2)
            cs = 3*np.ones(3)
            alphas = np.ones(3)
            betas = 2*np.ones(3)
            gammas = np.ones(3)

            ds = baryon_model._shear.delta_sigma_total(radii, zs, mus, cs, alphas, betas, gammas)

            assert np.all(ds > 0)

        def it_can_calculate_a_stacked_delta_sigma(baryon_model):
            radii = np.logspace(-1, 1, 10)
            cs = 3*np.ones(3)
            alphas = np.ones(3)
            betas = 2*np.ones(3)
            gammas = np.ones(3)
            a_szs = np.zeros(3)

            baryon_model._init_stacker()
            baryon_model.stacker.dnumber_dlogmass = lambda : np.ones(
                (baryon_model.mu_bins.size, baryon_model.redshift_bins.size)
            )
            assert np.all(baryon_model.stacker.dnumber_dlogmass() == 1)

            ds = baryon_model.stacked_delta_sigma(radii, cs, alphas, betas, gammas, a_szs)

            assert ds.shape == (10, 3)

    def describe_units():

        @pytest.fixture
        def baryon_model():
            mus = np.linspace(np.log(1e14), np.log(1e16), 9)
            zs = np.linspace(0, 1, 8)
            return maszcal.lensing.IntegratedShearModel(
                mus,
                zs,
                rho_func=fake_rho_total,
                units=u.Msun/u.pc**2,
            )

        def it_has_correct_units(baryon_model):
            radii = np.logspace(-1, 1, 10)
            zs = np.linspace(0, 1, 8)
            mus = np.log(1e14)*np.ones(2)
            cs = 3*np.ones(3)
            alphas = np.ones(3)
            betas = 2*np.ones(3)
            gammas = np.ones(3)

            ds = baryon_model._shear.delta_sigma_total(radii, zs, mus, cs, alphas, betas, gammas)

            assert np.all(radii[:, None, None, None]*ds < 1e2)


def describe_CmStacker():

    def describe_math():

        @pytest.fixture
        def stacker():
            mus = np.linspace(np.log(1e14), np.log(1e15), 10)
            redshifts = np.linspace(0, 1, 8)
            return maszcal.lensing.CmStacker(
                mus,
                redshifts,
                units=u.Msun/u.pc**2,
                delta=200,
                sz_scatter=0.2,
                comoving=True,
                mass_definition='mean',
            )

        def prob_musz_given_mu_is_not_negative(stacker):
            mu_szs = np.linspace(np.log(1e12), np.log(1e16), 10)
            mus = np.linspace(np.log(1e12), np.log(1e16), 20)

            a_szs = np.linspace(-1, 1, 5)

            prob_sz = stacker.prob_musz_given_mu(mu_szs, mus, a_szs)
            assert np.all(prob_sz >= 0)

        def prob_musz_given_mu_integrates_to_1(stacker):
            mu_szs = np.linspace(np.log(1e11), np.log(1e17), 100)
            mus = np.array([np.log(1e15)])

            a_szs = np.linspace(-1, 1, 5)

            prob_sz = stacker.prob_musz_given_mu(mu_szs, mus, a_szs)
            integ = np.trapz(prob_sz, x=mu_szs, axis=0)
            assert np.allclose(integ, 1)

        def delta_sigma_of_r_divided_by_nsz_always_one(stacker):
            """
            This test functions by setting delta_sigma_of_mass to be constant,
            resulting in it being identical to the normalization. Thus this test should
            always return 1s, rather than a true precomputed value
            """
            zs = np.linspace(0, 2, 8)

            stacker.dnumber_dlogmass = lambda : np.ones(
                (stacker.mu_bins.size, stacker.redshift_bins.size)
            )
            assert np.all(stacker.dnumber_dlogmass() == 1)

            rs = np.logspace(-1, 1, 21)
            a_szs = np.linspace(-1, 1, 1)

            delta_sigmas_of_mass = np.ones((stacker.mu_bins.size, zs.size, rs.size))

            delta_sigmas = stacker.stacked_delta_sigma(delta_sigmas_of_mass, rs, a_szs)

            precomp_delta_sigmas = np.ones((rs.size, 1))

            np.testing.assert_allclose(delta_sigmas, precomp_delta_sigmas)

        def it_can_handle_delta_sigmas_of_mass_with_different_params(stacker):
            N_PARAMS = 3
            rs = np.logspace(-1, 1, 21)
            a_szs = np.linspace(-1, 1, N_PARAMS)

            stacker.dnumber_dlogmass = lambda : np.ones(
                (stacker.mu_bins.size, stacker.redshift_bins.size)
            )
            assert np.all(stacker.dnumber_dlogmass() == 1)

            delta_sigmas_of_mass = np.ones((stacker.mu_bins.size, stacker.redshift_bins.size, rs.size))

            delta_sigmas = stacker.stacked_delta_sigma(delta_sigmas_of_mass, rs, a_szs)

            assert delta_sigmas.shape == (21, N_PARAMS)

        def it_can_compute_wl_avg_masses(stacker):
            zs = np.linspace(0, 2, 8)

            stacker.dnumber_dlogmass = lambda : np.ones(
                (stacker.mu_bins.size, stacker.redshift_bins.size)
            )
            assert np.all(stacker.dnumber_dlogmass() == 1)

            a_szs = np.linspace(-1, 1, 4)

            avg_masses = stacker.weak_lensing_avg_mass(a_szs)

            assert avg_masses.shape == (4,)

        def it_complains_about_nans(stacker):
            zs = np.linspace(0, 2, 8)

            # Ugly mock of mass function
            stacker.dnumber_dlogmass = lambda : np.full(
                (stacker.mu_bins.size, stacker.redshift_bins.size),
                np.nan,
            )

            rs = np.logspace(-1, 1, 10)
            a_szs = np.linspace(-1, 1, 2)
            delta_sigmas_of_mass = np.ones((stacker.mu_bins.size, zs.size, rs.size))

            with pytest.raises(ValueError):
                stacker.stacked_delta_sigma(delta_sigmas_of_mass, rs, a_szs)


def describe_MiyatakeShearModel():

    def describe_math_functions():

        @pytest.fixture
        def stacked_model():
            mus = np.linspace(np.log(1e12), np.log(1e16), 20)
            zs = np.linspace(0, 2, 8)

            model = maszcal.lensing.MiyatakeShearModel(mus, zs, con_class=FakeConModel)
            model._init_stacker()

            return model

        def it_computes_weak_lensing_avg_mass(stacked_model):
            stacked_model.stacker.dnumber_dlogmass = lambda : np.ones(
                (stacked_model.mu_bins.size, stacked_model.redshift_bins.size)
            )
            assert np.all(stacked_model.stacker.dnumber_dlogmass() == 1)

            a_szs = np.linspace(-1, 1, 1)

            avg_wl_mass = stacked_model.weak_lensing_avg_mass(a_szs)

            assert avg_wl_mass.shape == (1,)
            assert avg_wl_mass > 0

        def it_computes_stacked_delta_sigmas_with_the_right_shape(stacked_model):
            stacked_model.stacker.dnumber_dlogmass = lambda : np.ones(
                (stacked_model.mu_bins.size, stacked_model.redshift_bins.size)
            )
            assert np.all(stacked_model.stacker.dnumber_dlogmass() == 1)

            a_szs = np.linspace(-1, 1, 1)

            rs = np.logspace(-1, 1, 4)

            delta_sigs_stacked = stacked_model.stacked_delta_sigma(rs, a_szs)

            assert delta_sigs_stacked.shape == (4, 1)


def describe_MiyatakeStacker():

    def describe_math():

        @pytest.fixture
        def stacker():
            mus = np.linspace(np.log(1e14), np.log(1e15), 10)
            redshifts = np.linspace(0, 1, 8)
            return maszcal.lensing.MiyatakeStacker(
                mus,
                redshifts,
                units=u.Msun/u.pc**2,
                delta=200,
                sz_scatter=0.2,
                comoving=True,
                mass_definition='mean',
            )

        def prob_musz_given_mu_is_not_negative(stacker):
            mu_szs = np.linspace(np.log(1e12), np.log(1e16), 10)
            mus = np.linspace(np.log(1e12), np.log(1e16), 20)

            a_szs = np.linspace(-1, 1, 5)

            prob_sz = stacker.prob_musz_given_mu(mu_szs, mus, a_szs)
            assert np.all(prob_sz >= 0)

        def prob_musz_given_mu_integrates_to_1(stacker):
            mu_szs = np.linspace(np.log(1e11), np.log(1e17), 100)
            mus = np.array([np.log(1e15)])

            a_szs = np.linspace(-1, 1, 5)

            prob_sz = stacker.prob_musz_given_mu(mu_szs, mus, a_szs)
            integ = np.trapz(prob_sz, x=mu_szs, axis=0)
            assert np.allclose(integ, 1)

        def delta_sigma_of_r_divided_by_nsz_always_one(stacker):
            """
            This test functions by setting delta_sigma_of_mass to be constant,
            resulting in it being identical to the normalization. Thus this test should
            always return 1s, rather than a true precomputed value
            """
            zs = np.linspace(0, 2, 8)

            stacker.dnumber_dlogmass = lambda : np.ones(
                (stacker.mu_bins.size, stacker.redshift_bins.size)
            )
            assert np.all(stacker.dnumber_dlogmass() == 1)

            rs = np.logspace(-1, 1, 21)
            a_szs = np.linspace(-1, 1, 1)

            delta_sigmas_of_mass = np.ones((stacker.mu_bins.size, zs.size, rs.size))

            delta_sigmas = stacker.stacked_delta_sigma(delta_sigmas_of_mass, rs, a_szs)

            precomp_delta_sigmas = np.ones((rs.size, 1))

            np.testing.assert_allclose(delta_sigmas, precomp_delta_sigmas)

        def it_can_handle_delta_sigmas_of_mass_with_different_params(stacker):
            N_PARAMS = 3
            rs = np.logspace(-1, 1, 21)
            a_szs = np.linspace(-1, 1, N_PARAMS)

            stacker.dnumber_dlogmass = lambda : np.ones(
                (stacker.mu_bins.size, stacker.redshift_bins.size)
            )
            assert np.all(stacker.dnumber_dlogmass() == 1)

            delta_sigmas_of_mass = np.ones((stacker.mu_bins.size, stacker.redshift_bins.size, rs.size))

            delta_sigmas = stacker.stacked_delta_sigma(delta_sigmas_of_mass, rs, a_szs)

            assert delta_sigmas.shape == (21, N_PARAMS)

        def it_complains_about_nans(stacker):
            zs = np.linspace(0, 2, 8)

            # Ugly mock of mass function
            stacker.dnumber_dlogmass = lambda : np.full(
                (stacker.mu_bins.size, stacker.redshift_bins.size),
                np.nan,
            )

            # Ugly mock of concentration model
            stacker._m500c = lambda mus: np.exp(mus)

            rs = np.logspace(-1, 1, 10)
            a_szs = np.linspace(-1, 1, 2)
            delta_sigmas_of_mass = np.ones((stacker.mu_bins.size, zs.size, rs.size))

            with pytest.raises(ValueError):
                stacker.stacked_delta_sigma(delta_sigmas_of_mass, rs, a_szs)


def describe_NfwCmShearModel():

    def describe_math_functions():

        @pytest.fixture
        def stacked_model():
            mus = np.linspace(np.log(1e12), np.log(1e16), 20)
            zs = np.linspace(0, 2, 8)

            model = maszcal.lensing.NfwCmShearModel(mus, zs)

            return model

        def it_computes_stacked_delta_sigma():
            mus = np.linspace(np.log(1e12), np.log(1e16), 20)
            zs = np.linspace(0, 2, 8)
            stacked_model = maszcal.lensing.NfwCmShearModel(mus, zs, con_class=FakeConModel)
            stacked_model._init_stacker()

            stacked_model.stacker.dnumber_dlogmass = lambda : np.ones(
                (stacked_model.mu_bins.size, stacked_model.redshift_bins.size)
            )
            assert np.all(stacked_model.stacker.dnumber_dlogmass() == 1)

            rs = np.logspace(-1, 1, 4)
            a_szs = np.linspace(-1, 1, 3)

            avg_wl_mass = stacked_model.stacked_delta_sigma(rs, a_szs)

            assert avg_wl_mass.shape == (rs.size, a_szs.size)

        def it_computes_weak_lensing_avg_mass():
            mus = np.linspace(np.log(1e12), np.log(1e16), 20)
            zs = np.linspace(0, 2, 8)
            stacked_model = maszcal.lensing.NfwCmShearModel(mus, zs, con_class=FakeConModel)
            stacked_model._init_stacker()

            stacked_model.stacker.dnumber_dlogmass = lambda : np.ones(
                (stacked_model.mu_bins.size, stacked_model.redshift_bins.size)
            )
            assert np.all(stacked_model.stacker.dnumber_dlogmass() == 1)

            a_szs = np.linspace(-1, 1, 3)

            avg_wl_mass = stacked_model.weak_lensing_avg_mass(a_szs)

            assert avg_wl_mass.shape == (3,)


def describe_NfwShearModel():

    def describe_math_functions():

        @pytest.fixture
        def stacked_model():
            mus = np.linspace(np.log(1e12), np.log(1e16), 20)
            zs = np.linspace(0, 2, 8)

            model = maszcal.lensing.NfwShearModel(mus, zs)

            return model

        def it_computes_weak_lensing_avg_mass():
            mus = np.linspace(np.log(1e12), np.log(1e16), 20)
            zs = np.linspace(0, 2, 8)
            stacked_model = maszcal.lensing.NfwShearModel(mus, zs)
            stacked_model._init_stacker()

            stacked_model.stacker.dnumber_dlogmass = lambda : np.ones(
                (stacked_model.mu_bins.size, stacked_model.redshift_bins.size)
            )
            assert np.all(stacked_model.stacker.dnumber_dlogmass() == 1)

            a_szs = np.linspace(-1, 1, 1)

            avg_wl_mass = stacked_model.weak_lensing_avg_mass(a_szs)

            assert avg_wl_mass.shape == (1,)


@dataclass
class FakeMatterPower:
    cosmo_params: object

    def spectrum(self, ks, zs, is_nonlinear):
        return np.ones(zs.shape + ks.shape)


def describe_Stacker():

    def describe_math():

        @pytest.fixture
        def stacker():
            mus = np.linspace(np.log(1e14), np.log(1e15), 10)
            redshifts = np.linspace(0, 1, 8)
            return maszcal.lensing.Stacker(
                mus,
                redshifts,
                units=u.Msun/u.pc**2,
                delta=200,
                sz_scatter=0.2,
                comoving=True,
                mass_definition='mean',
            )

        def prob_musz_given_mu_is_not_negative(stacker):
            mu_szs = np.linspace(np.log(1e12), np.log(1e16), 10)
            mus = np.linspace(np.log(1e12), np.log(1e16), 20)

            a_szs = np.linspace(-1, 1, 5)

            prob_sz = stacker.prob_musz_given_mu(mu_szs, mus, a_szs)
            assert np.all(prob_sz >= 0)

        def prob_musz_given_mu_integrates_to_1(stacker):
            mu_szs = np.linspace(np.log(1e11), np.log(1e17), 100)
            mus = np.array([np.log(1e15)])

            a_szs = np.linspace(-1, 1, 5)

            prob_sz = stacker.prob_musz_given_mu(mu_szs, mus, a_szs)
            integ = np.trapz(prob_sz, x=mu_szs, axis=0)
            assert np.allclose(integ, 1)

        def delta_sigma_of_r_divided_by_nsz_always_one(stacker):
            """
            This test functions by setting delta_sigma_of_mass to be constant,
            resulting in it being identical to the normalization. Thus this test should
            always return 1s, rather than a true precomputed value
            """
            zs = np.linspace(0, 2, 8)

            stacker.dnumber_dlogmass = lambda : np.ones(
                (stacker.mu_bins.size, stacker.redshift_bins.size)
            )
            assert np.all(stacker.dnumber_dlogmass() == 1)

            rs = np.logspace(-1, 1, 21)
            cons = np.linspace(2, 4, 1)
            a_szs = np.linspace(-1, 1, 1)

            delta_sigmas_of_mass = np.ones((stacker.mu_bins.size, zs.size, rs.size, cons.size))

            delta_sigmas = stacker.stacked_delta_sigma(delta_sigmas_of_mass, rs, a_szs)

            precomp_delta_sigmas = np.ones((rs.size, 1))

            np.testing.assert_allclose(delta_sigmas, precomp_delta_sigmas)

        def it_can_handle_delta_sigmas_of_mass_with_different_params(stacker):
            N_PARAMS = 3
            rs = np.logspace(-1, 1, 21)
            a_szs = np.linspace(-1, 1, N_PARAMS)

            stacker.dnumber_dlogmass = lambda : np.ones(
                (stacker.mu_bins.size, stacker.redshift_bins.size)
            )
            assert np.all(stacker.dnumber_dlogmass() == 1)

            delta_sigmas_of_mass = np.ones((stacker.mu_bins.size, stacker.redshift_bins.size, rs.size, N_PARAMS))

            delta_sigmas = stacker.stacked_delta_sigma(delta_sigmas_of_mass, rs, a_szs)

        def it_complains_about_nans(stacker):
            zs = np.linspace(0, 2, 8)
            stacker.dnumber_dlogmass = lambda : np.full(
                (stacker.mu_bins.size, stacker.redshift_bins.size),
                np.nan,
            )

            rs = np.logspace(-1, 1, 10)
            cons = np.linspace(2, 4, 1)
            a_szs = np.linspace(-1, 1, 1)
            delta_sigmas_of_mass = np.ones((stacker.mu_bins.size, zs.size, rs.size, cons.size))

            with pytest.raises(ValueError):
                stacker.stacked_delta_sigma(delta_sigmas_of_mass, rs, a_szs)

            with pytest.raises(ValueError):
                stacker.weak_lensing_avg_mass(a_szs)

        def describe_power_spectrum_works():

            @pytest.fixture
            def stacker():
                mus = np.linspace(np.log(1e14), np.log(1e15), 10)
                redshifts = np.linspace(0, 1, 8)
                return maszcal.lensing.Stacker(
                    mus,
                    redshifts,
                    units=u.Msun/u.pc**2,
                    delta=200,
                    sz_scatter=0.2,
                    comoving=True,
                    mass_definition='mean',
                    matter_power_class=FakeMatterPower,
                )

            def it_can_calculate_the_mass_function(stacker):
                assert np.all(stacker.dnumber_dlogmass() >= 0)
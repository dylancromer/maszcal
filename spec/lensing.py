from dataclasses import dataclass
import pytest
import numpy as np
import astropy.units as u
import maszcal.lensing


class FakeProjector:
    @staticmethod
    def esd(rs, rho_func):
        rhos = rho_func(rs)
        return np.ones(rhos.shape)


class FakeConModel:
    def __init__(self, mass_def, cosmology=None):
        pass

    def c(self, masses, redshifts, mass_def):
        return np.ones((masses.size, redshifts.size))

    def convert_mass_def(self, masses, redshifts, old_def, new_def):
        return np.ones((masses.size, redshifts.size))


def describe_BaryonCmShearModel():

    def describe_math():

        @pytest.fixture
        def baryon_model(mocker):
            mocker.patch('maszcal.lensing.ConModel', new=FakeConModel)
            mocker.patch('maszcal.lensing.projector', new=FakeProjector)
            mus = np.linspace(np.log(1e14), np.log(1e16), 9)
            zs = np.linspace(0, 1, 8)
            return maszcal.lensing.BaryonCmShearModel(mus, zs)

        def it_can_calculate_a_gnfw_rho(baryon_model):
            radii = np.logspace(-1, 1, 10)
            mus = np.log(1e14)*np.ones(2)
            alphas = np.ones(3)
            betas = 2*np.ones(3)
            gammas = np.ones(3)

            rhos = baryon_model.rho_bary(radii, mus, alphas, betas, gammas)

            assert np.all(rhos > 0)

        def it_can_calculate_an_nfw_rho(baryon_model):
            radii = np.logspace(-1, 1, 10)
            mus = np.log(1e14)*np.ones(1)

            rhos = baryon_model.rho_cdm(radii, mus)

            assert np.all(rhos > 0)

        def it_has_the_correct_baryon_fraction(baryon_model):
            rs = np.linspace(
                baryon_model.MIN_INTEGRATION_RADIUS,
                baryon_model.MAX_INTEGRATION_RADIUS,
                baryon_model.NUM_INTEGRATION_RADII
            )
            mus = np.log(1e14)*np.ones(1)
            alphas = 0.88*np.ones(1)
            betas = 3.8*np.ones(1)
            gammas = 0.2*np.ones(1)

            rho_barys = baryon_model.rho_bary(rs, mus, alphas, betas, gammas)
            rho_cdms = np.moveaxis(baryon_model.rho_cdm(rs, mus)[..., None], 2, 0)

            ratio = np.trapz(
                rho_barys * rs[:, None, None, None]**2,
                x=rs,
                axis=0
            ) / np.trapz(
                (rho_barys + rho_cdms) * rs[:, None, None, None]**2,
                x=rs,
                axis=0
            )

            f_b = baryon_model.baryon_frac
            assert np.allclose(ratio, f_b)

        def it_can_calculate_an_nfw_delta_sigma(baryon_model):
            radii = np.logspace(-1, 1, 10)
            mus = np.log(1e14)*np.ones(1)

            ds = baryon_model.delta_sigma_cdm(radii, mus)

            assert np.all(ds > 0)

        def it_can_calculate_a_gnfw_delta_sigma(baryon_model):
            radii = np.logspace(-1, 1, 10)
            mus = np.log(1e14)*np.ones(2)
            alphas = np.ones(3)
            betas = 2*np.ones(3)
            gammas = np.ones(3)

            ds = baryon_model.delta_sigma_bary(radii, mus, alphas, betas, gammas)

            assert np.all(ds > 0)

        def it_can_calculate_a_total_delta_sigma(baryon_model):
            radii = np.logspace(-1, 1, 10)
            mus = np.log(1e14)*np.ones(2)
            alphas = np.ones(3)
            betas = 2*np.ones(3)
            gammas = np.ones(3)

            ds = baryon_model.delta_sigma_total(radii, mus, alphas, betas, gammas)

            assert np.all(ds > 0)

        def it_can_calculate_a_stacked_delta_sigma(baryon_model):
            radii = np.logspace(-1, 1, 10)
            alphas = np.ones(3)
            betas = 2*np.ones(3)
            gammas = np.ones(3)
            a_szs = np.zeros(3)

            baryon_model._init_stacker()
            baryon_model.stacker.dnumber_dlogmass = lambda : np.ones(
                (baryon_model.mus.size, baryon_model.zs.size)
            )

            ds = baryon_model.stacked_delta_sigma(radii, alphas, betas, gammas, a_szs)

            assert np.all(ds > 0)

    def describe_units():

        @pytest.fixture
        def baryon_model(mocker):
            mocker.patch('maszcal.lensing.ConModel', new=FakeConModel)
            mus = np.linspace(np.log(1e14), np.log(1e16), 9)
            zs = np.linspace(0, 1, 8)
            return maszcal.lensing.BaryonCmShearModel(mus, zs, units=u.Msun/u.pc**2)

        def it_has_correct_units(baryon_model):
            radii = np.logspace(-1, 1, 10)
            mus = np.log(1e14)*np.ones(2)
            alphas = np.ones(3)
            betas = 2*np.ones(3)
            gammas = np.ones(3)

            ds = baryon_model.delta_sigma_bary(radii, mus, alphas, betas, gammas)

            assert np.all(radii[None, None, :, None]*ds < 1e2)


class FakeProjector:
    @staticmethod
    def esd(rs, rho_func):
        rhos = rho_func(rs)
        return np.ones(rhos.shape)


def describe_BaryonShearModel():

    def describe_math():

        @pytest.fixture
        def baryon_model(mocker):
            mocker.patch('maszcal.lensing.projector', new=FakeProjector)
            mus = np.linspace(np.log(1e14), np.log(1e16), 9)
            zs = np.linspace(0, 1, 8)
            return maszcal.lensing.BaryonShearModel(mus, zs)

        def it_can_calculate_a_gnfw_rho(baryon_model):
            radii = np.logspace(-1, 1, 10)
            mus = np.log(1e14)*np.ones(2)
            cs = 3*np.ones(3)
            alphas = np.ones(3)
            betas = 2*np.ones(3)
            gammas = np.ones(3)

            rhos = baryon_model.rho_bary(radii, mus, cs, alphas, betas, gammas)

            assert np.all(rhos > 0)

        def it_can_calculate_an_nfw_rho(baryon_model):
            radii = np.logspace(-1, 1, 10)
            mus = np.log(1e14)*np.ones(1)
            cs = 3*np.ones(1)

            rhos = baryon_model.rho_cdm(radii, mus, cs)

            assert np.all(rhos > 0)

        def it_has_the_correct_baryon_fraction(baryon_model):
            rs = np.linspace(
                baryon_model.MIN_INTEGRATION_RADIUS,
                baryon_model.MAX_INTEGRATION_RADIUS,
                baryon_model.NUM_INTEGRATION_RADII
            )
            mus = np.log(1e14)*np.ones(1)
            cs = 3*np.ones(1)
            alphas = 0.88*np.ones(1)
            betas = 3.8*np.ones(1)
            gammas = 0.2*np.ones(1)

            rho_barys = baryon_model.rho_bary(rs, mus, cs, alphas, betas, gammas)
            rho_cdms = np.moveaxis(baryon_model.rho_cdm(rs, mus, cs), 2, 0)

            ratio = np.trapz(
                rho_barys * rs[:, None, None, None]**2,
                x=rs,
                axis=0
            ) / np.trapz(
                (rho_barys + rho_cdms) * rs[:, None, None, None]**2,
                x=rs,
                axis=0
            )

            f_b = baryon_model.baryon_frac
            assert np.allclose(ratio, f_b)

        def it_can_calculate_an_nfw_delta_sigma(baryon_model):
            radii = np.logspace(-1, 1, 10)
            mus = np.log(1e14)*np.ones(1)
            cs = 3*np.ones(1)

            ds = baryon_model.delta_sigma_cdm(radii, mus, cs)

            assert np.all(ds > 0)

        def it_can_calculate_a_gnfw_delta_sigma(baryon_model):
            radii = np.logspace(-1, 1, 10)
            mus = np.log(1e14)*np.ones(2)
            cs = 3*np.ones(3)
            alphas = np.ones(3)
            betas = 2*np.ones(3)
            gammas = np.ones(3)

            ds = baryon_model.delta_sigma_bary(radii, mus, cs, alphas, betas, gammas)

            assert np.all(ds > 0)

        def it_can_calculate_a_total_delta_sigma(baryon_model):
            radii = np.logspace(-1, 1, 10)
            mus = np.log(1e14)*np.ones(2)
            cs = 3*np.ones(3)
            alphas = np.ones(3)
            betas = 2*np.ones(3)
            gammas = np.ones(3)

            ds = baryon_model.delta_sigma_total(radii, mus, cs, alphas, betas, gammas)

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
                (baryon_model.mus.size, baryon_model.zs.size)
            )

            ds = baryon_model.stacked_delta_sigma(radii, cs, alphas, betas, gammas, a_szs)

            assert np.all(ds > 0)

    def describe_units():

        @pytest.fixture
        def baryon_model():
            mus = np.linspace(np.log(1e14), np.log(1e16), 9)
            zs = np.linspace(0, 1, 8)
            return maszcal.lensing.BaryonShearModel(mus, zs, units=u.Msun/u.pc**2)

        def it_has_correct_units(baryon_model):
            radii = np.logspace(-1, 1, 10)
            mus = np.log(1e14)*np.ones(2)
            cs = 3*np.ones(3)
            alphas = np.ones(3)
            betas = 2*np.ones(3)
            gammas = np.ones(3)

            ds = baryon_model.delta_sigma_bary(radii, mus, cs, alphas, betas, gammas)

            assert np.all(radii[None, None, :, None]*ds < 1e2)


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
                (stacker.mus.size, stacker.zs.size)
            )

            rs = np.logspace(-1, 1, 21)
            a_szs = np.linspace(-1, 1, 1)

            delta_sigmas_of_mass = np.ones((stacker.mus.size, zs.size, rs.size))

            delta_sigmas = stacker.stacked_delta_sigma(delta_sigmas_of_mass, rs, a_szs)

            precomp_delta_sigmas = np.ones((rs.size, 1))

            np.testing.assert_allclose(delta_sigmas, precomp_delta_sigmas)

        def it_can_handle_delta_sigmas_of_mass_with_different_params(stacker):
            N_PARAMS = 3
            rs = np.logspace(-1, 1, 21)
            a_szs = np.linspace(-1, 1, N_PARAMS)

            stacker.dnumber_dlogmass = lambda : np.ones(
                (stacker.mus.size, stacker.zs.size)
            )

            delta_sigmas_of_mass = np.ones((stacker.mus.size, stacker.zs.size, rs.size))

            delta_sigmas = stacker.stacked_delta_sigma(delta_sigmas_of_mass, rs, a_szs)

            assert delta_sigmas.shape == (21, N_PARAMS)

        def it_can_compute_wl_avg_masses(stacker):
            zs = np.linspace(0, 2, 8)

            stacker.dnumber_dlogmass = lambda : np.ones(
                (stacker.mus.size, stacker.zs.size)
            )

            a_szs = np.linspace(-1, 1, 4)

            avg_masses = stacker.weak_lensing_avg_mass(a_szs)

            assert avg_masses.shape == (4,)

        def it_complains_about_nans(stacker):
            zs = np.linspace(0, 2, 8)

            # Ugly mock of mass function
            stacker.dnumber_dlogmass = lambda : np.full(
                (stacker.mus.size, stacker.zs.size),
                np.nan,
            )

            rs = np.logspace(-1, 1, 10)
            a_szs = np.linspace(-1, 1, 2)
            delta_sigmas_of_mass = np.ones((stacker.mus.size, zs.size, rs.size))

            with pytest.raises(ValueError):
                stacker.stacked_delta_sigma(delta_sigmas_of_mass, rs, a_szs)


def describe_MiyatakeShearModel():

    def describe_math_functions():

        @pytest.fixture
        def stacked_model(mocker):
            mus = np.linspace(np.log(1e12), np.log(1e16), 20)
            zs = np.linspace(0, 2, 8)
            mocker.patch('maszcal.lensing.ConModel', new=FakeConModel)

            model = maszcal.lensing.MiyatakeShearModel(mus, zs)
            model._init_stacker()

            return model

        def it_computes_weak_lensing_avg_mass(stacked_model):
            stacked_model.stacker.dnumber_dlogmass = lambda : np.ones(
                (stacked_model.mus.size, stacked_model.zs.size)
            )

            a_szs = np.linspace(-1, 1, 1)

            avg_wl_mass = stacked_model.weak_lensing_avg_mass(a_szs)

            assert avg_wl_mass.shape == (1,)
            assert avg_wl_mass > 0

        def it_can_use_different_mass_definitions(mocker):
            mocker.patch('maszcal.lensing.ConModel', new=FakeConModel)
            rs = np.logspace(-1, 1, 10)

            mus = np.linspace(np.log(1e12), np.log(1e15), 20)
            zs = np.linspace(0, 2, 7)

            delta = 500
            mass_def = 'crit'
            model = maszcal.lensing.MiyatakeShearModel(mus, zs, delta=delta, mass_definition=mass_def)

            delta_sigs_500c = model.delta_sigma(rs, mus)

            delta = 200
            kind = 'mean'
            model = maszcal.lensing.MiyatakeShearModel(mus, zs, delta=delta, mass_definition=mass_def)

            delta_sigs_200m = model.delta_sigma(rs, mus)

            assert np.all(delta_sigs_200m < delta_sigs_500c)

        def it_computes_stacked_delta_sigmas_with_the_right_shape(stacked_model):
            stacked_model.stacker.dnumber_dlogmass = lambda : np.ones(
                (stacked_model.mus.size, stacked_model.zs.size)
            )

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
                (stacker.mus.size, stacker.zs.size)
            )

            rs = np.logspace(-1, 1, 21)
            a_szs = np.linspace(-1, 1, 1)

            delta_sigmas_of_mass = np.ones((stacker.mus.size, zs.size, rs.size))

            delta_sigmas = stacker.stacked_delta_sigma(delta_sigmas_of_mass, rs, a_szs)

            precomp_delta_sigmas = np.ones((rs.size, 1))

            np.testing.assert_allclose(delta_sigmas, precomp_delta_sigmas)

        def it_can_handle_delta_sigmas_of_mass_with_different_params(stacker):
            N_PARAMS = 3
            rs = np.logspace(-1, 1, 21)
            a_szs = np.linspace(-1, 1, N_PARAMS)

            stacker.dnumber_dlogmass = lambda : np.ones(
                (stacker.mus.size, stacker.zs.size)
            )

            delta_sigmas_of_mass = np.ones((stacker.mus.size, stacker.zs.size, rs.size))

            delta_sigmas = stacker.stacked_delta_sigma(delta_sigmas_of_mass, rs, a_szs)

            assert delta_sigmas.shape == (21, N_PARAMS)

        def it_complains_about_nans(stacker):
            zs = np.linspace(0, 2, 8)

            # Ugly mock of mass function
            stacker.dnumber_dlogmass = lambda : np.full(
                (stacker.mus.size, stacker.zs.size),
                np.nan,
            )

            # Ugly mock of concentration model
            stacker._m500c = lambda mus: np.exp(mus)

            rs = np.logspace(-1, 1, 10)
            a_szs = np.linspace(-1, 1, 2)
            delta_sigmas_of_mass = np.ones((stacker.mus.size, zs.size, rs.size))

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

        def it_computes_stacked_delta_sigma(mocker):
            mocker.patch('maszcal.lensing.ConModel', new=FakeConModel)

            mus = np.linspace(np.log(1e12), np.log(1e16), 20)
            zs = np.linspace(0, 2, 8)
            stacked_model = maszcal.lensing.NfwCmShearModel(mus, zs)
            stacked_model._init_stacker()

            stacked_model.stacker.dnumber_dlogmass = lambda : np.ones(
                (stacked_model.mus.size, stacked_model.zs.size)
            )

            rs = np.logspace(-1, 1, 4)
            a_szs = np.linspace(-1, 1, 3)

            avg_wl_mass = stacked_model.stacked_delta_sigma(rs, a_szs)

            assert avg_wl_mass.shape == (rs.size, a_szs.size)

        def it_computes_weak_lensing_avg_mass(mocker):
            mocker.patch('maszcal.lensing.ConModel', new=FakeConModel)

            mus = np.linspace(np.log(1e12), np.log(1e16), 20)
            zs = np.linspace(0, 2, 8)
            stacked_model = maszcal.lensing.NfwCmShearModel(mus, zs)
            stacked_model._init_stacker()

            stacked_model.stacker.dnumber_dlogmass = lambda : np.ones(
                (stacked_model.mus.size, stacked_model.zs.size)
            )

            a_szs = np.linspace(-1, 1, 3)

            avg_wl_mass = stacked_model.weak_lensing_avg_mass(a_szs)

            assert avg_wl_mass.shape == (3,)

        def it_can_use_different_mass_definitions(mocker):
            mocker.patch('maszcal.lensing.ConModel', new=FakeConModel)

            rs = np.logspace(-1, 1, 10)

            mus = np.linspace(np.log(1e12), np.log(1e15), 20)
            zs = np.linspace(0, 2, 7)

            delta = 500
            mass_def = 'crit'
            model = maszcal.lensing.NfwCmShearModel(mus, zs, delta=delta, mass_definition=mass_def)

            delta_sigs_500c = model.delta_sigma(rs, mus)

            assert delta_sigs_500c.shape == (20, 7, 10)

            delta = 200
            kind = 'mean'
            model = maszcal.lensing.NfwCmShearModel(mus, zs, delta=delta, mass_definition=mass_def)

            delta_sigs_200m = model.delta_sigma(rs, mus)

            assert np.all(delta_sigs_200m < delta_sigs_500c)


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
                (stacked_model.mus.size, stacked_model.zs.size)
            )

            a_szs = np.linspace(-1, 1, 1)

            avg_wl_mass = stacked_model.weak_lensing_avg_mass(a_szs)

            assert avg_wl_mass.shape == (1,)

        def it_can_use_different_mass_definitions():
            cons = np.array([2, 3, 4])
            rs = np.logspace(-1, 1, 10)

            mus = np.linspace(np.log(1e12), np.log(1e15), 20)
            zs = np.linspace(0, 2, 7)

            delta = 500
            mass_def = 'crit'
            model = maszcal.lensing.NfwShearModel(mus, zs, delta=delta, mass_definition=mass_def)

            delta_sigs_500c = model.delta_sigma(rs, mus, cons)

            delta = 200
            kind = 'mean'
            model = maszcal.lensing.NfwShearModel(mus, zs, delta=delta, mass_definition=mass_def)

            delta_sigs_200m = model.delta_sigma(rs, mus, cons)

            assert np.all(delta_sigs_200m < delta_sigs_500c)


def describe_SingleMassBaryonShearModel():

    def describe_math():

        @pytest.fixture
        def single_mass_model():
            redshifts = 0.4*np.ones(2)
            model = maszcal.lensing.SingleMassBaryonShearModel(redshifts=redshifts)
            return model

        def it_can_calculate_delta_sigma_of_mass(single_mass_model):
            base = np.ones(11)

            mus = np.log(1e15)*base
            cons = 3*base
            alphas = 0.88*base
            betas = 3.8*base
            gammas = 0.2*base
            rs = np.logspace(-1, 1, 5)

            delta_sigs = single_mass_model.delta_sigma(rs, mus, cons, alphas, betas, gammas)

            assert np.all(delta_sigs > 0)

        def it_can_use_different_units():
            redshifts = 0.4*np.ones(1)
            model = maszcal.lensing.SingleMassBaryonShearModel(redshifts=redshifts, units=u.Msun/u.Mpc**2)

            mu = np.array([np.log(1e15)])
            con = np.array([3])
            alpha = np.array([0.88])
            beta = np.array([3.8])
            gamma = np.array([0.2])
            rs = np.logspace(-1, 1, 5)

            delta_sigs = model.delta_sigma(rs, mu, con, alpha, beta, gamma)/1e12

            assert np.all(rs*delta_sigs < 1e6)

        def it_can_use_different_mass_definitions():
            mu = np.array([np.log(1e15)])
            con = np.array([3])
            alpha = np.array([0.88])
            beta = np.array([3.8])
            gamma = np.array([0.2])
            rs = np.logspace(-1, 1, 5)

            redshifts = 0.4*np.ones(1)
            delta = 500
            mass_def = 'crit'
            model = maszcal.lensing.SingleMassBaryonShearModel(redshifts=redshifts, delta=delta, mass_definition=mass_def)

            delta_sigs_500c = model.delta_sigma(rs, mu, con, alpha, beta, gamma)

            delta = 200
            kind = 'mean'
            model = maszcal.lensing.SingleMassBaryonShearModel(redshifts=redshifts, delta=delta, mass_definition=mass_def)

            delta_sigs_200m = model.delta_sigma(rs, mu, con, alpha, beta, gamma)

            assert np.all(delta_sigs_200m < delta_sigs_500c)


def describe_SingleMassNfwShearModel():

    def describe_math():

        @pytest.fixture
        def single_mass_model():
            redshift = 0.4*np.ones(1)
            model = maszcal.lensing.SingleMassNfwShearModel(redshifts=redshift)
            return model

        def it_can_calculate_delta_sigma_of_mass(single_mass_model):
            mu = np.array([np.log(1e15)])
            con = np.array([3])
            rs = np.logspace(-1, 1, 5)

            delta_sigs = single_mass_model.delta_sigma(rs, mu, con)

            assert np.all(delta_sigs > 0)
            assert delta_sigs.shape == (1, 1, 5, 1)

        def it_can_use_different_units():
            redshift = 0.4*np.ones(1)
            model = maszcal.lensing.SingleMassNfwShearModel(redshifts=redshift, units=u.Msun/u.Mpc**2)

            mu = np.array([np.log(1e15)])
            con = np.array([3])
            rs = np.logspace(-1, 1, 5)

            delta_sigs = model.delta_sigma(rs, mu, con)/1e12

            assert np.all(rs*delta_sigs < 1e6)

        def it_can_use_different_mass_definitions():
            mu = np.array([np.log(1e15)])
            con = np.array([3])
            rs = np.logspace(-1, 1, 5)

            redshift = 0.4*np.ones(1)
            delta = 500
            mass_def = 'crit'
            model = maszcal.lensing.SingleMassNfwShearModel(redshifts=redshift, delta=delta, mass_definition=mass_def)

            delta_sigs_500c = model.delta_sigma(rs, mu, con)

            delta = 200
            kind = 'mean'
            model = maszcal.lensing.SingleMassNfwShearModel(redshifts=redshift, delta=delta, mass_definition=mass_def)

            delta_sigs_200m = model.delta_sigma(rs, mu, con)

            assert np.all(delta_sigs_200m < delta_sigs_500c)


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
                (stacker.mus.size, stacker.zs.size)
            )

            rs = np.logspace(-1, 1, 21)
            cons = np.linspace(2, 4, 1)
            a_szs = np.linspace(-1, 1, 1)

            delta_sigmas_of_mass = np.ones((stacker.mus.size, zs.size, rs.size, cons.size))

            delta_sigmas = stacker.stacked_delta_sigma(delta_sigmas_of_mass, rs, a_szs)

            precomp_delta_sigmas = np.ones((rs.size, 1))

            np.testing.assert_allclose(delta_sigmas, precomp_delta_sigmas)

        def it_can_handle_delta_sigmas_of_mass_with_different_params(stacker):
            N_PARAMS = 3
            rs = np.logspace(-1, 1, 21)
            a_szs = np.linspace(-1, 1, N_PARAMS)

            stacker.dnumber_dlogmass = lambda : np.ones(
                (stacker.mus.size, stacker.zs.size)
            )

            delta_sigmas_of_mass = np.ones((stacker.mus.size, stacker.zs.size, rs.size, N_PARAMS))

            delta_sigmas = stacker.stacked_delta_sigma(delta_sigmas_of_mass, rs, a_szs)

        def it_complains_about_nans(stacker):
            zs = np.linspace(0, 2, 8)
            stacker.dnumber_dlogmass = lambda : np.full(
                (stacker.mus.size, stacker.zs.size),
                np.nan,
            )

            rs = np.logspace(-1, 1, 10)
            cons = np.linspace(2, 4, 1)
            a_szs = np.linspace(-1, 1, 1)
            delta_sigmas_of_mass = np.ones((stacker.mus.size, zs.size, rs.size, cons.size))

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

from dataclasses import dataclass
import pytest
import numpy as np
import astropy.units as u
import maszcal.lensing
import maszcal.cosmology


def fake_rho_total(rs, zs, mus, *params):
    return np.ones(rs.shape + zs.shape + (params[0].size,))


def fake_projector_esd(rs, rho_func):
    rhos = rho_func(rs)
    return np.ones(rhos.shape)


def describe_SingleMassConvergenceModel():

    def describe_angle_scale_distance():

        @pytest.fixture
        def model_comoving():
            model = maszcal.lensing.SingleMassConvergenceModel(
                rho_func=fake_rho_total,
                comoving=True,
                sd_func=fake_projector_esd,
            )
            return model

        @pytest.fixture
        def model_physical():
            model = maszcal.lensing.SingleMassConvergenceModel(
                rho_func=fake_rho_total,
                comoving=False,
                sd_func=fake_projector_esd,
            )
            return model

        def it_differs_between_comoving_and_noncomoving_cases(model_physical, model_comoving):
            zs = np.random.rand(4) + 0.1
            scale_physical = model_physical.angle_scale_distance(zs)
            scale_comoving = model_comoving.angle_scale_distance(zs)
            assert np.all(scale_physical != scale_comoving)

    def describe_math():

        @pytest.fixture
        def single_mass_model():
            model = maszcal.lensing.SingleMassConvergenceModel(
                rho_func=fake_rho_total,
                sd_func=fake_projector_esd,
            )
            return model

        def it_can_calculate_excess_surface_density_of_mass(single_mass_model):
            base = np.ones(11)

            zs= 0.4*np.ones(2)
            mus = np.log(1e15)*base
            cons = 3*base
            thetas = np.logspace(-4, -1, 5)

            convergences = single_mass_model.convergence(thetas, zs, mus, cons)

            assert convergences.shape == (5, 2, 11)


def describe_SingleMassShearModel():

    def describe_math():

        @pytest.fixture
        def single_mass_model():
            redshifts = 0.4*np.ones(2)
            model = maszcal.lensing.SingleMassShearModel(
                redshifts=redshifts,
                rho_func=fake_rho_total,
                esd_func=fake_projector_esd,
            )
            return model

        def it_can_calculate_excess_surface_density_of_mass(single_mass_model):
            base = np.ones(11)

            mus = np.log(1e15)*base
            cons = 3*base
            alphas = 0.88*base
            betas = 3.8*base
            gammas = 0.2*base
            rs = np.logspace(-1, 1, 5)

            delta_sigs = single_mass_model.excess_surface_density(rs, mus, cons, alphas, betas, gammas)

            assert delta_sigs.shape == (5, 2, 11)

        def it_can_use_different_units():
            redshifts = 0.4*np.ones(1)
            model = maszcal.lensing.SingleMassShearModel(
                redshifts=redshifts,
                rho_func=fake_rho_total,
                units=u.Msun/u.Mpc**2,
                esd_func=fake_projector_esd,
            )

            zs = 0.4*np.ones(2)
            mu = np.array([np.log(1e15)])
            con = np.array([3])
            alpha = np.array([0.88])
            beta = np.array([3.8])
            gamma = np.array([0.2])
            rs = np.logspace(-1, 1, 5)

            delta_sigs = model.excess_surface_density(rs, mu, con, alpha, beta, gamma)/1e12

            assert np.all(rs*delta_sigs < 1e6)


def describe_SingleMassNfwShearModel():

    def describe_math():

        @pytest.fixture
        def single_mass_model():
            redshift = 0.4*np.ones(1)
            model = maszcal.lensing.SingleMassNfwShearModel(redshifts=redshift)
            return model

        def it_can_calculate_excess_surface_density_of_mass(single_mass_model):
            mu = np.array([np.log(1e15)])
            con = np.array([3])
            rs = np.logspace(-1, 1, 5)

            delta_sigs = single_mass_model.excess_surface_density(rs, mu, con)

            assert np.all(delta_sigs > 0)
            assert delta_sigs.shape == (1, 5, 1)

        def it_can_use_different_units():
            redshift = 0.4*np.ones(1)
            model = maszcal.lensing.SingleMassNfwShearModel(redshifts=redshift, units=u.Msun/u.Mpc**2)

            mu = np.array([np.log(1e15)])
            con = np.array([3])
            rs = np.logspace(-1, 1, 5)

            delta_sigs = model.excess_surface_density(rs, mu, con)/1e12

            assert np.all(rs*delta_sigs < 1e6)

        def it_can_use_different_mass_definitions():
            mu = np.array([np.log(1e15)])
            con = np.array([3])
            rs = np.logspace(-1, 1, 5)

            redshift = 0.4*np.ones(1)
            delta = 500
            mass_def = 'crit'
            model = maszcal.lensing.SingleMassNfwShearModel(redshifts=redshift, delta=delta, mass_definition=mass_def)

            delta_sigs_500c = model.excess_surface_density(rs, mu, con)

            delta = 200
            kind = 'mean'
            model = maszcal.lensing.SingleMassNfwShearModel(redshifts=redshift, delta=delta, mass_definition=mass_def)

            delta_sigs_200m = model.excess_surface_density(rs, mu, con)

            assert np.all(delta_sigs_200m < delta_sigs_500c)

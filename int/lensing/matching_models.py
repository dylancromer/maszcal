import pytest
import numpy as np
import astropy.units as u
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import seaborn as sns
sns.set(style='whitegrid', font_scale=1.5, rc={"lines.linewidth": 2,'lines.markersize': 8.0,})
import maszcal.concentration
import maszcal.density
import maszcal.lensing
import maszcal.cosmology


def describe_MatchingConvergenceModel():

    def describe_stacked_kappa():

        @pytest.fixture
        def density_model():
            return maszcal.density.MatchingGnfw(
                cosmo_params=maszcal.cosmology.CosmoParams(),
                mass_definition='mean',
                delta=200,
                units=u.Msun/u.pc**2,
                comoving_radii=True,
                nfw_class=maszcal.density.MatchingNfwModel,
            )

        @pytest.fixture
        def convergence_model(density_model):
            NUM_CLUSTERS = 1
            sz_masses = 2e13*np.random.randn(NUM_CLUSTERS) + 2e14
            zs = np.random.rand(NUM_CLUSTERS)
            weights = np.ones(NUM_CLUSTERS)
            cosmo_params = maszcal.cosmology.CosmoParams()
            return maszcal.lensing.MatchingConvergenceModel(
                sz_masses=sz_masses,
                redshifts=zs,
                lensing_weights=weights,
                cosmo_params=cosmo_params,
                rho_func=density_model.rho_tot,
            )

        def the_plots_look_right(convergence_model):
            from_arcmin = 2 * np.pi / 360 / 60
            to_arcmin = 1/from_arcmin
            thetas = np.geomspace(0.05*from_arcmin, 60*from_arcmin, 60)
            cons = 3*np.ones(1)
            alphas = 0.5*np.ones(1)
            betas = np.linspace(2.8, 3.2, 3)
            gammas = 0.5*np.ones(1)
            a_szs = np.array([0, 0.1, -0.1, 0.01])

            sds = convergence_model.stacked_kappa(thetas, a_szs, cons, alphas, betas, gammas)

            plt.plot(thetas*to_arcmin, thetas[:, None]*sds[..., 0])
            plt.xscale('log')
            plt.xlabel(r'$\theta$')
            plt.ylabel(r'$\theta \; \kappa(\theta)$')
            plt.savefig('figs/test/matching_stacked_gnfw_theta_times_kappa.svg')
            plt.gcf().clear()

            plt.plot(thetas*to_arcmin, sds[..., 0])
            plt.xscale('log')
            plt.xlabel(r'$\theta$')
            plt.ylabel(r'$\kappa(\theta)$')
            plt.savefig('figs/test/matching_stacked_gnfw_kappa.svg')
            plt.gcf().clear()


def describe_MatchingShearModel():

    def describe_stacked_delta_sigma():

        @pytest.fixture
        def density_model():
            return maszcal.density.MatchingGnfw(
                cosmo_params=maszcal.cosmology.CosmoParams(),
                mass_definition='mean',
                delta=200,
                units=u.Msun/u.pc**2,
                comoving_radii=True,
                nfw_class=maszcal.density.MatchingNfwModel,
            )

        @pytest.fixture
        def shear_model(density_model):
            NUM_CLUSTERS = 10
            sz_masses = 2e13*np.random.randn(NUM_CLUSTERS) + 2e14
            zs = np.random.rand(NUM_CLUSTERS)
            weights = np.random.rand(NUM_CLUSTERS)
            cosmo_params = maszcal.cosmology.CosmoParams()
            return maszcal.lensing.MatchingShearModel(
                sz_masses=sz_masses,
                redshifts=zs,
                lensing_weights=weights,
                rho_func=density_model.rho_tot,
            )

        def the_plots_look_right(shear_model):
            radii = np.logspace(-1, 1, 30)
            cons = 3*np.ones(1)
            alphas = 0.8*np.ones(1)
            betas = np.linspace(2.8, 3.2, 3)
            gammas = 0.2*np.ones(1)
            a_szs = 0.3*np.ones(2)

            esds = shear_model.stacked_delta_sigma(radii, a_szs, cons, alphas, betas, gammas)

            plt.plot(radii, radii[:, None]*esds[:, 0, :])
            plt.xscale('log')

            plt.xlabel(r'$R$')
            plt.ylabel(r'$R \Delta\Sigma(R)$')

            plt.savefig('figs/test/matching_stacked_gnfw_delta_sigma.svg')
            plt.gcf().clear()


def describe_MatchingCmShearModel():

    def describe_stacked_delta_sigma():

        @pytest.fixture
        def density_model():
            return maszcal.density.MatchingCmGnfw(
                cosmo_params=maszcal.cosmology.CosmoParams(),
                mass_definition='mean',
                delta=200,
                units=u.Msun/u.pc**2,
                comoving_radii=True,
                nfw_class=maszcal.density.MatchingCmNfwModel,
                con_class=maszcal.concentration.MatchingConModel,
            )

        @pytest.fixture
        def shear_model(density_model):
            NUM_CLUSTERS = 10
            sz_masses = 2e13*np.random.randn(NUM_CLUSTERS) + 2e14
            zs = np.random.rand(NUM_CLUSTERS)
            weights = np.random.rand(NUM_CLUSTERS)
            cosmo_params = maszcal.cosmology.CosmoParams()
            return maszcal.lensing.MatchingShearModel(
                sz_masses=sz_masses,
                redshifts=zs,
                lensing_weights=weights,
                rho_func=density_model.rho_tot,
            )

        def the_plots_look_right(shear_model):
            radii = np.logspace(-1, 1, 30)
            alphas = 0.5*np.ones(1)
            betas = np.linspace(2.8, 3.2, 3)
            gammas = 0.5*np.ones(1)
            a_szs = 0.3*np.ones(1)

            esds = shear_model.stacked_delta_sigma(radii, a_szs, alphas, betas, gammas)

            plt.plot(radii, radii[:, None]*esds[:, 0, :])
            plt.xscale('log')

            plt.xlabel(r'$R$')
            plt.ylabel(r'$R \Delta\Sigma(R)$')

            plt.savefig('figs/test/matching_cm_stacked_gnfw_delta_sigma.svg')
            plt.gcf().clear()


def describe_miscentered_MatchingShearModel():

    def describe_stacked_kappa():

        @pytest.fixture
        def density_model():
            return maszcal.density.MatchingMiscenteredGnfw(
                cosmo_params=maszcal.cosmology.CosmoParams(),
                mass_definition='mean',
                delta=200,
                units=u.Msun/u.pc**2,
                comoving_radii=True,
                nfw_class=maszcal.density.MatchingNfwModel,
            )

        @pytest.fixture
        def shear_model(density_model):
            NUM_CLUSTERS = 10
            sz_masses = 2e13*np.random.randn(NUM_CLUSTERS) + 2e14
            zs = np.random.rand(NUM_CLUSTERS)
            weights = np.random.rand(NUM_CLUSTERS)
            cosmo_params = maszcal.cosmology.CosmoParams()
            return maszcal.lensing.MatchingShearModel(
                sz_masses=sz_masses,
                redshifts=zs,
                lensing_weights=weights,
                rho_func=density_model.rho_tot,
            )

        def the_plots_look_right(shear_model):
            radii = np.logspace(-1, 1, 30)
            cons = 3*np.ones(1)
            alphas = 0.8*np.ones(1)
            betas = 3.8*np.ones(1)
            gammas = 0.2*np.ones(1)
            miscenter_scales = np.logspace(-2, np.log10(2e-1), 3)
            a_szs = 0*np.ones(1)

            esds = shear_model.stacked_delta_sigma(radii, a_szs, cons, alphas, betas, gammas, miscenter_scales)

            plt.plot(radii, radii[:, None]*esds[:, 0, :])
            plt.xscale('log')

            plt.xlabel(r'$R$')
            plt.ylabel(r'$R \Delta\Sigma(R)$')

            plt.savefig('figs/test/miscentered_matching_stacked_gnfw_delta_sigma.svg')
            plt.gcf().clear()

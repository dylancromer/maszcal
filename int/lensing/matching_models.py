import pytest
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import seaborn as sns
sns.set(style='whitegrid', font_scale=1.5, rc={"lines.linewidth": 2,'lines.markersize': 8.0,})
import meso
import maszcal.stats
import maszcal.concentration
import maszcal.density
import maszcal.lensing
import maszcal.cosmology


def describe_MatchingConvergenceModel():

    def describe_comoving_coordinates():

        @pytest.fixture
        def comov_density_model():
            return maszcal.density.MatchingGnfw(
                cosmo_params=maszcal.cosmology.CosmoParams(),
                mass_definition='mean',
                delta=200,
                comoving_radii=True,
                nfw_class=maszcal.density.MatchingNfwModel,
            )

        @pytest.fixture
        def comov_convergence_model(comov_density_model):
            NUM_CLUSTERS = 1
            sz_masses = 2e14*np.ones(1)
            zs = np.ones(1)
            weights = np.ones(NUM_CLUSTERS)
            cosmo_params = maszcal.cosmology.CosmoParams()
            return maszcal.lensing.MatchingConvergenceModel(
                sz_masses=sz_masses,
                redshifts=zs,
                lensing_weights=weights,
                cosmo_params=cosmo_params,
                comoving=True,
                rho_func=comov_density_model.rho_tot,
            )

        @pytest.fixture
        def phys_density_model():
            return maszcal.density.MatchingGnfw(
                cosmo_params=maszcal.cosmology.CosmoParams(),
                mass_definition='mean',
                delta=200,
                comoving_radii=False,
                nfw_class=maszcal.density.MatchingNfwModel,
            )

        @pytest.fixture
        def phys_convergence_model(phys_density_model):
            NUM_CLUSTERS = 1
            sz_masses = 2e14*np.ones(1)
            zs = np.ones(1)
            weights = np.ones(NUM_CLUSTERS)
            cosmo_params = maszcal.cosmology.CosmoParams()
            return maszcal.lensing.MatchingConvergenceModel(
                sz_masses=sz_masses,
                redshifts=zs,
                lensing_weights=weights,
                cosmo_params=cosmo_params,
                comoving=False,
                rho_func=phys_density_model.rho_tot,
            )


        def it_matches_no_matter_which_frame_you_use(comov_convergence_model, phys_convergence_model):
            from_arcmin = 2 * np.pi / 360 / 60
            to_arcmin = 1/from_arcmin
            thetas = np.geomspace(0.05*from_arcmin, 60*from_arcmin, 60)
            cons = 3*np.ones(1)
            alphas = 0.5*np.ones(1)
            betas = np.linspace(2.8, 3.2, 3)
            gammas = 0.5*np.ones(1)
            a_szs = np.array([0, 0.1, -0.1, 0.01])

            sds_comov = comov_convergence_model.stacked_kappa(thetas, a_szs, cons, alphas, betas, gammas)
            sds_phys = phys_convergence_model.stacked_kappa(thetas, a_szs, cons, alphas, betas, gammas)

            assert np.allclose(sds_comov, sds_phys, rtol=1e-1)

    def describe_stacked_kappa():

        @pytest.fixture
        def density_model():
            return maszcal.density.MatchingGnfw(
                cosmo_params=maszcal.cosmology.CosmoParams(),
                mass_definition='mean',
                delta=200,
                comoving_radii=True,
                nfw_class=maszcal.density.MatchingNfwModel,
            )

        @pytest.fixture
        def convergence_model(density_model):
            NUM_CLUSTERS = 1
            rng = np.random.default_rng(seed=13)
            sz_masses = 2e13*rng.normal(size=NUM_CLUSTERS) + 2e14
            zs = rng.random(size=NUM_CLUSTERS) + 0.01
            weights = rng.random(size=NUM_CLUSTERS)
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


def describe_ScatteredMatchingConvergenceModel():

    def describe_stacked_kappa():

        @pytest.fixture
        def density_model():
            return maszcal.density.Gnfw(
                cosmo_params=maszcal.cosmology.CosmoParams(),
                mass_definition='mean',
                delta=200,
                comoving_radii=True,
                nfw_class=maszcal.density.NfwModel,
            )

        @pytest.fixture
        def hmf_interp():
            return maszcal.tinker.HmfInterpolator(
                mu_samples=np.log(np.geomspace(1e12, 1e16, 600)),
                redshift_samples=np.linspace(0.01, 4, 120),
                delta=200,
                mass_definition='mean',
                cosmo_params=maszcal.cosmology.CosmoParams(),
            )

        @pytest.fixture
        def convergence_model(density_model, hmf_interp):
            NUM_CLUSTERS = 1
            rng = np.random.default_rng(seed=13)
            sz_masses = 2e13*rng.normal(size=NUM_CLUSTERS) + 2e14
            zs = rng.random(size=NUM_CLUSTERS) + 0.01
            weights = rng.random(size=NUM_CLUSTERS)
            cosmo_params = maszcal.cosmology.CosmoParams()
            return maszcal.lensing.ScatteredMatchingConvergenceModel(
                sz_masses=sz_masses,
                redshifts=zs,
                lensing_weights=weights,
                cosmo_params=cosmo_params,
                rho_func=density_model.rho_tot,
                logmass_prob_dist_func=hmf_interp,
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
            plt.savefig('figs/test/scattered_matching_stacked_gnfw_theta_times_kappa.svg')
            plt.gcf().clear()

            plt.plot(thetas*to_arcmin, sds[..., 0])
            plt.xscale('log')
            plt.xlabel(r'$\theta$')
            plt.ylabel(r'$\kappa(\theta)$')
            plt.savefig('figs/test/scattered_matching_stacked_gnfw_kappa.svg')
            plt.gcf().clear()


def describe_MatchingShearModel():

    def describe_stacked_delta_sigma():

        @pytest.fixture
        def density_model():
            return maszcal.density.MatchingGnfw(
                cosmo_params=maszcal.cosmology.CosmoParams(),
                mass_definition='mean',
                delta=200,
                comoving_radii=True,
                nfw_class=maszcal.density.MatchingNfwModel,
            )

        @pytest.fixture
        def shear_model(density_model):
            NUM_CLUSTERS = 100
            rng = np.random.default_rng(seed=13)
            sz_masses = 2e13*rng.normal(size=NUM_CLUSTERS) + 2e14
            zs = rng.random(size=NUM_CLUSTERS) + 0.01
            weights = rng.random(size=NUM_CLUSTERS)
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
            betas = 3.4*np.ones(1)
            gammas = 0.2*np.ones(1)
            a_szs = np.linspace(0, 0.5, 4)

            esds = shear_model.stacked_delta_sigma(radii, a_szs, cons, alphas, betas, gammas)

            plt.plot(radii, radii[:, None]*esds[:, :, 0])
            plt.xscale('log')

            plt.xlabel(r'$R$')
            plt.ylabel(r'$R \Delta\Sigma(R)$')

            plt.savefig('figs/test/matching_stacked_gnfw_delta_sigma.svg')
            plt.gcf().clear()


def describe_ScatteredMatchingShearModel():

    def describe_stacked_delta_sigma():

        @pytest.fixture
        def density_model():
            return maszcal.density.Gnfw(
                cosmo_params=maszcal.cosmology.CosmoParams(),
                mass_definition='mean',
                delta=200,
                comoving_radii=True,
                nfw_class=maszcal.density.NfwModel,
            )

        @pytest.fixture
        def hmf_interp():
            return maszcal.tinker.HmfInterpolator(
                mu_samples=np.log(np.geomspace(1e12, 1e16, 600)),
                redshift_samples=np.linspace(0.01, 4, 120),
                delta=200,
                mass_definition='mean',
                cosmo_params=maszcal.cosmology.CosmoParams(),
            )

        @pytest.fixture
        def shear_model(density_model, hmf_interp):
            NUM_CLUSTERS = 100
            rng = np.random.default_rng(seed=13)
            sz_masses = 2e13*rng.normal(size=NUM_CLUSTERS) + 2e14
            zs = rng.random(size=NUM_CLUSTERS) + 0.01
            weights = rng.random(size=NUM_CLUSTERS)
            return maszcal.lensing.ScatteredMatchingShearModel(
                sz_masses=sz_masses,
                redshifts=zs,
                lensing_weights=weights,
                rho_func=density_model.rho_tot,
                logmass_prob_dist_func=hmf_interp,
            )

        def the_plots_look_right(shear_model):
            radii = np.logspace(-1, 1, 30)
            cons = 3*np.ones(1)
            alphas = 0.8*np.ones(1)
            betas = 3.4*np.ones(1)
            gammas = 0.2*np.ones(1)
            a_szs = np.linspace(0, 0.5, 4)

            esds = shear_model.stacked_delta_sigma(radii, a_szs, cons, alphas, betas, gammas)

            plt.plot(radii, radii[:, None]*esds[:, :, 0])
            plt.xscale('log')

            plt.xlabel(r'$R$')
            plt.ylabel(r'$R \Delta\Sigma(R)$')

            plt.savefig('figs/test/scattered_matching_stacked_gnfw_delta_sigma.svg')
            plt.gcf().clear()


def describe_MatchingCmShearModel():

    def describe_stacked_delta_sigma():

        @pytest.fixture
        def density_model():
            return maszcal.density.MatchingCmGnfw(
                cosmo_params=maszcal.cosmology.CosmoParams(),
                mass_definition='mean',
                delta=200,
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
            return maszcal.density.MatchingGnfw(
                cosmo_params=maszcal.cosmology.CosmoParams(),
                mass_definition='mean',
                delta=200,
                comoving_radii=True,
                nfw_class=maszcal.density.MatchingNfwModel,
            )

        @pytest.fixture
        def miscentering(density_model):
            return maszcal.lensing.Miscentering(
                rho_func=density_model.rho_tot,
                misc_distrib=maszcal.stats.MiscenteringDistributions.rayleigh_dist,
                miscentering_func=meso.Rho().miscenter,
            )

        @pytest.fixture
        def miscentered_rho_func(miscentering):
            def _misc_rho_func(radii, *params):
                misc_params = params[-2:]
                rho_params = params[:-2]
                return miscentering.rho(radii, misc_params, rho_params)
            return _misc_rho_func

        @pytest.fixture
        def shear_model(miscentered_rho_func):
            NUM_CLUSTERS = 10
            sz_masses = 2e13*np.random.randn(NUM_CLUSTERS) + 2e14
            zs = np.random.rand(NUM_CLUSTERS)
            weights = np.random.rand(NUM_CLUSTERS)
            cosmo_params = maszcal.cosmology.CosmoParams()
            return maszcal.lensing.MatchingShearModel(
                sz_masses=sz_masses,
                redshifts=zs,
                lensing_weights=weights,
                rho_func=miscentered_rho_func,
            )

        def the_plots_look_right(shear_model):
            radii = np.logspace(-1, 1, 30)
            cons = 3*np.ones(1)
            alphas = 0.8*np.ones(1)
            betas = 3.8*np.ones(1)
            gammas = 0.2*np.ones(1)
            miscenter_scales = 1e-1*np.ones(1)
            centering_probs = np.linspace(0, 1, 3)
            a_szs = 0*np.ones(1)

            esds = shear_model.stacked_delta_sigma(radii, a_szs, cons, alphas, betas, gammas, miscenter_scales, centering_probs)

            plt.plot(radii, radii[:, None]*esds[:, 0, :])
            plt.xscale('log')

            plt.xlabel(r'$R$')
            plt.ylabel(r'$R \Delta\Sigma(R)$')

            plt.savefig('figs/test/miscentered_matching_stacked_gnfw_delta_sigma.svg')
            plt.gcf().clear()


def describe_miscentered_MatchingConvergenceModel():

    def describe_stacked_kappa():

        @pytest.fixture
        def density_model():
            return maszcal.density.MatchingGnfw(
                cosmo_params=maszcal.cosmology.CosmoParams(),
                mass_definition='mean',
                delta=200,
                comoving_radii=True,
                nfw_class=maszcal.density.MatchingNfwModel,
            )

        @pytest.fixture
        def miscentering(density_model):
            return maszcal.lensing.Miscentering(
                rho_func=density_model.rho_tot,
                misc_distrib=maszcal.stats.MiscenteringDistributions.rayleigh_dist,
                miscentering_func=meso.Rho().miscenter,
            )

        @pytest.fixture
        def miscentered_rho_func(miscentering):
            def _misc_rho_func(radii, *params):
                misc_params = params[-2:]
                rho_params = params[:-2]
                return miscentering.rho(radii, misc_params, rho_params)
            return _misc_rho_func

        @pytest.fixture
        def convergence_model(miscentered_rho_func):
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
                rho_func=miscentered_rho_func,
            )

        def the_plots_look_right(convergence_model):
            from_arcmin = 2 * np.pi / 360 / 60
            to_arcmin = 1/from_arcmin
            thetas = np.geomspace(0.05*from_arcmin, 60*from_arcmin, 60)
            cons = 3*np.ones(1)
            alphas = 0.8*np.ones(1)
            betas = 3.8*np.ones(1)
            gammas = 0.2*np.ones(1)
            miscenter_scales = 1e-1*np.ones(1)
            centering_probs = np.linspace(0, 1, 3)
            a_szs = 0*np.ones(1)

            sds = convergence_model.stacked_kappa(thetas, a_szs, cons, alphas, betas, gammas, miscenter_scales, centering_probs)

            plt.plot(thetas*to_arcmin, thetas[:, None]*sds[:, 0, :])
            plt.xscale('log')
            plt.xlabel(r'$\theta$')
            plt.ylabel(r'$\theta \; \kappa(\theta)$')
            plt.savefig('figs/test/miscentered_matching_stacked_gnfw_theta_times_kappa.svg')
            plt.gcf().clear()

            plt.plot(thetas*to_arcmin, sds[:, 0, :])
            plt.xscale('log')
            plt.xlabel(r'$\theta$')
            plt.ylabel(r'$\kappa(\theta)$')
            plt.savefig('figs/test/miscentered_matching_stacked_gnfw_kappa.svg')
            plt.gcf().clear()

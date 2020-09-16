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
import meso
import projector
import supercubos
import maszcal.corrections
import maszcal.emulate
import maszcal.twohalo
import maszcal.stats
import maszcal.concentration
import maszcal.density
import maszcal.lensing
import maszcal.cosmology


def describe_2HaloCorrected_MatchingConvergenceModel():

    def describe_stacked_convergence():

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
        def two_halo_conv():
            cosmo = maszcal.cosmology.CosmoParams()
            model = maszcal.twohalo.TwoHaloConvergenceModel(cosmo_params=cosmo)
            return model.radius_space_convergence

        @pytest.fixture
        def conv_emulator(two_halo_conv):
            return maszcal.twohalo.TwoHaloEmulator(
                two_halo_func=two_halo_conv,
                r_grid=np.geomspace(0.0001, 60, 160),
                z_lims=np.array([0, 1.2]),
                mu_lims=np.log(np.array([1e13, 1e15])),
                num_emulator_samples=800,
            )

        @pytest.fixture
        def corrected_lensing_func(density_model, conv_emulator):
            return maszcal.corrections.Matching2HaloCorrection(
                one_halo_func=density_model.convergence,
                two_halo_func=conv_emulator,
            ).corrected_profile

        @pytest.fixture
        def convergence_model(corrected_lensing_func):
            NUM_CLUSTERS = 100
            rng = np.random.default_rng(seed=13)
            sz_masses = 2e13*rng.normal(size=NUM_CLUSTERS) + 2e14
            zs = rng.random(size=NUM_CLUSTERS) + 0.01
            weights = rng.random(size=NUM_CLUSTERS)
            cosmo_params = maszcal.cosmology.CosmoParams()
            return maszcal.lensing.MatchingConvergenceModel(
                sz_masses=sz_masses,
                redshifts=zs,
                lensing_weights=weights,
                lensing_func=corrected_lensing_func,
                cosmo_params=cosmo_params,
            )

        def the_plots_look_right(convergence_model):
            from_arcmin = 2 * np.pi / 360 / 60
            to_arcmin = 1/from_arcmin
            thetas = np.geomspace(0.05*from_arcmin, 60*from_arcmin, 60)
            cons = 3*np.ones(1)
            alphas = 0.5*np.ones(1)
            betas = 3.6*np.ones(1)
            gammas = 0.5*np.ones(1)
            a_szs = -0.3*np.ones(1)
            a_2hs = np.linspace(0, 1, 3)

            sds = convergence_model.stacked_convergence(thetas, a_szs, a_2hs, cons, alphas, betas, gammas)

            plt.plot(thetas*to_arcmin, thetas[:, None]*sds[:, 0, :])
            plt.xscale('log')
            plt.xlabel(r'$\theta$')
            plt.ylabel(r'$\theta \; \kappa(\theta)$')
            plt.savefig('figs/test/2halo_corrected_matching_stacked_gnfw_theta_times_convergence.svg')
            plt.gcf().clear()

            plt.plot(thetas*to_arcmin, sds[:, 0, :])
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$\theta$')
            plt.ylabel(r'$\kappa(\theta)$')
            plt.savefig('figs/test/2halo_corrected_matching_stacked_gnfw_convergence.svg')
            plt.gcf().clear()


def describe_2HaloCorrected_ScatteredMatchingConvergenceModel():

    def describe_stacked_convergence():

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
        def two_halo_conv():
            cosmo = maszcal.cosmology.CosmoParams()
            model = maszcal.twohalo.TwoHaloConvergenceModel(cosmo_params=cosmo)
            return model.radius_space_convergence

        @pytest.fixture
        def conv_emulator(two_halo_conv):
            return maszcal.twohalo.TwoHaloEmulator(
                two_halo_func=two_halo_conv,
                r_grid=np.geomspace(0.0001, 60, 160),
                z_lims=np.array([0, 1.2]),
                mu_lims=np.log(np.array([1e13, 1e15])),
                num_emulator_samples=800,
                separate_mu_and_z_axes=True,
            )

        @pytest.fixture
        def corrected_lensing_func(density_model, conv_emulator):
            return maszcal.corrections.TwoHaloCorrection(
                one_halo_func=density_model.convergence,
                two_halo_func=conv_emulator,
            ).corrected_profile

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
        def convergence_model(corrected_lensing_func, hmf_interp):
            NUM_CLUSTERS = 100
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
                lensing_func=corrected_lensing_func,
                logmass_prob_dist_func=hmf_interp,
            )

        def the_plots_look_right(convergence_model):
            from_arcmin = 2 * np.pi / 360 / 60
            to_arcmin = 1/from_arcmin
            thetas = np.geomspace(0.05*from_arcmin, 60*from_arcmin, 60)
            cons = 3*np.ones(1)
            alphas = 0.5*np.ones(1)
            betas = 3.6*np.ones(1)
            gammas = 0.5*np.ones(1)
            a_szs = -0.3*np.ones(1)
            a_2hs = np.linspace(0, 1, 3)

            sds = convergence_model.stacked_convergence(thetas, a_szs, a_2hs, cons, alphas, betas, gammas).squeeze()

            plt.plot(thetas*to_arcmin, thetas[:, None]*sds)
            plt.xscale('log')
            plt.xlabel(r'$\theta$')
            plt.ylabel(r'$\theta \; \kappa(\theta)$')
            plt.savefig('figs/test/2halo_corrected_scattered_matching_stacked_gnfw_theta_times_convergence.svg')
            plt.gcf().clear()

            plt.plot(thetas*to_arcmin, sds)
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$\theta$')
            plt.ylabel(r'$\kappa(\theta)$')
            plt.savefig('figs/test/2halo_corrected_scattered_matching_stacked_gnfw_convergence.svg')
            plt.gcf().clear()


def describe_2HaloCorrected_MatchingShearModel():

    def describe_stacked_excess_surface_density():

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
        def two_halo_esd():
            cosmo = maszcal.cosmology.CosmoParams()
            model = maszcal.twohalo.TwoHaloShearModel(cosmo_params=cosmo)
            return model.excess_surface_density

        @pytest.fixture
        def esd_emulator(two_halo_esd):
            return maszcal.twohalo.TwoHaloEmulator(
                two_halo_func=two_halo_esd,
                r_grid=np.geomspace(0.01, 100, 120),
                z_lims=np.array([0, 1.2]),
                mu_lims=np.log(np.array([1e13, 1e15])),
                num_emulator_samples=800,
            )

        @pytest.fixture
        def corrected_lensing_func(density_model, esd_emulator):
            return maszcal.corrections.Matching2HaloCorrection(
                one_halo_func=density_model.excess_surface_density,
                two_halo_func=esd_emulator,
            ).corrected_profile

        @pytest.fixture
        def shear_model(corrected_lensing_func):
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
                lensing_func=corrected_lensing_func,
            )

        def the_plots_look_right(shear_model):
            radii = np.geomspace(1e-1, 60, 99)
            cons = 3*np.ones(1)
            alphas = 0.8*np.ones(1)
            betas = 3.4*np.ones(1)
            gammas = 0.2*np.ones(1)
            a_szs = -0.3*np.ones(1)
            a_2hs = np.linspace(0, 1, 3)

            esds = shear_model.stacked_excess_surface_density(radii, a_szs, a_2hs, cons, alphas, betas, gammas)

            plt.plot(radii, radii[:, None]*esds.squeeze())
            plt.xscale('log')

            plt.xlabel(r'$R$')
            plt.ylabel(r'$R \Delta\Sigma(R)$')

            plt.savefig('figs/test/2halo_corrected_matching_stacked_gnfw_excess_surface_density.svg')
            plt.gcf().clear()


def describe_2HaloCorrected_ScatteredMatchingShearModel():

    def describe_stacked_excess_surface_density():

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
        def two_halo_esd():
            cosmo = maszcal.cosmology.CosmoParams()
            model = maszcal.twohalo.TwoHaloShearModel(cosmo_params=cosmo)
            return model.excess_surface_density

        @pytest.fixture
        def esd_emulator(two_halo_esd):
            return maszcal.twohalo.TwoHaloEmulator(
                two_halo_func=two_halo_esd,
                r_grid=np.geomspace(0.01, 100, 120),
                z_lims=np.array([0, 1.2]),
                mu_lims=np.log(np.array([1e13, 1e15])),
                num_emulator_samples=800,
                separate_mu_and_z_axes=True,
            )

        @pytest.fixture
        def corrected_lensing_func(density_model, esd_emulator):
            return maszcal.corrections.TwoHaloCorrection(
                one_halo_func=density_model.excess_surface_density,
                two_halo_func=esd_emulator,
            ).corrected_profile

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
        def shear_model(corrected_lensing_func, hmf_interp):
            NUM_CLUSTERS = 100
            rng = np.random.default_rng(seed=13)
            sz_masses = 2e13*rng.normal(size=NUM_CLUSTERS) + 2e14
            zs = rng.random(size=NUM_CLUSTERS) + 0.01
            weights = rng.random(size=NUM_CLUSTERS)
            return maszcal.lensing.ScatteredMatchingShearModel(
                sz_masses=sz_masses,
                redshifts=zs,
                lensing_weights=weights,
                lensing_func=corrected_lensing_func,
                logmass_prob_dist_func=hmf_interp,
            )

        def the_plots_look_right(shear_model):
            radii = np.geomspace(1e-1, 60, 100)
            cons = 3*np.ones(1)
            alphas = 0.8*np.ones(1)
            betas = 3.4*np.ones(1)
            gammas = 0.2*np.ones(1)
            a_szs = -0.3*np.ones(1)
            a_2hs = np.linspace(0, 1, 3)

            esds = shear_model.stacked_excess_surface_density(radii, a_szs, a_2hs, cons, alphas, betas, gammas).squeeze()

            plt.plot(radii, radii[:, None]*esds)
            plt.xscale('log')

            plt.xlabel(r'$R$')
            plt.ylabel(r'$R \Delta\Sigma(R)$')

            plt.savefig('figs/test/2halo_corrected_scattered_matching_stacked_gnfw_excess_surface_density.svg')
            plt.gcf().clear()

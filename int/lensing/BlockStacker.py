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
import maszcal.stats
import maszcal.concentration
import maszcal.density
import maszcal.lensing
import maszcal.cosmology
import maszcal.tinker


def describe_BlockStacker():

    def describe_stacked_signal():

        @pytest.fixture
        def cluster_data():
            NUM_CLUSTERS = 20
            rng = np.random.default_rng(seed=13)
            sz_masses = 2e13*rng.normal(size=NUM_CLUSTERS) + 2e14
            zs = rng.random(size=NUM_CLUSTERS) + 0.01
            weights = rng.random(size=NUM_CLUSTERS)
            return sz_masses, zs, weights

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
        def convergence_model(cluster_data, hmf_interp, density_model):
            cosmo_params = maszcal.cosmology.CosmoParams()
            ms, zs, ws = cluster_data
            return maszcal.lensing.ScatteredMatchingConvergenceModel(
                sz_masses=ms,
                redshifts=zs,
                lensing_weights=ws,
                cosmo_params=cosmo_params,
                lensing_func=density_model.convergence,
                logmass_prob_dist_func=hmf_interp,
            )

        @pytest.fixture
        def block_stacker(cluster_data, density_model, hmf_interp):
            ms, zs, ws = cluster_data
            return maszcal.lensing.BlockStacker(
                sz_masses=ms,
                redshifts=zs,
                lensing_weights=ws,
                block_size=10,
                model=maszcal.lensing.ScatteredMatchingConvergenceModel,
                model_kwargs={
                    'lensing_func': density_model.convergence,
                    'logmass_prob_dist_func': hmf_interp,
                },
            )

        def it_gives_the_same_result_as_stacking_without_blocks(block_stacker, convergence_model):
            from_arcmin = 2 * np.pi / 360 / 60
            to_arcmin = 1/from_arcmin
            thetas = np.geomspace(0.05*from_arcmin, 60*from_arcmin, 60)
            cons = 3*np.ones(1)
            alphas = 0.6*np.ones(1)
            betas = 3.8*np.ones(1)
            gammas = 0.2*np.ones(1)
            a_szs = 0*np.ones(1)

            rho_params = (cons, alphas, betas, gammas)

            sds_block = block_stacker.stacked_signal(thetas, a_szs, *rho_params).squeeze()
            sds_noblock = convergence_model.stacked_signal(thetas, a_szs, *rho_params).squeeze()

            assert np.allclose(sds_block, sds_noblock, rtol=0.03)

    def describe_miscentered_stacked_signal():

        @pytest.fixture
        def cluster_data():
            NUM_CLUSTERS = 23
            rng = np.random.default_rng(seed=13)
            sz_masses = 2e13*rng.normal(size=NUM_CLUSTERS) + 2e14
            zs = rng.random(size=NUM_CLUSTERS) + 0.01
            weights = rng.random(size=NUM_CLUSTERS)
            return sz_masses, zs, weights

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
        def miscentered_conv(miscentered_rho_func):
            return maszcal.lensing.Convergence(
                rho_func=miscentered_rho_func,
                cosmo_params=maszcal.cosmology.CosmoParams(),
                units=u.Msun/u.pc**2,
                comoving=True,
                sd_func=projector.SurfaceDensity.calculate,
                sd_kwargs={'radial_axis_to_broadcast': 1, 'density_axis': -1},
            ).convergence

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
        def block_stacker(cluster_data, miscentered_conv, hmf_interp):
            ms, zs, ws = cluster_data
            return maszcal.lensing.BlockStacker(
                sz_masses=ms,
                redshifts=zs,
                lensing_weights=ws,
                block_size=10,
                model=maszcal.lensing.ScatteredMatchingConvergenceModel,
                model_kwargs={
                    'lensing_func': miscentered_conv,
                    'logmass_prob_dist_func': hmf_interp,
                },
            )

        def it_can_calculate_the_miscentered_stacked_signal(block_stacker):
            from_arcmin = 2 * np.pi / 360 / 60
            to_arcmin = 1/from_arcmin
            thetas = np.geomspace(0.05*from_arcmin, 60*from_arcmin, 60)
            cons = 3*np.ones(1)
            alphas = 0.6*np.ones(1)
            betas = 3.8*np.ones(1)
            gammas = 0.2*np.ones(1)
            miscenter_scales = 1e-1*np.ones(1)
            centering_probs = 0.5*np.ones(1)
            a_szs = 0*np.ones(1)

            rho_params = (cons, alphas, betas, gammas, miscenter_scales, centering_probs)

            sds_block = block_stacker.stacked_signal(thetas, a_szs, *rho_params).squeeze()
            assert not np.any(np.isnan(sds_block))
            assert False, sds_block

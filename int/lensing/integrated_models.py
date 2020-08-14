import pytest
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import seaborn as sns
sns.set(style='whitegrid', font_scale=1.5, rc={"lines.linewidth": 2,'lines.markersize': 8.0,})
import maszcal.concentration
import maszcal.density
from maszcal.lensing import IntegratedShearModel


def describe_GnfwCm_model():

    def describe_gnfw_rho():

        @pytest.fixture
        def density_model():
            return maszcal.density.CmGnfw(
                cosmo_params=maszcal.cosmology.CosmoParams(),
                mass_definition='mean',
                delta=200,
                comoving_radii=True,
                nfw_class=maszcal.density.NfwCmModel,
                con_class=maszcal.concentration.ConModel,
            )

        def the_plots_look_right(density_model):
            radii = np.logspace(-1, 1, 30)
            zs = np.linspace(0, 1, 20)
            mus = np.log(1e14)*np.ones(1)
            alphas = 0.5*np.ones(1)
            betas = np.linspace(2.8, 3.2, 3)
            gammas = 0.5*np.ones(1)

            rhos = density_model.rho_bary(radii, zs, mus, alphas, betas, gammas)[:, 0, 0, :]

            plt.plot(radii, rhos)
            plt.xscale('log')
            plt.yscale('log')

            plt.xlabel(r'$R$')
            plt.ylabel(r'$\rho(R)$')

            plt.savefig('figs/test/rho_bary_baryCm.svg')
            plt.gcf().clear()

    def describe_total_delta_sigma():

        @pytest.fixture
        def density_model():
            return maszcal.density.CmGnfw(
                cosmo_params=maszcal.cosmology.CosmoParams(),
                mass_definition='mean',
                delta=200,
                comoving_radii=True,
                nfw_class=maszcal.density.NfwCmModel,
                con_class=maszcal.concentration.ConModel,
            )

        @pytest.fixture
        def shear_model(density_model):
            mus = np.linspace(np.log(1e14), np.log(1e15), 30)
            zs = np.linspace(0, 1, 20)
            return IntegratedShearModel(mus, zs, rho_func=density_model.rho_tot)

        def the_plots_look_right(shear_model):
            radii = np.logspace(-1, 1, 30)
            zs = np.linspace(0, 1, 2)
            mus = np.log(1e14)*np.ones(1)
            alphas = 0.5*np.ones(1)
            betas = np.linspace(2.8, 3.2, 3)
            gammas = 0.5*np.ones(1)

            esds = shear_model._shear.delta_sigma_total(radii, zs, mus, alphas, betas, gammas)[:, 0, 0, :]

            plt.plot(radii, radii[:, None]*esds)
            plt.xscale('log')

            plt.xlabel(r'$R$')
            plt.ylabel(r'$R \Delta\Sigma(R)$')

            plt.savefig('figs/test/delta_sigma_total_baryCm.svg')
            plt.gcf().clear()

    def describe_stacked_delta_sigma():

        @pytest.fixture
        def density_model():
            return maszcal.density.CmGnfw(
                cosmo_params=maszcal.cosmology.CosmoParams(),
                mass_definition='mean',
                delta=200,
                comoving_radii=True,
                nfw_class=maszcal.density.NfwCmModel,
                con_class=maszcal.concentration.ConModel,
            )

        @pytest.fixture
        def shear_model(density_model):
            mus = np.linspace(np.log(1e14), np.log(1e15), 30)
            zs = np.linspace(0, 1, 20)
            return IntegratedShearModel(mus, zs, rho_func=density_model.rho_tot)

        def the_plots_look_right(shear_model):
            radii = np.logspace(-1, 1, 30)
            alphas = 0.5*np.ones(1)
            betas = np.linspace(2.8, 3.2, 3)
            gammas = 0.5*np.ones(1)
            a_szs = 0.3*np.ones(1)

            esds = shear_model.stacked_delta_sigma(radii, a_szs, alphas, betas, gammas)

            plt.plot(radii, radii[:, None]*esds)
            plt.xscale('log')

            plt.xlabel(r'$R$')
            plt.ylabel(r'$R \Delta\Sigma(R)$')

            plt.savefig('figs/test/stacked_gnfw_delta_sigma_baryCm.svg')
            plt.gcf().clear()


def describe_Gnfw_model():

    def describe_gnfw_rho():

        @pytest.fixture
        def density_model():
            return maszcal.density.Gnfw(
                cosmo_params=maszcal.cosmology.CosmoParams(),
                mass_definition='mean',
                delta=200,
                comoving_radii=True,
                nfw_class=maszcal.density.NfwModel,
            )

        def the_plots_look_right(density_model):
            radii = np.logspace(-1, 1, 30)
            zs = np.linspace(0, 1, 20)
            mus = np.log(1e14)*np.ones(1)
            cons = 2*np.ones(1)
            alphas = 0.5*np.ones(1)
            betas = np.linspace(2.8, 3.2, 3)
            gammas = 0.5*np.ones(1)

            rhos = density_model.rho_bary(radii, zs, mus, cons, alphas, betas, gammas)[:, 0, 0, :]

            plt.plot(radii, rhos)
            plt.xscale('log')
            plt.yscale('log')

            plt.xlabel(r'$R$')
            plt.ylabel(r'$\rho(R)$')

            plt.savefig('figs/test/rho_bary.svg')
            plt.gcf().clear()

    def describe_total_delta_sigma():

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
        def density_model_():
            model = maszcal.density.Gnfw(
                cosmo_params=maszcal.cosmology.CosmoParams(),
                mass_definition='mean',
                delta=200,
                comoving_radii=True,
                nfw_class=maszcal.density.NfwModel,
            )
            model.CORE_RADIUS = 1/2 # 1/concentration to be used
            return model

        @pytest.fixture
        def shear_model(density_model):
            mus = np.linspace(np.log(1e14), np.log(1e15), 30)
            zs = np.linspace(0, 1, 20)
            return IntegratedShearModel(mus, zs, rho_func=density_model.rho_tot)

        def it_can_recreate_an_nfw_model(density_model_):
            radii = np.logspace(-1, 1, 30)
            zs = np.linspace(0, 1, 20)
            mus = np.log(1e14)*np.ones(1)
            cons = 2*np.ones(1) # It is critical this concentration match the core radius above
            alphas = np.ones(1)
            betas = 3*np.ones(1)
            gammas = np.ones(1)

            assert np.all(density_model_.CORE_RADIUS == 1/cons)

            rho_barys = density_model_.rho_bary(radii,
                                                zs,
                                                mus,
                                                cons,
                                                alphas,
                                                betas,
                                                gammas)/density_model_.baryon_frac

            rho_nfws = density_model_.rho_cdm(radii,
                                              zs,
                                              mus,
                                              cons)/(1-density_model_.baryon_frac)
            rho_nfws = np.moveaxis(rho_nfws, -2, 0)

            assert np.allclose(rho_nfws, rho_barys, rtol=1e-2)

        def the_plots_look_right(shear_model):
            radii = np.logspace(-1, 1, 30)
            zs = np.linspace(0, 1, 20)
            mus = np.log(1e14)*np.ones(1)
            cons = 2*np.ones(1)
            alphas = 0.5*np.ones(1)
            betas = np.linspace(2.8, 3.2, 3)
            gammas = 0.5*np.ones(1)

            esds = shear_model._shear.delta_sigma_total(radii, zs, mus, cons, alphas, betas, gammas)[:, 0, 0, :]

            plt.plot(radii, radii[:, None]*esds)
            plt.xscale('log')

            plt.xlabel(r'$R$')
            plt.ylabel(r'$R \Delta\Sigma(R)$')

            plt.savefig('figs/test/delta_sigma_total.svg')
            plt.gcf().clear()

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
        def shear_model(density_model):
            mus = np.linspace(np.log(1e14), np.log(1e15), 30)
            zs = np.linspace(0, 1, 20)
            return IntegratedShearModel(mus, zs, rho_func=density_model.rho_tot)

        def the_plots_look_right(shear_model):
            radii = np.logspace(-1, 1, 30)
            cons = 3*np.ones(1)
            alphas = 0.5*np.ones(1)
            betas = np.linspace(3.4, 4, 3)
            gammas = 0.2*np.ones(1)
            a_szs = 0.3*np.ones(1)

            esds = shear_model.stacked_delta_sigma(radii, a_szs, cons, alphas, betas, gammas)

            plt.plot(radii, radii[:, None]*esds)
            plt.xscale('log')

            plt.xlabel(r'$R$')
            plt.ylabel(r'$R \Delta\Sigma(R)$')

            plt.savefig('figs/test/stacked_gnfw_delta_sigma.svg')
            plt.gcf().clear()

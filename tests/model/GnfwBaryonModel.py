import pytest
import numpy as np
import astropy.units as u
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import seaborn as sns
sns.set(style='whitegrid', font_scale=1.5, rc={"lines.linewidth": 2,'lines.markersize': 8.0,})
from maszcal.model import GnfwBaryonModel


def describe_gaussian_baryonic_model():

    def describe_gnfw_rho():

        @pytest.fixture
        def baryon_model():
            mus = np.linspace(np.log(1e14), np.log(1e15), 30)
            zs = np.linspace(0, 1, 20)
            return GnfwBaryonModel(mus, zs)

        def the_plots_look_right(baryon_model):
            radii = np.logspace(-1, 1, 30)
            mus = np.log(1e14)*np.ones(1)
            cons = 3*np.ones(1)
            alphas = np.ones(1)
            betas = np.linspace(3, 5, 3)
            gammas = np.ones(1)

            rhos = baryon_model.rho_bary(radii, mus, cons, alphas, betas, gammas)[:, 0, 0, :]

            plt.plot(radii, rhos)
            plt.xscale('log')
            plt.yscale('log')

            plt.xlabel(r'$R$')
            plt.ylabel(r'$\rho(R)$')

            plt.savefig('figs/test/rho_bary.svg')
            plt.gcf().clear()

    def describe_gnfw_delta_sigma():

        @pytest.fixture
        def baryon_model():
            mus = np.linspace(np.log(1e14), np.log(1e15), 30)
            zs = np.linspace(0, 1, 20)
            return GnfwBaryonModel(mus, zs, units=u.Msun/u.pc**2)

        def the_plots_look_right(baryon_model):
            radii = np.logspace(-1, 1, 30)
            mus = np.log(1e14)*np.ones(1)
            cons = 3*np.ones(1)
            alphas = np.ones(1)
            betas = np.linspace(3, 5, 3)
            gammas = np.ones(1)

            esds = baryon_model.delta_sigma_bary(radii, mus, cons, alphas, betas, gammas)[0, 0, :, :]

            plt.plot(radii, radii[:, None]*esds)
            plt.xscale('log')

            plt.xlabel(r'$R$')
            plt.ylabel(r'$R \Delta\Sigma(R)$')

            plt.savefig('figs/test/delta_sigma_bary.svg')
            plt.gcf().clear()

    def describe_total_delta_sigma():

        @pytest.fixture
        def baryon_model():
            mus = np.linspace(np.log(1e14), np.log(1e15), 30)
            zs = np.linspace(0, 1, 20)
            return GnfwBaryonModel(mus, zs, units=u.Msun/u.pc**2)

        def the_plots_look_right(baryon_model):
            radii = np.logspace(-1, 1, 30)
            mus = np.log(1e14)*np.ones(1)
            cons = 3*np.ones(1)
            alphas = np.ones(1)
            betas = np.linspace(3, 5, 3)
            gammas = np.ones(1)

            esds = baryon_model.delta_sigma_total(radii, mus, cons, alphas, betas, gammas)[0, 0, :, :]

            plt.plot(radii, radii[:, None]*esds)
            plt.xscale('log')

            plt.xlabel(r'$R$')
            plt.ylabel(r'$R \Delta\Sigma(R)$')

            plt.savefig('figs/test/delta_sigma_total.svg')
            plt.gcf().clear()

    def describe_stacked_delta_sigma():

        @pytest.fixture
        def baryon_model():
            mus = np.linspace(np.log(1e14), np.log(1e15), 30)
            zs = np.linspace(0, 1, 20)
            return GnfwBaryonModel(mus, zs, units=u.Msun/u.pc**2)

        def the_plots_look_right(baryon_model):
            radii = np.logspace(-1, 1, 30)
            cons = 3*np.ones(1)
            alphas = np.ones(1)
            betas = np.linspace(1, 5, 5)
            gammas = np.ones(1)
            a_szs = -0.3*np.ones(1)

            esds = baryon_model.stacked_delta_sigma(radii, cons, alphas, betas, gammas, a_szs)

            plt.plot(radii, radii[:, None]*esds)
            plt.xscale('log')

            plt.xlabel(r'$R$')
            plt.ylabel(r'$R \Delta\Sigma(R)$')

            plt.savefig('figs/test/stacked_gnfw_delta_sigma.svg')
            plt.gcf().clear()
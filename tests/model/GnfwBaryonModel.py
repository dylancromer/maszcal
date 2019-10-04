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
            alphas = np.ones(1)
            betas = np.linspace(1, 3, 3)
            gammas = np.ones(1)

            rhos = baryon_model.rho_gnfw(radii, mus, alphas, betas, gammas)[:, 0, :]

            plt.plot(radii, rhos)
            plt.xscale('log')
            plt.yscale('log')

            plt.xlabel(r'$R$')
            plt.ylabel(r'$\rho(R)$')

            plt.savefig('figs/test/rho_gnfw.svg')
            plt.gcf().clear()

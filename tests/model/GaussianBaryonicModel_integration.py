
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
from maszcal.model import GaussianBaryonModel


def describe_gaussian_baryonic_model():

    def describe_baryonic_profile():

        @pytest.fixture
        def baryon_model():
            mus = np.log(2e14)*np.ones(1)
            zs = np.zeros(1)
            return GaussianBaryonModel(mus, zs)

        def it_makes_a_correct_baryon_delta_sigma(baryon_model):
            rs = np.logspace(-1, 1, 100)
            mus = np.log(2e14)*np.ones(1)
            baryon_vars = np.logspace(-3, 0, 10)

            ds = baryon_model.delta_sigma_baryon(rs, mus, baryon_vars)[0, :, :]

            plt.plot(rs, ds)
            plt.xscale('log')
            plt.xlabel(r'$R$')
            plt.yscale('log')
            plt.ylabel(r'$\Delta \Sigma$')

            plt.savefig('figs/test/baryon_delta_sigma.svg')
            plt.gcf().clear()

            plt.plot(rs, rs[:, None]*ds)
            plt.xscale('log')
            plt.xlabel(r'$R$')
            plt.ylabel(r'$R \Delta \Sigma$')

            plt.savefig('figs/test/baryon_r_delta_sigma.svg')
            plt.gcf().clear()

        def it_makes_a_correct_total_delta_sigma(baryon_model):
            rs = np.logspace(-1, 1, 100)
            mus = np.log(2e14)*np.ones(1)
            baryon_vars = np.logspace(-3, 0, 10)
            cons = 3*np.ones(1)

            ds = baryon_model.delta_sigma_of_mass(rs, mus, cons, baryon_vars)[0, 0, :, :]

            plt.plot(rs, ds)
            plt.xscale('log')
            plt.xlabel(r'$R$')
            plt.yscale('log')
            plt.ylabel(r'$\Delta \Sigma$')

            plt.savefig('figs/test/baryon_plus_cdm_delta_sigma.svg')
            plt.gcf().clear()

            plt.plot(rs, rs[:, None]*ds)
            plt.xscale('log')
            plt.xlabel(r'$R$')
            plt.ylabel(r'$R \Delta \Sigma$')

            plt.savefig('figs/test/baryon_plus_cdm_r_delta_sigma.svg')
            plt.gcf().clear()


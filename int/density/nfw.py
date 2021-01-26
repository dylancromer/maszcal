import numpy as np
import pytest
import astropy.units as u
from maszcal.cosmology import CosmoParams
import maszcal.density
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import seaborn as sns
sns.set(style='whitegrid', font_scale=1.5, rc={"lines.linewidth": 2,'lines.markersize': 8.0,})


def describe_MatchingNfwModel():

    @pytest.fixture
    def nfw_model():
        cosmo = CosmoParams()
        return maszcal.density.MatchingNfwModel(cosmo_params=cosmo)

    def it_calculates_excess_surface_density(nfw_model):
        rs = np.logspace(-1, 1, 10)[:, None]
        zs = np.linspace(0, 1, 8)
        rs = rs * np.ones_like(zs)[None, :]
        masses = np.logspace(14, 14.8, 8)
        cons = np.linspace(2, 3.4, 4)

        ds = nfw_model.excess_surface_density(rs, zs, masses, cons)

        assert np.all(ds > 0)
        assert ds.shape == (10, 8, 4)

    def it_can_calculate_surface_density(nfw_model):
        rs = np.logspace(-1, 1, 10)[:, None]
        zs = np.linspace(0, 1, 8)
        rs = rs * np.ones_like(zs)[None, :]
        masses = np.logspace(14, 14.8, 8)
        cons = np.linspace(2, 3.4, 4)

        sd = nfw_model.surface_density(rs, zs, masses, cons)

        assert np.all(sd > 0)
        assert sd.shape == (10, 8, 4)

    def it_can_calculate_a_rho(nfw_model):
        rs = np.logspace(-1, 1, 10)[:, None]
        zs = np.linspace(0, 1, 8)
        rs = rs * np.ones_like(zs)[None, :]
        masses = np.logspace(14, 15, 8)
        cons = np.linspace(2, 4, 4)

        rhos = nfw_model.rho(rs, zs, masses, cons)

        assert np.all(rhos > 0)
        assert rhos.shape == (10, 8, 4)

    def it_can_calculate_convergence(nfw_model):
        rs = np.logspace(-2, 1, 100)[:, None]
        zs = np.linspace(0.1, 1, 8)
        rs = rs * np.ones_like(zs)[None, :]
        masses = np.logspace(14, 14.8, 8)
        cons = 2.2*np.ones(1)

        conv = nfw_model.convergence(rs, zs, np.log(masses), cons).squeeze()
        plt.plot(rs, conv)
        plt.xscale('log')
        plt.xlabel(r'$r$')
        plt.ylabel(r'$\kappa(r)$')
        plt.savefig('figs/test/nfw_conv_test.svg')

from dataclasses import dataclass
import pytest
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import seaborn as sns
sns.set(style='whitegrid', font_scale=1.5, rc={"lines.linewidth": 2,'lines.markersize': 8.0,})
import maszcal.twohalo
import maszcal.cosmology


def describe_TwoHaloShearModel():

    @pytest.fixture
    def two_halo_model(mocker):
        cosmo = maszcal.cosmology.CosmoParams()
        model = maszcal.twohalo.TwoHaloShearModel(cosmo_params=cosmo)
        return model

    def it_calculates_two_halo_esds(two_halo_model):
        zs = np.linspace(0, 1, 10)
        mus = np.linspace(np.log(1e14), np.log(1e15), 5)
        rs = np.logspace(-3, 3, 200)

        esds = two_halo_model.esd(rs, mus, zs)

        assert not np.any(np.isnan(esds))
        assert esds.shape == mus.shape + zs.shape + rs.shape

        plt.plot(rs, esds[0, :, :].T)
        plt.xscale('log')
        plt.xlabel(r'$R \; (\mathrm{Mpc}$)')
        plt.ylabel(r'$\Delta \Sigma \; (M_\odot/\mathrm{pc}^2)$')
        plt.savefig('figs/test/two_halo_esd.svg')

        plt.gcf().clear()

        plt.plot(rs, rs[:, None]*esds[0, :, :].T)
        plt.xscale('log')
        plt.xlabel(r'$R \; (\mathrm{Mpc}$)')
        plt.ylabel(r'$R \, \Delta \Sigma \; (10^6 \, M_\odot/\mathrm{pc})$')
        plt.savefig('figs/test/two_halo_r_esd.svg')

        plt.gcf().clear()

    def it_calculates_halo_matter_correlations(two_halo_model):
        zs = np.linspace(0, 1, 10)
        mus = np.linspace(np.log(1e14), np.log(1e15), 5)
        rs = np.logspace(-2, 3, 600)

        xis = two_halo_model.halo_matter_correlation(rs, mus, zs)

        assert not np.any(np.isnan(xis))
        assert xis.shape == mus.shape + zs.shape + rs.shape

        plt.plot(rs, rs[:, None]**2 * xis[:, 0, :].T)
        plt.xscale('log')
        plt.xlabel(r'$R \; (\mathrm{Mpc}$)')
        plt.ylabel(r'$\xi$')
        plt.savefig('figs/test/halo_matter_correlation.svg')

        plt.gcf().clear()

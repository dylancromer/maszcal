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
from maszcal.lensing import SingleMassShearModel
import maszcal.density


def describe_single_mass_bin():

    @pytest.fixture
    def density_model():
        return maszcal.density.SingleMassGnfw(
            cosmo_params=maszcal.cosmology.CosmoParams(),
            mass_definition='mean',
            delta=200,
            comoving_radii=True,
            nfw_class=maszcal.density.MatchingNfwModel,
        )

    @pytest.fixture
    def single_mass_model(density_model):
        zs = np.ones(1)
        return SingleMassShearModel(redshifts=zs, rho_func=density_model.rho_tot)

    def the_plot_looks_correct(single_mass_model):
        z = np.array([0.43])

        rs = np.logspace(-1, 1, 50)
        mu = np.array([np.log(4.26e14)])
        concentration = np.array([2.08])
        alpha = np.array([0.88])
        beta = np.array([3.8])
        gamma = np.array([0.2])
        params = np.array([[mu, concentration, alpha, beta, gamma]])

        ds = single_mass_model.delta_sigma(rs, mu, concentration, alpha, beta, gamma)

        plt.plot(rs, ds.flatten())
        plt.xlabel(r'$R \; (\mathrm{Mpc}/h)$')
        plt.ylabel(r'$\Delta \Sigma \; (h\, M_\circ / \mathrm{pc}^2)$')

        plt.ylim((1e0, 3.3e2))

        plt.xscale('log')
        plt.yscale('log')
        plt.savefig('figs/test/single_mass_bin_baryons.svg')
        plt.gcf().clear()

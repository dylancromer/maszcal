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
from maszcal.lensing import SingleMassNfwShearModel


def describe_single_mass_bin():

    @pytest.fixture
    def single_mass_model():
        zs = np.ones(1)
        return SingleMassNfwShearModel(redshift=zs)

    def the_plot_looks_correct():
        z = np.array([0.43])
        single_mass_model = SingleMassNfwShearModel(redshifts=z, delta=500, mass_definition='crit')

        rs = np.logspace(-1, 1, 50)
        mu = np.array([np.log(4.26e14)])
        concentration = np.array([2.08])
        params = np.array([[mu, concentration]])

        ds = single_mass_model.delta_sigma(rs, mu, concentration)

        plt.plot(rs, ds.flatten())
        plt.xlabel(r'$R \; (\mathrm{Mpc}/h)$')
        plt.ylabel(r'$\Delta \Sigma \; (h\, M_\circ / \mathrm{pc}^2)$')

        plt.ylim((1e0, 3.3e2))

        plt.xscale('log')
        plt.yscale('log')
        plt.savefig('figs/test/single_mass_bin.svg')
        plt.gcf().clear()

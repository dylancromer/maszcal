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
import maszcal.lensing
import maszcal.density


def describe_NfwOnly_example():

    @pytest.fixture
    def rho_func():
        density_model = maszcal.density.ProjectorSafeNfwModel(
            delta=200,
            mass_definition='mean',
        )

        def rho_func_(r, z, mu, con):
            mass = np.exp(mu)
            return density_model.rho(r, z, mass, con)
        return rho_func_

    @pytest.fixture
    def convergence_model(rho_func):
        return maszcal.lensing.SingleMassConvergenceModel(
            rho_func=rho_func,
            units=u.Msun/u.pc**2,
        )

    def the_plot_looks_correct(convergence_model):
        from_arcmin = 2 * np.pi / 360 / 60
        thetas = np.logspace(np.log10(0.05*from_arcmin), np.log10(15*from_arcmin), 30)
        z = np.array([0.123])
        mu = np.array([np.log(1.23e14)])
        con = np.array([3])

        kappas = convergence_model.kappa(thetas, z, mu, con)

        to_arcmin = 1/from_arcmin

        plt.plot(thetas*to_arcmin, kappas.squeeze())
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$\kappa$')

        plt.yscale('log')
        plt.xscale('log')
        plt.savefig('figs/test/single_mass_bin_nfw_kappa.svg')
        plt.gcf().clear()

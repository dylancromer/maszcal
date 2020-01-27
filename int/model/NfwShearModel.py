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
from maszcal.model import NfwShearModel


def describe_nfw_shear_model():

    @pytest.fixture
    def stacked_model():
        mus = np.array([np.log(1e15)])
        zs = np.linspace(0, 2, 20)
        return NfwShearModel(mus, zs, units=u.Msun/(u.pc**2), delta=500, mass_definition='crit')

    def test_delta_sigma_of_m(stacked_model):
        rs = np.logspace(-1, 2, 40)
        mus = np.array([np.log(1e15)])
        cons = np.array([2])

        delta_sigmas = stacked_model.delta_sigma(rs,
                                                         mus,
                                                         cons)

        delta_sigmas = delta_sigmas[0,:,:,0]


        plt.plot(rs, rs[:, None]*delta_sigmas.T)
        plt.title(rf'$ M = {round(np.exp(mus[0])/1e14, 2)} \; 10^{{14}} M_{{\odot}}$')
        plt.xlabel(r'$ r $')
        plt.ylabel(r'$ r \Delta \Sigma (10^6 \, M_{\odot} / \mathrm{{pc}}) $')
        plt.xscale('log')

        plt.savefig('figs/test/delta_sigma_r_m.svg')
        plt.gcf().clear()


    def test_delta_sigma_of_r(stacked_model):
        mubins = np.linspace(np.log(1e14), np.log(1e16), 30)
        zbins = np.linspace(0, 2, 30)

        cons = 2*np.ones(1)
        a_szs = np.zeros(1)

        stacked_model = NfwShearModel(mubins, zbins)

        rs = np.logspace(-1, 2, 40)

        params = np.array([[2, 2]])

        stacked_model.params = params

        delta_sigmas = stacked_model.stacked_delta_sigma(rs, cons, a_szs)[:,0]

        plt.plot(rs, rs * delta_sigmas)
        plt.xlabel(r'$ r $')
        plt.ylabel(r'$ r \Delta \Sigma (10^6 \, M_{\odot} / \mathrm{{pc}}) $')
        plt.xscale('log')

        plt.savefig('figs/test/delta_sigma_r.svg')
        plt.gcf().clear()


    def test_delta_sigma_of_m_nocomoving(stacked_model):
        stacked_model.comoving_radii = False

        rs = np.logspace(-1, 2, 40)
        mus = np.array([np.log(1e15)])
        cons = np.array([2])

        delta_sigmas = stacked_model.delta_sigma(rs, mus, cons)


        delta_sigmas = delta_sigmas[0,:,:,0]


        plt.plot(rs, rs[:, None]*delta_sigmas.T)
        plt.title(rf'$ M = {round(np.exp(mus[0])/1e14, 2)} \; 10^{{14}} M_{{\odot}}$')
        plt.xlabel(r'$ r $')
        plt.ylabel(r'$ r \Delta \Sigma (10^6 \, M_{\odot} / \mathrm{{pc}}) $')
        plt.xscale('log')

        plt.savefig('figs/test/delta_sigma_r_m_comoving_false.svg')
        plt.gcf().clear()


    def test_delta_sigma_of_r_nocomoving(stacked_model):
        mubins = np.linspace(np.log(1e14), np.log(1e16), 30)
        zbins = np.linspace(0, 2, 30)

        stacked_model = NfwShearModel(mubins, zbins)
        stacked_model.comoving_radii = False

        rs = np.logspace(-1, 2, 40)

        cons = 2*np.ones(1)
        a_szs = np.zeros(1)

        delta_sigmas = stacked_model.stacked_delta_sigma(rs, cons, a_szs)[:,0]

        plt.plot(rs, rs * delta_sigmas)
        plt.xlabel(r'$ r $')
        plt.ylabel(r'$ r \Delta \Sigma (10^6 \, M_{\odot} / \mathrm{{pc}}) $')
        plt.xscale('log')

        plt.savefig('figs/test/delta_sigma_r_comoving_false.svg')
        plt.gcf().clear()

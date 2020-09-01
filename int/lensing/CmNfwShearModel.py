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
from maszcal.lensing import CmNfwShearModel


def describe_nfw_cm_shear_model():

    @pytest.fixture
    def stacked_model():
        mus = np.array([np.log(1e15)])
        zs = np.linspace(0, 2, 20)
        return CmNfwShearModel(mus, zs, units=u.Msun/(u.pc**2), delta=500, mass_definition='crit')

    def test_excess_surface_density_of_m(stacked_model):
        rs = np.logspace(-1, 2, 40)
        mus = np.array([np.log(1e15)])

        excess_surface_densitys = stacked_model.excess_surface_density(rs, mus)

        excess_surface_densitys = excess_surface_densitys[0,:,:]


        plt.plot(rs, rs[:, None]*excess_surface_densitys.T)
        plt.title(rf'$ M = {round(np.exp(mus[0])/1e14, 2)} \; 10^{{14}} M_{{\odot}}$')
        plt.xlabel(r'$ r $')
        plt.ylabel(r'$ r \Delta \Sigma (10^6 \, M_{\odot} / \mathrm{{pc}}) $')
        plt.xscale('log')

        plt.savefig('figs/test/excess_surface_density_r_m_cm_relation.svg')
        plt.gcf().clear()


    def test_excess_surface_density_of_r(stacked_model):
        mubins = np.linspace(np.log(1e14), np.log(1e16), 30)
        zbins = np.linspace(0, 2, 30)

        a_szs = np.zeros(1)

        stacked_model = CmNfwShearModel(mubins, zbins)

        rs = np.logspace(-1, 2, 40)

        params = np.array([[2, 2]])

        stacked_model.params = params

        excess_surface_densitys = stacked_model.stacked_excess_surface_density(rs, a_szs)[:,0]

        plt.plot(rs, rs * excess_surface_densitys)
        plt.xlabel(r'$ r $')
        plt.ylabel(r'$ r \Delta \Sigma (10^6 \, M_{\odot} / \mathrm{{pc}}) $')
        plt.xscale('log')

        plt.savefig('figs/test/excess_surface_density_r_cm_relation.svg')
        plt.gcf().clear()


    def test_excess_surface_density_of_m_nocomoving(stacked_model):
        stacked_model.comoving_radii = False

        rs = np.logspace(-1, 2, 40)
        mus = np.array([np.log(1e15)])

        excess_surface_densitys = stacked_model.excess_surface_density(rs, mus)


        excess_surface_densitys = excess_surface_densitys[0,:,:]


        plt.plot(rs, rs[:, None]*excess_surface_densitys.T)
        plt.title(rf'$ M = {round(np.exp(mus[0])/1e14, 2)} \; 10^{{14}} M_{{\odot}}$')
        plt.xlabel(r'$ r $')
        plt.ylabel(r'$ r \Delta \Sigma (10^6 \, M_{\odot} / \mathrm{{pc}}) $')
        plt.xscale('log')

        plt.savefig('figs/test/excess_surface_density_r_m_comoving_false_cm_relation.svg')
        plt.gcf().clear()


    def test_excess_surface_density_of_r_nocomoving(stacked_model):
        mubins = np.linspace(np.log(1e14), np.log(1e16), 30)
        zbins = np.linspace(0, 2, 30)

        stacked_model = CmNfwShearModel(mubins, zbins)
        stacked_model.comoving_radii = False

        rs = np.logspace(-1, 2, 40)

        a_szs = np.zeros(1)

        excess_surface_densitys = stacked_model.stacked_excess_surface_density(rs, a_szs)[:,0]

        plt.plot(rs, rs * excess_surface_densitys)
        plt.xlabel(r'$ r $')
        plt.ylabel(r'$ r \Delta \Sigma (10^6 \, M_{\odot} / \mathrm{{pc}}) $')
        plt.xscale('log')

        plt.savefig('figs/test/excess_surface_density_r_comoving_false_cm_relation.svg')
        plt.gcf().clear()

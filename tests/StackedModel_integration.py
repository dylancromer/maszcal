from maszcal.model import StackedModel
import numpy as np

import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import seaborn as sns
sns.set(style='whitegrid', font_scale=1.5, rc={"lines.linewidth": 2,'lines.markersize': 8.0,})


stacked_model = StackedModel()


def test_delta_sigma_of_m():
    #TODO: Once other collaborator functions are implemented, will input precomputed vals
    rs = np.logspace(-1,2, 20)
    mus = stacked_model.mus
    cons = stacked_model.concentrations

#    delta_sigmas = stacked_model.delta_sigma_of_mass(rs, mus, cons)

#    precomp_delta_sigmas = np.ones((rs.size, mus.size))

    #np.testing.assert_allclose(delta_sigmas, precomp_delta_sigmas)


def test_delta_sigma_of_r():
    #TODO: Once other collaborator functions are implemented, will input precomputed vals
    rs = np.logspace(-1, 1, 40)

#    delta_sigmas = stacked_model.delta_sigma(rs)

#    precomp_delta_sigmas = np.ones(rs.shape)

    #np.testing.assert_allclose(delta_sigmas, precomp_delta_sigmas)


def test_power_spectrum():
    #Need to check plots on this one!
    stacked_model.calc_power_spect()

    z = 0
    ks = np.logspace(-4, -1, 200)

    power_spect_redshift0 = stacked_model.power_spectrum_interp.P(z, ks)
    plt.plot(ks, power_spect_redshift0.T)
    plt.xlabel(r'$ kh $')
    plt.ylabel(r'$ P(z=0, k) $')
    plt.xscale('log')
    plt.yscale('log')

    plt.savefig('figs/test/power_redshift0.svg')
    plt.gcf().clear()

def test_tinker_mf():
    masses = stacked_model.mass(stacked_model.mus)
    dn_dlnms = stacked_model.dnumber_dlogmass() #masses, zs

    assert False, print(dn_dlnms)

    plt.plot(masses, dn_dlnms[:, 0])
    plt.xlabel(r'$ M \; (M_{\odot}) $')
    plt.ylabel(r'$ \frac{dn}{d \ln{M}} $')
    plt.xscale('log')
    plt.yscale('log')

    plt.savefig('figs/test/dn_dlnm_redshift0.svg')
    plt.gcf().clear()

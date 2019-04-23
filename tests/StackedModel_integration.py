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


def _delta_sigma_of_m():
    #TODO: Once other collaborator functions are implemented, will input precomputed vals
    rs = np.logspace(-1, 2, 20)
    mus = stacked_model.mus

    delta_sigmas = stacked_model.delta_sigma_of_mass(rs, mus)

    precomp_delta_sigmas = np.ones((rs.size, mus.size))

    np.testing.assert_allclose(delta_sigmas, precomp_delta_sigmas)


def _delta_sigma_of_r():
    #TODO: Once other collaborator functions are implemented, will input precomputed vals
    rs = np.logspace(-1, 1, 40)

    delta_sigmas = stacked_model.delta_sigma(rs)

    precomp_delta_sigmas = np.ones(rs.shape)

    np.testing.assert_allclose(delta_sigmas, precomp_delta_sigmas)


def test_power_spectrum():
    #Need to check plots on this one!
    stacked_model.calc_power_spect()

    z = 0
    ks = stacked_model.ks

    power_spect_redshift0 = stacked_model.power_spect
    plt.plot(ks, power_spect_redshift0.T)
    plt.xlabel(r'$ kh $')
    plt.ylabel(r'$ P(z=0, k) $')
    plt.xscale('log')
    plt.yscale('log')

    plt.savefig('figs/test/power_redshift0.svg')
    plt.gcf().clear()


def test_comoving_vol():
    vols = stacked_model.comoving_vol()
    zs = stacked_model.zs

    plt.plot(zs, vols)
    plt.xlabel(r'$ z $')
    plt.ylabel(r'$ c r^2(z)/H(z)$')

    plt.savefig('figs/test/comoving_vol.svg')
    plt.gcf().clear()


from maszcal.cosmology import CosmoParams
stacked_model = StackedModel()
def test_tinker_mf():
    #WMAP cosmology
    used_ppf = True
    stacked_model.cosmo_params = CosmoParams(
        hubble_constant = 72,
        omega_bary_hsqr = 0.024,
        omega_cdm_hsqr = 0.14,
        omega_lambda = 0.742,
        tau_reion = 0.166,
        spectral_index = 0.99,
        neutrino_mass_sum = 0,
        use_ppf = used_ppf,
    )

    h = stacked_model.cosmo_params.h
    rho_matter = stacked_model.cosmo_params.rho_crit * stacked_model.cosmo_params.omega_matter / h**2

    z = 0
    mink = 1e-4
    maxks = [1, 3, 5, 10]
    for maxk in maxks:
        stacked_model.zs = np.array([z])
        stacked_model.min_k = mink
        stacked_model.max_k = maxk

        stacked_model.calc_power_spect()

        masses = stacked_model.mass(stacked_model.mus)
        dn_dlnms = stacked_model.dnumber_dlogmass() #masses, zs
        dn_dms = dn_dlnms[0, :]/masses


        plotlabel = rf'$k_{{\mathrm{{max}}}}={maxk}$'
        plt.plot(masses, masses**2 * dn_dms / rho_matter, label=plotlabel)

    plt.title(rf'$z = {z}$, ppf {used_ppf}')
    plt.xlabel(r'$ M \; (M_{\odot}) $')
    plt.ylabel(r'$ M^2/\rho_m \; dn/dM$')
    plt.legend(loc='best')
    plt.ylim((4e-4, 3e-1))
    plt.xscale('log')
    plt.yscale('log')

    filename = f'dn_dlnm_redshift{z}-ppf_{str(used_ppf)}.svg'
    plt.savefig('figs/test/' + filename)
    plt.gcf().clear()

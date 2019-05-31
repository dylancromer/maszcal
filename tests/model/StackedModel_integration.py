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

from maszcal.model import StackedModel




stacked_model = StackedModel()

def test_delta_sigma_of_m():
    rs = np.logspace(-1, 2, 40)
    mus = np.array([15])
    cons = np.array([2])

    delta_sigmas = stacked_model.delta_sigma_of_mass(rs,
                                                     mus,
                                                     concentrations=cons,
                                                     units=u.Msun/(u.Mpc * u.pc))


    delta_sigmas = delta_sigmas[0,:,:,0]


    plt.plot(rs, rs[:, None]*delta_sigmas.T/1e6)
    plt.title(rf'$ M = 10^{{{mus[0]}}} \; M_{{\odot}}$')
    plt.xlabel(r'$ r $')
    plt.ylabel(r'$ r \Delta \Sigma (10^6 \, M_{\odot} / \mathrm{{pc}}) $')
    plt.xscale('log')

    plt.savefig('figs/test/delta_sigma_r_m.svg')
    plt.gcf().clear()


def test_delta_sigma_of_r():
    rs = np.logspace(-1, 2, 40)

    delta_sigmas = stacked_model.delta_sigma(rs, units=u.Msun/(u.Mpc * u.pc))[:,0,0]

    plt.plot(rs, rs * delta_sigmas/1e6)
    plt.xlabel(r'$ r $')
    plt.ylabel(r'$ r \Delta \Sigma (10^6 \, M_{\odot} / \mathrm{{pc}}) $')
    plt.xscale('log')

    plt.savefig('figs/test/delta_sigma_r.svg')
    plt.gcf().clear()


def test_delta_sigma_of_m_nocomoving():
    stacked_model.comoving_radii = False

    rs = np.logspace(-1, 2, 40)
    mus = np.array([15])
    cons = np.array([2])

    delta_sigmas = stacked_model.delta_sigma_of_mass(rs,
                                                     mus,
                                                     concentrations=cons,
                                                     units=u.Msun/(u.Mpc * u.pc))


    delta_sigmas = delta_sigmas[0,:,:,0]


    plt.plot(rs, rs[:, None]*delta_sigmas.T/1e6)
    plt.title(rf'$ M = 10^{{{mus[0]}}} \; M_{{\odot}}$')
    plt.xlabel(r'$ r $')
    plt.ylabel(r'$ r \Delta \Sigma (10^6 \, M_{\odot} / \mathrm{{pc}}) $')
    plt.xscale('log')

    plt.savefig('figs/test/delta_sigma_r_m_comoving_false.svg')
    plt.gcf().clear()


def test_delta_sigma_of_r_nocomoving():
    stacked_model.comoving_radii = False

    rs = np.logspace(-1, 2, 40)

    delta_sigmas = stacked_model.delta_sigma(rs, units=u.Msun/(u.Mpc * u.pc))[:,0,0]

    plt.plot(rs, rs * delta_sigmas/1e6)
    plt.xlabel(r'$ r $')
    plt.ylabel(r'$ r \Delta \Sigma (10^6 \, M_{\odot} / \mathrm{{pc}}) $')
    plt.xscale('log')

    plt.savefig('figs/test/delta_sigma_r_comoving_false.svg')
    plt.gcf().clear()


def test_power_spectrum():
    stacked_model = StackedModel()
    #Need to check plots on this one!
    stacked_model.calc_power_spect()

    z = 0
    ks = stacked_model.ks

    power_spect_redshift0 = stacked_model.power_spect
    plt.plot(ks, power_spect_redshift0.T)
    plt.xlabel(r'$ k/h $')
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
    rho_matter = (stacked_model.cosmo_params.rho_crit
                  * stacked_model.cosmo_params.omega_matter
                  / h**2)

    stacked_model.mu_szs = np.linspace(10, 16, 20)
    stacked_model.mus = np.linspace(10, 16, 20)

    z = 0
    mink = 1e-4
    maxks = [1, 3, 5, 10]
    for maxk in maxks:
        stacked_model.min_k = mink
        stacked_model.max_k = maxk

        stacked_model.calc_power_spect()

        masses = stacked_model.mass(stacked_model.mus)
        dn_dlnms = stacked_model.dnumber_dlogmass() #masses, zs
        dn_dms = dn_dlnms[0, :]/masses


        plotlabel = rf'$k_{{\mathrm{{max}}}}={maxk}$'
        plt.plot(masses, masses**2 * dn_dms / rho_matter, label=plotlabel)

    plt.title(rf'$z = {z}$, ppf {used_ppf}')
    plt.xlabel(r'$ M \; (M_{\odot}/h) $')
    plt.ylabel(r'$ M^2/\rho_m \; dn/dM$')
    plt.legend(loc='best')
    plt.ylim((4e-4, 3e-1))
    plt.xscale('log')
    plt.yscale('log')

    filename = f'dn_dlnm_redshift{z}-ppf_{str(used_ppf)}.svg'
    plt.savefig('figs/test/' + filename)
    plt.gcf().clear()

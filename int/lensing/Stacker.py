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
from maszcal.lensing import Stacker
from maszcal.cosmology import CosmoParams


mus = np.array([np.log(1e15)])
zs = np.linspace(0, 2, 20)


def test_power_spectrum():
    stacker = Stacker(
        mus,
        zs,
        delta=200,
        units=u.Msun/u.pc**2,
        sz_scatter=0.2,
        mass_definition='mean',
        comoving=True,
    )
    #Need to check plots on this one!
    stacker.calc_power_spect()

    z = 0
    ks = stacker.ks

    power_spect_redshift0 = stacker.power_spect
    plt.plot(ks, power_spect_redshift0.T)
    plt.xlabel(r'$ k $')
    plt.ylabel(r'$ P(z=0, k) $')
    plt.xscale('log')
    plt.yscale('log')

    plt.savefig('figs/test/power_redshift0.svg')
    plt.gcf().clear()


def test_comoving_vol():
    vols = stacker.comoving_vol()
    zs = stacker.redshift_bins

    plt.plot(zs, vols)
    plt.xlabel(r'$ z $')
    plt.ylabel(r'$ c r^2(z)/H(z)$')

    plt.savefig('figs/test/comoving_vol.svg')
    plt.gcf().clear()


used_ppf = True
#WMAP cosmology
cosmo_params = CosmoParams(
    hubble_constant=72,
    omega_bary_hsqr=0.024,
    omega_bary=0.024/(.72**2),
    omega_cdm_hsqr=0.14,
    omega_cdm=0.14/(.72**2),
    omega_matter=0.024/(.72**2)+0.14/(.72**2),
    omega_lambda=0.742,
    h=0.72,
    tau_reion=0.166,
    spectral_index=0.99,
    neutrino_mass_sum=0,
    use_ppf=used_ppf,
    flat=False,
)

stacker = Stacker(
    mus,
    zs,
    delta=200,
    units=u.Msun/u.pc**2,
    cosmo_params=cosmo_params,
    sz_scatter=0.2,
    comoving=True,
    mass_definition='mean',
)


def test_tinker_mf():
    h = stacker.cosmo_params.h
    rho_matter = (stacker.cosmo_params.rho_crit
                  * stacker.cosmo_params.omega_matter
                  / h**2)

    stacker.mu_bins = np.linspace(np.log(1e10), np.log(1e16), 100)
    stacker.mu_szs = np.linspace(np.log(1e10), np.log(1e16), 100)

    z = 0
    mink = 1e-4
    maxks = [1, 3, 5, 10]
    for maxk in maxks:
        stacker.ks = np.logspace(np.log10(mink), np.log10(maxk), 400)

        stacker.calc_power_spect()

        masses = stacker.mass(stacker.mu_bins)
        dn_dlnms = stacker.dnumber_dlogmass() #masses, zs
        dn_dms = dn_dlnms[:, 0]/masses


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

import pytest
import numpy as np
import camb
from astropy.cosmology import Planck15
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
from maszcal.tinker import TinkerBias
import maszcal.cosmology
import maszcal.matter


def describe_tinker_hmf():

    def describe_mdef_crit():

        @pytest.fixture
        def ks_and_power_spect():
            cosmo_params = maszcal.cosmology.CosmoParams()
            power = maszcal.matter.Power(cosmo_params=cosmo_params)
            zs = np.linspace(0, 1, 8)
            ks = np.logspace(-4, np.log10(0.3), 200)
            lin_spect = power.spectrum(ks, zs, is_nonlinear=False)
            return ks, lin_spect

        @pytest.fixture
        def bias_func():
            delta = 500
            mass_definition = 'crit'
            return TinkerBias(delta, mass_definition, astropy_cosmology=Planck15, comoving=True)

        def it_calculates_biases(bias_func, ks_and_power_spect):
            ks, power_spect = ks_and_power_spect
            masses = np.logspace(14, 15, 30)
            zs = np.linspace(0, 1, 8)
            biases = bias_func.bias(masses, zs, ks, power_spect)

            plt.plot(masses, biases)
            plt.xlabel(r'$M_{500c}$')
            plt.ylabel(r'$b(M)$')

            plt.xscale('log')

            plt.savefig('figs/test/halo_bias_tinker_mcrit.svg')
            plt.gcf().clear()

    def describe_mdef_mean():

        @pytest.fixture
        def ks_and_power_spect():
            cosmo_params = maszcal.cosmology.CosmoParams()
            power = maszcal.matter.Power(cosmo_params=cosmo_params)
            zs = np.linspace(0, 1, 8)
            ks = np.logspace(-4, np.log10(0.3), 200)
            lin_spect = power.spectrum(ks, zs, is_nonlinear=False)
            return ks, lin_spect

        @pytest.fixture
        def bias_func():
            delta = 200
            mass_definition = 'mean'
            return TinkerBias(delta, mass_definition, astropy_cosmology=Planck15, comoving=True)

        def it_calculates_biases(bias_func, ks_and_power_spect):
            ks, power_spect = ks_and_power_spect
            masses = np.logspace(14, 15, 30)
            zs = np.linspace(0, 1, 8)
            biases = bias_func.bias(masses, zs, ks, power_spect)

            plt.plot(masses, biases)
            plt.xlabel(r'$M_{200m}$')
            plt.ylabel(r'$b(M)$')

            plt.xscale('log')

            plt.savefig('figs/test/halo_bias_tinker_mmean.svg')
            plt.gcf().clear()

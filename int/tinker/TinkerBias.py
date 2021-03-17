import pytest
import numpy as np
import camb
from astropy.cosmology import Planck15
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import seaborn as sns
sns.set(style='whitegrid', font_scale=1.5, rc={"lines.linewidth": 2,'lines.markersize': 8.0,})
from maszcal.cosmology import CosmoParams
from maszcal.tinker import TinkerBias
import maszcal.interp_utils
import maszcal.cosmology
import maszcal.matter


def describe_TinkerBias():

    def describe_bias_mdef_crit():

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
            power_spect = power_spect[0, :]
            masses = np.logspace(14, 15, 30)
            zs = np.zeros(30)
            biases = bias_func.bias(masses, zs, ks, power_spect[None, :])

            plt.plot(masses, biases)
            plt.xlabel(r'$M_{500c}$')
            plt.ylabel(r'$b(M)$')

            plt.xscale('log')

            plt.savefig('figs/test/halo_bias_tinker_mcrit.svg')
            plt.gcf().clear()

    def describe_bias_mdef_mean():

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
            power_spect = power_spect[0, :]
            masses = np.logspace(14, 15, 30)
            zs = np.zeros(30)
            biases = bias_func.bias(masses, zs, ks, power_spect[None, :])

            plt.plot(masses, biases)
            plt.xlabel(r'$M_{200m}$')
            plt.ylabel(r'$b(M)$')

            plt.xscale('log')

            plt.savefig('figs/test/halo_bias_tinker_mmean.svg')
            plt.gcf().clear()


    def describe_plot_as_function_of_nu():

        @pytest.fixture
        def cosmo_params():
            return maszcal.cosmology.CosmoParams()

        @pytest.fixture
        def tinker_bias(cosmo_params):
            return maszcal.tinker.TinkerBias(
                delta=200,
                mass_definition='mean',
                astropy_cosmology=maszcal.cosmo_utils.get_astropy_cosmology(cosmo_params),
                comoving=True,
            )

        @pytest.fixture
        def power_spectrum_and_ks(cosmo_params):
            zs = np.zeros(1)
            ks = np.logspace(-4, np.log10(100), 400)
            return maszcal.matter.Power(cosmo_params).spectrum(ks, zs, is_nonlinear=False), ks

        def it_matches_tinker_etal_2010(tinker_bias, power_spectrum_and_ks):
            power_spectrum, ks = power_spectrum_and_ks
            delta_collapse = 1.686
            zs = np.zeros(1)
            masses = np.logspace(np.log10(8e13), np.log10(6e15), 60)
            radii = tinker_bias.radius_from_mass(masses[:, None], zs)
            sigmas = np.sqrt(maszcal.tinker.sigma_sq_integral(radii, power_spectrum, ks))
            nus = delta_collapse/sigmas

            bias = tinker_bias.bias(masses, zs, ks, power_spectrum)

            plt.plot(nus.squeeze(), bias.squeeze())
            plt.xticks(np.arange(2, 5, 1))
            plt.yticks(np.arange(2, 13, 1))
            plt.xlabel(r'$\nu$')
            plt.ylabel(r'$b$')
            plt.gcf().set_size_inches(4, 5)

            plt.savefig('figs/test/halo_bias_tinker_of_nu.svg')
            plt.gcf().clear()

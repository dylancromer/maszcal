import pytest
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import seaborn as sns
sns.set(style='whitegrid', font_scale=1.5, rc={"lines.linewidth": 2,'lines.markersize': 8.0,})
import maszcal.cosmology
import maszcal.matter


def describe_power():

    def it_calculates_the_matter_power_spectrum_with_camb():
        cosmo_params = maszcal.cosmology.CosmoParams()

        power = maszcal.matter.Power(cosmo_params=cosmo_params)

        zs = np.linspace(0, 2, 20)
        ks = np.logspace(-4, 1, 400)

        lin_spect = power.spectrum(ks, zs, is_nonlinear=False)
        nonlin_spect = power.spectrum(ks, zs, is_nonlinear=True)

        assert not np.any(np.isnan(lin_spect))
        assert not np.any(np.isnan(nonlin_spect))

        plt.plot(ks, lin_spect.T)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$ k/h $')
        plt.ylabel(r'$ P(z, k) $')
        plt.savefig('figs/test/linear_matter_power_spect.svg')
        plt.gcf().clear()

        plt.plot(ks, nonlin_spect.T)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$ k/h $')
        plt.ylabel(r'$ P(z, k) $')
        plt.savefig('figs/test/nonlinear_matter_power_spect.svg')
        plt.gcf().clear()

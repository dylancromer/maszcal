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
from maszcal.lensing import MatchingCmBaryonShearModel
import maszcal.cosmology


def describe_MatchingCmBaryonModel():

    def describe_stacked_delta_sigma():

        @pytest.fixture
        def baryon_model():
            NUM_CLUSTERS = 10
            sz_masses = 2e13*np.random.randn(NUM_CLUSTERS) + 2e14
            zs = np.random.rand(NUM_CLUSTERS)
            weights = np.random.rand(NUM_CLUSTERS)
            cosmo_params = maszcal.cosmology.CosmoParams()
            return MatchingCmBaryonShearModel(
                sz_masses=sz_masses,
                redshifts=zs,
                lensing_weights=weights,
                cosmo_params=cosmo_params,
            )

        def the_plots_look_right(baryon_model):
            radii = np.logspace(-1, 1, 30)
            alphas = 0.5*np.ones(1)
            betas = np.linspace(2.8, 3.2, 3)
            gammas = 0.5*np.ones(1)
            a_szs = 0.3*np.ones(1)

            esds = baryon_model.stacked_delta_sigma(radii, alphas, betas, gammas, a_szs)

            plt.plot(radii, radii[:, None]*esds[0, ...])
            plt.xscale('log')

            plt.xlabel(r'$R$')
            plt.ylabel(r'$R \Delta\Sigma(R)$')

            plt.savefig('figs/test/matching_cm_stacked_gnfw_delta_sigma.svg')
            plt.gcf().clear()

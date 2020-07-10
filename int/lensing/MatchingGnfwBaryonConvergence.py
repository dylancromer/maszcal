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
import projector
from maszcal.lensing import MatchingGnfwBaryonConvergence
import maszcal.cosmology
import maszcal.nfw


def describe_MatchingBaryonModel():

    def describe_kappa():

        @pytest.fixture
        def convergence():
            cosmo_params = maszcal.cosmology.CosmoParams()
            return MatchingGnfwBaryonConvergence(
                cosmo_params=cosmo_params,
                mass_definition='mean',
                delta=200,
                units=u.Msun/u.pc**2,
                comoving_radii=True,
                nfw_class=maszcal.nfw.MatchingNfwModel,
                sd_func=projector.sd,
            )

        def the_plots_look_right(convergence):
            radii = np.logspace(-1, np.log10(2), 30)
            cons = 3*np.ones(1)
            alphas = 0.9*np.ones(1)
            betas = 3.8*np.ones(1)
            gammas = 0.2*np.ones(1)

            NUM_CLUSTERS = 3
            mus = np.linspace(np.log(0.8e14), np.log(1.4e14), NUM_CLUSTERS)
            zs = np.ones(NUM_CLUSTERS) * 0.123

            rhos = convergence.rho_tot(radii, zs, mus, cons, alphas, betas, gammas).squeeze()
            kappas = convergence.kappa_total(radii, zs, mus, cons, alphas, betas, gammas).squeeze()

            plt.plot(radii, radii[:, None]**2 * rhos)
            plt.xscale('log')

            plt.xlabel(r'$R$')
            plt.ylabel(r'$R^2 \; \rho(R)$')

            plt.savefig('figs/test/gnfw_total_rho.svg')
            plt.gcf().clear()

            plt.plot(radii, radii[:, None]*kappas.T)
            plt.xscale('log')

            plt.xlabel(r'$R$')
            plt.ylabel(r'$R \; \kappa(R)$')

            plt.savefig('figs/test/gnfw_total_kappa.svg')
            plt.gcf().clear()

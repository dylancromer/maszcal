import pytest
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import seaborn as sns
sns.set(style='whitegrid', font_scale=1.5, rc={'lines.linewidth':2, 'lines.markersize':8.0})
from maszcal.concentration import ConModel


def _turn_on_logticks_seaborn():
        ax = plt.gca()
        ax.tick_params(which='both', bottom=True, left=True)
        locmaj = matplotlib.ticker.LogLocator(base=10, numticks=12)
        ax.xaxis.set_major_locator(locmaj)
        ax.yaxis.set_major_locator(locmaj)
        locmin = matplotlib.ticker.LogLocator(base = 10.0,
                                              subs = np.arange(2, 10) * .1,
                                              numticks = 100)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def describe_con_model():

    @pytest.fixture
    def con_model():
        return ConModel(mass_def='200c')

    @pytest.fixture
    def colors():
        return ["#ca5d47", "#b3943f", "#62a85a", "#5da3d1", "#706ecd"][::-1]

    def the_plot_looks_correct(con_model, colors):
        masses = np.logspace(9, 15, 50)
        redshifts = np.array([0, 0.5, 1, 2, 4])

        cons = con_model.c(masses, redshifts, '200c')

        for i, c in enumerate(cons.T):
            plt.plot(masses, c, label=rf'$z={redshifts[i]}$', linestyle='--', color=colors[i])

        plt.ylim((2.9, 11))

        plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=3)

        plt.xlabel(r'$M_{200c} \; (M_\odot/h)$')
        plt.ylabel(r'$c_{200c}$')

        _turn_on_logticks_seaborn()

        plt.xscale('log')
        plt.yscale('log')

        fig = plt.gcf()
        fig.set_size_inches(7, 6)

        plt.savefig('figs/test/concentration_diemer.svg')
        plt.gcf().clear()

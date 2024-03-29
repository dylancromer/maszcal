from dataclasses import dataclass
import pytest
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import seaborn as sns
sns.set(style='whitegrid', font_scale=1.5, rc={"lines.linewidth": 2,'lines.markersize': 8.0,})
import maszcal.interp_utils
import maszcal.twohalo
import maszcal.cosmology


def describe_TwoHaloShearModel():

    @pytest.fixture
    def two_halo_model():
        cosmo = maszcal.cosmology.CosmoParams()
        model = maszcal.twohalo.TwoHaloShearModel(cosmo_params=cosmo)
        return model

    def it_calculates_two_halo_esds(two_halo_model):
        zs = np.linspace(0, 1, 4)
        mus = np.ones(4) * np.log(1e14)
        rs = np.logspace(-3, 3, 200)

        esds = two_halo_model.excess_surface_density(rs, zs, mus)

        assert not np.any(np.isnan(esds))
        assert np.all(np.abs(rs[None, :]*esds) < 500)
        assert esds.shape == zs.shape + rs.shape

        plt.plot(rs, esds.T)
        plt.xscale('log')
        plt.xlabel(r'$R \; (\mathrm{Mpc}$)')
        plt.ylabel(r'$\Delta \Sigma \; (M_\odot/\mathrm{pc}^2)$')
        plt.savefig('figs/test/two_halo_esd.svg')

        plt.gcf().clear()

    def it_calculates_halo_matter_correlations(two_halo_model):
        zs = np.linspace(0, 1, 10)
        mus = np.ones(1) * np.log(1e14)
        rs = np.logspace(-2, 3, 600)

        xis = two_halo_model.halo_matter_correlation(rs, zs, mus)

        assert not np.any(np.isnan(xis))
        assert xis.shape == zs.shape + rs.shape

        plt.plot(rs, rs[:, None]**2 * xis.T)
        plt.xscale('log')
        plt.xlabel(r'$R \; (\mathrm{Mpc}$)')
        plt.ylabel(r'$\xi$')
        plt.savefig('figs/test/halo_matter_correlation.svg')

        plt.gcf().clear()


def describe_TwoHaloConvergenceModel():

    @pytest.fixture
    def two_halo_model():
        cosmo = maszcal.cosmology.CosmoParams()
        model = maszcal.twohalo.TwoHaloConvergenceModel(
            cosmo_params=cosmo,
            sd_kwargs={'radial_axis_to_broadcast': 1, 'density_axis': -1},
        )
        return model

    def it_calculates_two_halo_convs(two_halo_model):
        zs = np.linspace(0.1, 1, 4)
        mus = np.ones(4) * np.log(1e14)
        from_arcmin = 2 * np.pi / 360 / 60
        to_arcmin = 1/from_arcmin
        thetas = np.geomspace(1e-2*from_arcmin, 360*from_arcmin, 200)

        kappas = two_halo_model.convergence(thetas, zs, mus)

        assert not np.any(np.isnan(kappas))
        assert np.all(np.abs(kappas) < 1)
        assert kappas.shape == zs.shape + thetas.shape

        plt.plot(thetas*to_arcmin, thetas[:, None] * kappas.T)
        plt.xscale('log')
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$\theta \kappa(\theta)$')
        plt.savefig('figs/test/two_halo_theta_times_kappa.svg')

        plt.gcf().clear()

        plt.plot(thetas*to_arcmin, kappas.T)
        plt.xscale('log')
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$\kappa(\theta)$')
        plt.savefig('figs/test/two_halo_kappa.svg')

        plt.gcf().clear()

    def it_calculates_two_halo_radius_space_convs(two_halo_model):
        zs = np.linspace(0.1, 1, 4)
        mus = np.ones(4) * np.log(1e14)
        rs = np.geomspace(0.1, 60, 30)
        kappas = two_halo_model.radius_space_convergence(rs, zs, mus)

        assert not np.any(np.isnan(kappas))
        assert np.all(np.abs(kappas) < 1)
        assert kappas.shape == zs.shape + rs.shape

        plt.plot(rs, kappas.T)
        plt.xscale('log')
        plt.xlabel(r'$r$')
        plt.ylabel(r'$\kappa(r)$')
        plt.savefig('figs/test/two_halo_kappa_of_r.svg')

        plt.gcf().clear()


def describe_TwoHaloEmulator():

    def describe_context_esd():

        @pytest.fixture
        def two_halo_esd():
            cosmo = maszcal.cosmology.CosmoParams()
            model = maszcal.twohalo.TwoHaloShearModel(cosmo_params=cosmo)
            return model.excess_surface_density

        @pytest.fixture
        def esd_emulator(two_halo_esd):
            return maszcal.twohalo.TwoHaloEmulator.from_function(
                two_halo_func=two_halo_esd,
                r_grid=np.geomspace(0.01, 100, 120),
                z_lims=np.array([0, 1.1]),
                mu_lims=np.log(np.array([1e13, 1e15])),
                num_emulator_samples=800,
            )

        def it_can_calculate_esds(esd_emulator):
            rs = np.geomspace(0.1, 60, 30)

            rng = np.random.default_rng(seed=13)
            zs = np.linspace(0.1, 1, 4)
            mus = np.ones(4) * np.log(1e14)

            esds = esd_emulator(rs, zs, mus)
            assert not np.any(np.isnan(esds))
            assert esds.shape == zs.shape + rs.shape

            plt.plot(rs, esds.T)
            plt.xscale('log')
            plt.xlabel(r'$R \; (\mathrm{Mpc}$)')
            plt.ylabel(r'$\Delta \Sigma \; (M_\odot/\mathrm{pc}^2)$')
            plt.savefig('figs/test/emulated_two_halo_esd.svg')

            plt.gcf().clear()

    def describe_context_convergence():

        @pytest.fixture
        def two_halo_conv():
            cosmo = maszcal.cosmology.CosmoParams()
            model = maszcal.twohalo.TwoHaloConvergenceModel(cosmo_params=cosmo)
            return model.radius_space_convergence

        @pytest.fixture
        def conv_emulator(two_halo_conv):
            return maszcal.twohalo.TwoHaloEmulator.from_function(
                two_halo_func=two_halo_conv,
                r_grid=np.geomspace(0.01, 100, 120),
                z_lims=np.array([0, 1.1]),
                mu_lims=np.log(np.array([1e13, 1e15])),
                num_emulator_samples=800,
            )

        def it_can_calculate_convergence(conv_emulator):
            rs = np.geomspace(0.1, 60, 30)

            rng = np.random.default_rng(seed=13)
            zs = np.linspace(0.1, 1, 4)
            mus = np.ones(4) * np.log(1e14)

            convs = conv_emulator(rs, zs, mus)
            assert not np.any(np.isnan(convs))
            assert convs.shape == zs.shape + rs.shape

            plt.plot(rs, rs[:, None] * convs.T)
            plt.xscale('log')
            plt.xlabel(r'$r$')
            plt.ylabel(r'$r \kappa(r)$')
            plt.savefig('figs/test/emulated_two_halo_r_times_kappa.svg')

            plt.gcf().clear()

            plt.plot(rs, convs.T)
            plt.xscale('log')
            plt.xlabel(r'$r$')
            plt.ylabel(r'$\kappa(r)$')
            plt.savefig('figs/test/emulated_two_halo_kappa.svg')

            plt.gcf().clear()

    def describe_non_matching_version():

        @pytest.fixture
        def two_halo_esd():
            cosmo = maszcal.cosmology.CosmoParams()
            model = maszcal.twohalo.TwoHaloShearModel(cosmo_params=cosmo)
            return model.excess_surface_density

        @pytest.fixture
        def esd_emulator(two_halo_esd):
            return maszcal.twohalo.TwoHaloEmulator.from_function(
                two_halo_func=two_halo_esd,
                r_grid=np.geomspace(0.01, 100, 120),
                z_lims=np.array([0, 1.1]),
                mu_lims=np.log(np.array([1e13, 1e15])),
                num_emulator_samples=800,
                separate_mu_and_z_axes=True,
            )

        def it_can_calculate_esds(esd_emulator):
            rs = np.geomspace(0.1, 30, 60)

            rng = np.random.default_rng(seed=13)
            mus = np.linspace(np.log(1e14), np.log(1e15), 4)
            zs = 0.5*np.ones(5)

            esds = esd_emulator(rs, zs, mus)
            assert not np.any(np.isnan(esds))
            assert esds.shape == mus.shape + zs.shape + rs.shape

            plt.plot(rs, esds[:, 0, :].T)
            plt.plot(rs, esds[0, :, :].T, linestyle=':')
            # This plot should show the dotted, redshift-varying lines to be all identical and located on the smallest mu line
            plt.xscale('log')
            plt.xlabel(r'$R \; (\mathrm{Mpc}$)')
            plt.ylabel(r'$\Delta \Sigma \; (M_\odot/\mathrm{pc}^2)$')
            plt.savefig('figs/test/non_matching_emulated_two_halo_esd.svg')

            plt.gcf().clear()

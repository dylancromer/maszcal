import numpy as np
import pytest
import astropy.units as u
from maszcal.density import NfwModel
import projector


def describe_nfw_model():
    """
    This is really an integration test of projector with maszcal.nfw
    """

    @pytest.fixture
    def nfw_model():
        return NfwModel()

    def its_densities_integrate_to_match_its_esds(nfw_model):
        rs = np.logspace(-1, 1, 10)[:, None]
        zs = np.linspace(0, 1, 3)
        masses = np.logspace(14, 15, 5)
        cons = np.linspace(2, 4, 6)

        ds = nfw_model.excess_surface_density(rs, zs, masses, cons)

        def rho_func(r):
            return nfw_model.rho(r, zs, masses, cons)

        esds = projector.ExcessSurfaceDensity.calculate(
            rs,
            rho_func,
            radial_axis_to_broadcast=1,
            density_axis=-2,
        ) * (u.Msun/u.Mpc**2).to(u.Msun/u.pc**2)

        assert np.allclose(ds, esds, rtol=1e-2)

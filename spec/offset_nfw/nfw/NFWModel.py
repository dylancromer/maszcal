import pytest
import numpy as np
from maszcal.offset_nfw.nfw import NFWModel
from maszcal.cosmo_utils import get_astropy_cosmology
from maszcal.cosmology import CosmoParams


def describe_nfw_model():

    def describe_sigma_theory():

        @pytest.fixture
        def nfw_model():
            astropy_cosmology = get_astropy_cosmology(CosmoParams())
            return NFWModel(astropy_cosmology, comoving=True)

        def it_uses_the_right_shape_of_arrays(nfw_model):
            ms = np.logspace(14, 16, 2)
            zs = np.linspace(0, 2, 3)
            rs = np.logspace(-1, 1, 4)
            cs = np.linspace(2, 9, 5)

            sigmas = nfw_model.sigma_theory(rs, ms, cs, zs)
            assert sigmas.shape == (ms.size, zs.size, rs.size, cs.size)

    def describe_offset_sigma_theory():

        @pytest.fixture
        def nfw_model():
            astropy_cosmology = get_astropy_cosmology(CosmoParams())
            return NFWModel(astropy_cosmology, comoving=True)

        def it_uses_the_right_shape_of_arrays(nfw_model):
            ms = np.logspace(14, 16, 2)
            zs = np.linspace(0, 2, 3)
            rs = np.logspace(-1, 1, 4)
            cs = np.linspace(2, 9, 5)
            r_offsets = np.logspace(-4, -1, 6)
            theta_offsets = np.linspace(0, 2*np.pi, 7, endpoint=False)

            offset_sigmas = nfw_model.offset_sigma_theory(rs, r_offsets, theta_offsets, ms, cs, zs)
            assert offset_sigmas.shape == (ms.size, zs.size, rs.size, cs.size, r_offsets.size, theta_offsets.size)


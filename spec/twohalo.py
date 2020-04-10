from dataclasses import dataclass
import pytest
import numpy as np
import maszcal.twohalo
import maszcal.cosmology
import maszcal.interp_utils


@dataclass
class FakePower:
    cosmo_params: object

    def get_spectrum_interpolator(self, ks, zs, is_nonlinear):
        return lambda ks, zs: np.ones(zs.shape + ks.shape)


def fake_quad_vec(func, a, b, limit=1):
    return np.ones_like(func(a)), 1


def describe_TwoHaloShearModel():

    @pytest.fixture
    def two_halo_model(mocker):
        mocker.patch('maszcal.twohalo.scipy.integrate.quad_vec', new=fake_quad_vec)
        cosmo = maszcal.cosmology.CosmoParams()
        model = maszcal.twohalo.TwoHaloShearModel(cosmo_params=cosmo, matter_power_class=FakePower)

        model.NUM_INTERP_ZS = 3
        model.NUM_INTERP_RADII = 4

        return model

    def it_calculates_two_halo_esds(two_halo_model):
        zs = np.linspace(0, 1, 4)
        mus = np.linspace(32, 33, 3)
        rs = np.logspace(-1, 1, 2)

        esds = two_halo_model.esd(rs, mus, zs)

        assert np.all(esds >= 0)
        assert not np.any(np.isnan(esds))
        assert esds.shape == mus.shape + zs.shape + rs.shape


@dataclass
class FakeTwoHaloShearModel:
    cosmo_params: object
    units: object
    delta: int
    mass_definition: str
    comoving: bool
    matter_power_class: object

    def _esd_radial_shape(self, rs, zs):
        return np.ones(rs.shape + zs.shape)

    def _bias(self, mus, zs):
        return np.ones(mus.shape + zs.shape)


def describe_EmulatedTwoHaloShear():

    @pytest.fixture
    def emulated_model():
        zs = np.linspace(0, 1, 8)
        rs = np.logspace(-1, 1, 8)

        emu = maszcal.twohalo.EmulatedTwoHaloShear(
            rs,
            zs,
            cosmo_params=maszcal.cosmology.CosmoParams(),
            two_halo_shear_class=FakeTwoHaloShearModel,
        )
        emu.process()
        return emu

    def it_emulated_two_halo_esds(emulated_model):
        zs = np.linspace(0.1, 0.9, 10)
        mus = np.linspace(32, 33, 10)
        rs = np.logspace(np.log10(0.2), np.log10(9), 10)

        esds = emulated_model.esd(rs, mus, zs)

        assert np.allclose(esds, 1, rtol=1e-1)
        assert not np.any(np.isnan(esds))
        assert esds.shape == mus.shape + zs.shape + rs.shape

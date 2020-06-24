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


@dataclass
class FakePowerInterpolator:
    nonlinear: bool

    def P(self, zs, ks):
        return (ks**4)[None, :] * np.ones(zs.shape + ks.shape)


@dataclass
class FakeCambResults:
    nonlinear: bool

    def calc_power_spectra(self):
        pass

    def get_matter_power_interpolator(self, **kwargs):
        return FakePowerInterpolator(nonlinear=self.nonlinear)


def fake_camb_get_results(params):
    if params.NonLinear == 'NonLinear_none':
        return FakeCambResults(nonlinear=False)
    else:
        return FakeCambResults(nonlinear=True)


def describe_TwoHaloShearModel():

    @pytest.fixture
    def two_halo_model(mocker):
        mocker.patch('maszcal.twohalo.scipy.integrate.quad_vec', new=fake_quad_vec)
        mocker.patch('maszcal.matter.camb.get_results', new=fake_camb_get_results)
        cosmo = maszcal.cosmology.CosmoParams()
        model = maszcal.twohalo.TwoHaloShearModel(cosmo_params=cosmo, matter_power_class=FakePower)

        model.NUM_INTERP_ZS = 3
        model.NUM_INTERP_RADII = 4

        return model

    def it_calculates_two_halo_esds(two_halo_model):
        zs = np.linspace(0, 1, 4)
        mus = np.linspace(32, 33, 4)
        rs = np.logspace(-1, 1, 2)

        esds = two_halo_model.esd(rs, mus, zs)

        assert np.all(esds >= 0)
        assert not np.any(np.isnan(esds))
        assert esds.shape == zs.shape + rs.shape

    def it_calculates_halo_matter_correlations(two_halo_model):
        zs = np.linspace(0, 1, 4)
        mus = np.linspace(32, 33, 4)
        rs = np.logspace(-1, 1, 2)

        xis = two_halo_model.halo_matter_correlation(rs, mus, zs)

        assert np.all(xis >= 0)
        assert not np.any(np.isnan(xis))
        assert xis.shape == zs.shape + rs.shape

    def it_reshapes_the_correlator_correctly(two_halo_model):
        zs = np.linspace(0, 1, 4)
        rs_1 = np.logspace(-1, 1, 5)
        rs_2 = np.logspace(-1.1, 0.9, 5)
        rs_3 = np.logspace(-0.9, 1.1, 5)
        rs = np.stack((rs_1, rs_2, rs_3))

        xis_1 = two_halo_model._density_shape_interpolator(rs, zs)
        xis_2 = np.array([two_halo_model._density_shape_interpolator(r, zs) for r in rs])
        assert np.all(xis_1 == xis_2)

from dataclasses import dataclass
import pytest
import numpy as np
import maszcal.cosmology
import maszcal.matter


@dataclass
class FakePowerInterpolator:
    nonlinear: bool

    def P(self, zs, ks):
        if not self.nonlinear:
            return np.ones(zs.shape+ks.shape)
        else:
            return 2 * np.ones(zs.shape+ks.shape)


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


def describe_Power():

    def it_calculates_the_matter_power_spectrum_with_camb(mocker):
        mocker.patch('maszcal.matter.camb.get_results', new=fake_camb_get_results)

        cosmo_params = maszcal.cosmology.CosmoParams()

        power = maszcal.matter.Power(cosmo_params=cosmo_params)

        zs = np.linspace(0, 1, 5)
        ks = np.linspace(0.1, 1, 10)

        assert np.all(power.spectrum(ks, zs, is_nonlinear=False) == 1)
        assert np.all(power.spectrum(ks, zs, is_nonlinear=True) == 2)

    def it_provides_a_power_spectrum_interpolator(mocker):
        mocker.patch('maszcal.matter.camb.get_results', new=fake_camb_get_results)

        cosmo_params = maszcal.cosmology.CosmoParams()

        power = maszcal.matter.Power(cosmo_params=cosmo_params)

        zs = np.linspace(0, 1, 5)
        ks = np.linspace(0.1, 1, 10)

        lin_spectrum = power.get_spectrum_interpolator(ks, zs, is_nonlinear=False)
        nonlin_spectrum = power.get_spectrum_interpolator(ks, zs, is_nonlinear=True)

        assert np.all(lin_spectrum(ks, zs) == 1)
        assert np.all(nonlin_spectrum(ks, zs) == 2)


def describe_Correlations():

    @pytest.fixture
    def ks_zs_and_power_spectrum():
        return np.logspace(-2, 40, 100), np.linspace(0, 1, 10), np.ones((10, 100))

    def it_converts_the_power_spectrum_to_a_correlator(ks_zs_and_power_spectrum):
        ks, zs, ps = ks_zs_and_power_spectrum
        xi_interpolator = maszcal.matter.Correlations.from_power_spectrum(ks, zs, ps)

        rs = np.logspace(-1, 1, 30)
        xis = xi_interpolator(rs)
        assert xis.shape == (10, 30)

    def it_can_calculate_a_correlator_from_just_a_cosmology(mocker):
        mocker.patch('maszcal.matter.camb.get_results', new=fake_camb_get_results)

        cosmo = maszcal.cosmology.CosmoParams()
        zs = np.linspace(0, 1, 10)
        xi_interpolator = maszcal.matter.Correlations.from_cosmology(cosmo, zs, is_nonlinear=True)

        rs = np.logspace(-1, 1, 30)
        xis = xi_interpolator(rs)
        assert xis.shape == (10, 30)

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
            return 'linear spectrum'
        else:
            return 'nonlinear spectrum'


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


def describe_power():

    def it_calculates_the_matter_power_spectrum_with_camb(mocker):
        mocker.patch('maszcal.matter.camb.get_results', new=fake_camb_get_results)

        cosmo_params = maszcal.cosmology.CosmoParams()

        power = maszcal.matter.Power(cosmo_params=cosmo_params)

        zs = np.linspace(0, 1, 5)
        ks = np.linspace(0.1, 1, 10)

        assert np.all(power.spectrum(ks, zs, is_nonlinear=False) == 'linear spectrum')
        assert np.all(power.spectrum(ks, zs, is_nonlinear=True) == 'nonlinear spectrum')

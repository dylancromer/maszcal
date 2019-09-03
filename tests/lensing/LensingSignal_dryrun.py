import pytest
import numpy as np
from maszcal.lensing import LensingSignal
from maszcal.model import StackedModel


def describe_LensingSignal():

    def describe_single_mass_esd():

        def it_can_use_different_mass_definitions():
            mu = np.array([np.log(1e15)])
            con = np.array([3])
            rs = np.logspace(-1, 1, 5)

            redshift = 0.4*np.ones(1)
            delta = 500
            mass_def = 'crit'
            lensing_signal = LensingSignal(redshifts=redshift, delta=delta, mass_definition=mass_def)

            esd_500c = lensing_signal.single_mass_esd(rs, np.array([[mu, con]]))

            delta = 200
            kind = 'mean'
            lensing_signal = LensingSignal(redshifts=redshift, delta=delta, mass_definition=mass_def)

            esd_200m = lensing_signal.single_mass_esd(rs, np.array([[mu, con]]))

            assert np.all(esd_200m < esd_500c)

    def describe_stacked_esd():

        def it_matches_the_stacked_model():
            mus = np.linspace(np.log(1e14), np.log(1e15), 10)
            redshifts = np.linspace(0, 2, 5)
            rs = np.logspace(-1, 1, 5)

            con = 3
            a_sz = 0
            params = np.array([[con, a_sz]])

            lensing_signal = LensingSignal(log_masses=mus, redshifts=redshifts)
            stacked_model = StackedModel(mu_bins=mus, redshift_bins=redshifts)

            esd_lensing_signal = lensing_signal.stacked_esd(rs, params)
            esd_stacked_model = stacked_model.delta_sigma(rs, np.array([con]), np.array([a_sz]))

            assert np.all(esd_stacked_model == esd_lensing_signal)

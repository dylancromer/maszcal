import pytest
import numpy as np
from maszcal.lensing import LensingSignal


def describe_LensingSignal():

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

import pytest
import numpy as np
from maszcal.lensing import LensingSignal




def describe_LensingSignal():

    @pytest.fixture
    def lensing_signal():
        return LensingSignal(comoving=False)

    def it_does_not_return_nans(lensing_signal):
        rs = np.logspace(-1, 1, 21)
        params = np.array([[4, -1]])

        esd = lensing_signal.stacked_esd(rs, params)

        assert not np.any(np.isnan(esd))

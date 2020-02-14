import pytest
import numpy as np
from maszcal.data.template import WeakLensingData


def describe_weak_lensing_data():

    @pytest.fixture
    def wl_data():
        return WeakLensingData(
            radii = np.logspace(-1, 1, 10),
            redshifts = np.linspace(0, 1, 5),
            wl_signals = np.ones((10, 5, 3)),
        )

    def it_must_be_inited_with_data(wl_data):
        assert wl_data.radii is not None

        with pytest.raises(TypeError):
            WeakLensingData()

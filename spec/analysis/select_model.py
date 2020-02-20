import pytest
import numpy as np
import maszcal.data.templates
import maszcal.lensing
from maszcal.analysis import select_model


def describe_select_model():

    @pytest.fixture
    def data():
        return maszcal.data.templates.WeakLensingData(
            radii=np.logspace(-1, 0, 10),
            redshifts=np.zeros(1),
            wl_signals=np.ones((10, 1)),
        )

    def it_returns_an_nfw_single_mass_model_when_asked_for_one(data):
        model = select_model(data=data, model='nfw', cm_relation=False, emulation=False, stacked=False)

        assert isinstance(model, maszcal.lensing.SingleMassNfwLensingSignal)

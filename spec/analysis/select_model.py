import pytest
import numpy as np
import maszcal.data.templates
import maszcal.model
import maszcal.lensing
from maszcal.analysis import select_model


def describe_select_model():

    @pytest.fixture
    def data():
        return maszcal.data.templates.WeakLensingData(
            radii=np.logspace(-1, 0, 10),
            redshifts=np.zeros(1),
            wl_signals=np.ones((10, 1, 1)),
            covariances=np.ones((10, 10, 1)),
        )

    def it_returns_the_right_model_when_asked(data):
        model = select_model(data=data, model='nfw', cm_relation=False, emulation=False, stacked=False)
        assert isinstance(model, maszcal.model.SingleMass)
        assert model.lensing_signal_class is maszcal.lensing.SingleMassNfwLensingSignal

        model = select_model(data=data, model='nfw', cm_relation=True, emulation=False, stacked=False)
        assert isinstance(model, maszcal.model.SingleMass)
        assert model.lensing_signal_class is maszcal.lensing.SingleMassNfwLensingSignal

        # model = select_model(data=data, model='baryon', cm_relation=False, emulation=False, stacked=False)
        # assert isinstance(model, maszcal.lensing.SingleBaryonLensingSignal)

    def it_errors_properly_when_a_bad_model_is_selected(data):
        with pytest.raises(ValueError):
            select_model(data=data, model='iuerie3ewfhjebf', cm_relation=False, emulation=False, stacked=False)

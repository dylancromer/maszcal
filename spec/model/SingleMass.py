import pytest
import numpy as np
import maszcal.model
import maszcal.data.templates
from maszcal.lensing import SingleMassNfwLensingSignal


class FakeSingleMassNfwLensingSignal:
    def __init__(
        self,
        redshift=None,
        units=None,
        comoving=True,
        delta=200,
        mass_definition='mean',
        cosmo_params=None,
    ):
        self.redshift = redshift

    def esd(self, rs, params):
        return np.ones((1, 1, rs.size, 1))


def fake_minimizer(func, param_mins, param_maxes, method):
    return np.array([3, np.log(1e14)])


def describe_single_mass():

    def describe_esd():

        @pytest.fixture
        def nfw_model():
            return maszcal.model.SingleMass(
                lensing_signal_class=FakeSingleMassNfwLensingSignal,
                cm_relation=False,
            )

        def it_calculates_an_esd(nfw_model):
            rs = np.logspace(-1, 1, 4)
            params = np.array([3, np.log(1e14)])

            esd = nfw_model.esd(rs, params)

            assert np.all(esd == np.ones((1, 1, rs.size, 1)))
            assert esd.shape == (1, 1, 4, 1)

    def describe_get_best_fit():

        @pytest.fixture
        def nfw_model():
            return maszcal.model.SingleMass(
                lensing_signal_class=FakeSingleMassNfwLensingSignal,
                cm_relation=False,
                minimize_func=fake_minimizer,
            )

        @pytest.fixture
        def data(nfw_model):
            rs = np.logspace(-1, 1, 4)
            zs = np.zeros(1)
            params = np.array([3, np.log(1e14)])

            esd = nfw_model.esd(rs, params)
            covariance = np.identity(4)/1e6

            return maszcal.data.templates.WeakLensingData(
                radii=rs,
                redshifts=zs,
                wl_signals=esd[0, 0, :, :],
                masses=np.ones((1, 1))
            )

        def it_gets_the_best_fit_for_the_input_data(nfw_model, data):
            param_mins = np.array([0, np.log(5e12)])
            param_maxes = np.array([6, np.log(5e15)])

            best_fit = nfw_model.get_best_fit(data, param_mins, param_maxes)

            assert np.allclose(best_fit, np.array([3, np.log(1e14)]))

    def describe_cm_relation():

        @pytest.fixture
        def nfw_model():
            return maszcal.model.SingleMass(
                lensing_signal_class=SingleMassNfwLensingSignal,
                cm_relation=True,
            )

        def it_can_use_a_cm_relation_for_the_esd(nfw_model):
            rs = np.logspace(-1, 1, 4)
            params = np.array([np.log(1e14)])

            esd = nfw_model.esd(rs, params)

            assert np.all(esd > 0)
            assert esd.shape == (1, 1, 4, 1)

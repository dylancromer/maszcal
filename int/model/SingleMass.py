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
        return rs[:, None, None] + params[None, 0, None]**2 + params[None, 1, None]**2


def describe_single_mass():

    def describe_get_best_fit():

        @pytest.fixture
        def nfw_model():
            return maszcal.model.SingleMass(lensing_signal_class=SingleMassNfwLensingSignal)

        @pytest.fixture
        def data_and_true_params(nfw_model):
            rs = np.logspace(-1, 1, 4)
            zs = np.zeros(1)

            con = 5 * np.random.rand()
            mu = np.random.rand() + np.log(1e13)
            params = np.array([con, mu])

            esd = nfw_model.esd(rs, params)
            covariance = np.identity(4)/1e6

            return maszcal.data.templates.WeakLensingData(
                radii=rs,
                redshifts=zs,
                wl_signals=esd[0, 0, :, :],
                covariances=covariance,
            ), params

        def it_gets_the_best_fit_for_the_input_data(nfw_model, data_and_true_params):
            data, true_params = data_and_true_params

            param_mins = np.array([0, np.log(5e12)])
            param_maxes = np.array([6, np.log(5e15)])

            best_fit = nfw_model.get_best_fit(data, param_mins, param_maxes)

            assert np.allclose(best_fit, true_params)

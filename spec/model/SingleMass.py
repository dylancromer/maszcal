import pytest
import numpy as np
import maszcal.model
import maszcal.data.templates


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
        return np.ones(rs.shape +self.redshift.shape + (1,))


def describe_single_mass():

    def describe_esd():

        @pytest.fixture
        def nfw_model():
            return maszcal.model.SingleMass(lensing_signal_class=FakeSingleMassNfwLensingSignal)

        def it_calculates_an_esd(nfw_model):
            rs = np.logspace(-1, 1, 4)
            params = np.array([3, np.log(1e14)])

            esd = nfw_model.esd(rs, params)

            assert np.all(esd == np.ones(rs.shape + params.shape))

    def describe_get_best_fit():

        @pytest.fixture
        def nfw_model():
            return maszcal.model.SingleMass(lensing_signal_class=FakeSingleMassNfwLensingSignal)

        @pytest.fixture
        def data(nfw_model):
            rs = np.logspace(-1, 1, 4)
            zs = np.zeros(1)
            params = np.array([3, np.log(1e14)])

            esd = nfw_model.esd(rs, params)
            covariance = np.identity(4)

            return maszcal.data.templates.WeakLensingData(
                radii=rs,
                redshifts=zs,
                wl_signals=esd,
                covariances=covariance,
            )

        @pytest.fixture
        def data_alt(nfw_model):
            rs = np.logspace(-1, 1, 4)
            zs = np.zeros(1)
            params = np.array([3.01, 2.008e14])

            esd = nfw_model.esd(rs, params)
            covariance = np.identity(4)

            return maszcal.data.templates.WeakLensingData(
                radii=rs,
                redshifts=zs,
                wl_signals=esd,
                covariances=covariance,
            )

        def it_gets_the_best_fit_for_the_input_data(nfw_model, data, data_alt):
            param_mins = np.array([0, np.log(5e12)])
            param_maxes = np.array([5, np.log(5e15)])

            best_fit = nfw_model.get_best_fit(data, param_mins, param_maxes)

            assert np.allclose(best_fit, np.array([3, np.log(1e14)]))

            alt_best_fit = nfw_model.get_best_fit(data_alt, param_mins, param_maxes)

            assert np.allclose(best_fit, np.array([3.01, 2.008e14]))

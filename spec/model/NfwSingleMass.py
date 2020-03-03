import pytest
import numpy as np
import maszcal.model
import maszcal.data.templates


def describe_nfw_single_mass():

    def describe_esd():

        @pytest.fixture
        def nfw_model():
            return maszcal.model.NfwSingleMass()

        def it_calculates_an_esd(nfw_model):
            rs = np.logspace(-1, 1, 4)
            params = np.array([3, 1e14])

            esd = nfw_model.esd(rs, params)

            assert np.all(esd > 0)

    def describe_get_best_fit():

        @pytest.fixture
        def nfw_model():
            return maszcal.model.NfwSingleMass()

        @pytest.fixture
        def data(nfw_model):
            rs = np.logspace(-1, 1, 4)
            zs = np.zeros(1)
            params = np.array([3, 1e14])

            esd = nfw_model.esd(rs, params)
            covariance = np.identity(4)

            return maszcal.data.templates.WeakLensingData(
                radii=rs,
                redshifts=zs,
                wl_signals=esd[..., None],
                covariances=covariance[..., None],
            )

        def it_gets_the_best_fit_for_the_input_data(nfw_model, data):
            best_fit = nfw_model.get_best_fit(data)

            assert np.allclose(best_fit, np.array([3, 1e14]))

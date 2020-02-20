import pytest
import numpy as np
from maszcal.data.templates import WeakLensingData


def describe_weak_lensing_data():

    def describe___init__():

        @pytest.fixture
        def wl_data():
            return WeakLensingData(
                radii=np.logspace(-1, 1, 10),
                redshifts=np.linspace(0, 1, 5),
                wl_signals=np.ones((10, 5, 3)),
            )

        def it_must_be_inited_with_data(wl_data):
            assert wl_data.radii is not None

            with pytest.raises(TypeError):
                WeakLensingData()

    def describe_select_redshift_index():

        @pytest.fixture
        def wl_data():
            return WeakLensingData(
                radii=np.logspace(-1, 1, 10),
                redshifts=np.linspace(0, 1, 5),
                wl_signals=np.ones((10, 5, 3)),
            )

        def it_returns_a_new_weak_lensing_data_instance_with_the_selected_redshift(wl_data):
            wl_at_redshift_index_3 = wl_data.select_redshift_index(3)
            assert isinstance(wl_at_redshift_index_3, WeakLensingData)
            assert wl_at_redshift_index_3.redshifts == np.array([wl_data.redshifts[3]])

    def describe__data_are_consistent():

        @pytest.fixture
        def mismatched_data():
            num_radii = 3
            num_redshifts = 2
            wrong_num_radii = 4
            wrong_num_redshifts = 5

            radii = np.logspace(-1, 1, num_radii)
            redshifts = np.linspace(0, 1, num_redshifts)

            wl_wrong_radii = np.ones((wrong_num_radii, num_redshifts))
            wl_wrong_redshifts = np.ones((num_radii, wrong_num_redshifts))
            wl_wrong_both = np.ones((wrong_num_radii, wrong_num_redshifts))

            return radii, redshifts, wl_wrong_radii, wl_wrong_redshifts, wl_wrong_both

        def it_makes_sure_the_data_shapes_are_consistent(mismatched_data):
            radii, redshifts, wl_wrong_radii, wl_wrong_redshifts, wl_wrong_both = mismatched_data

            with pytest.raises(ValueError):
                WeakLensingData(
                    radii=radii,
                    redshifts=redshifts,
                    wl_signals=wl_wrong_radii,
                )

            with pytest.raises(ValueError):
                WeakLensingData(
                    radii=radii,
                    redshifts=redshifts,
                    wl_signals=wl_wrong_redshifts,
                )

            with pytest.raises(ValueError):
                WeakLensingData(
                    radii=radii,
                    redshifts=redshifts,
                    wl_signals=wl_wrong_both,
                )

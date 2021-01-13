import pytest
import numpy as np
from maszcal.data.templates import WeakLensingData


def describe_weak_lensing_data():

    def describe___init__():

        @pytest.fixture
        def wl_data():
            return WeakLensingData(
                radial_coordinates=np.logspace(-1, 1, 10),
                redshifts=np.linspace(0, 1, 5),
                wl_signals=np.ones((10, 5, 3)),
                masses = np.ones((5, 3)),
            )

        def it_must_be_inited_with_data(wl_data):
            assert wl_data.radial_coordinates is not None

            with pytest.raises(TypeError):
                WeakLensingData()

    def describe_select_redshift_index():

        @pytest.fixture
        def wl_data():
            return WeakLensingData(
                radial_coordinates=np.logspace(-1, 1, 10),
                redshifts=np.linspace(0, 1, 5),
                wl_signals=np.ones((10, 5, 3)),
                masses = np.ones((5, 3)),
            )

        def it_returns_a_new_weak_lensing_data_instance_with_the_selected_redshift(wl_data):
            wl_at_redshift_index_3 = wl_data.select_redshift_index(3)
            assert isinstance(wl_at_redshift_index_3, WeakLensingData)
            assert wl_at_redshift_index_3.redshifts == np.array([wl_data.redshifts[3]])

    def describe__data_are_consistent():

        @pytest.fixture
        def mismatched_data():
            num_radial_coordinates = 3
            num_redshifts = 2
            wrong_num_radial_coordinates = 4
            wrong_num_redshifts = 5

            radial_coordinates = np.logspace(-1, 1, num_radial_coordinates)
            redshifts = np.linspace(0, 1, num_redshifts)

            wl_wrong_radial_coordinates = np.ones((wrong_num_radial_coordinates, num_redshifts, 1))
            wl_wrong_redshifts = np.ones((num_radial_coordinates, wrong_num_redshifts, 1))
            wl_wrong_both = np.ones((wrong_num_radial_coordinates, wrong_num_redshifts, 1))

            return radial_coordinates, redshifts, wl_wrong_radial_coordinates, wl_wrong_redshifts, wl_wrong_both

        def it_makes_sure_the_data_shapes_are_consistent(mismatched_data):
            radial_coordinates, redshifts, wl_wrong_radial_coordinates, wl_wrong_redshifts, wl_wrong_both = mismatched_data
            num_radial_coordinates = radial_coordinates.size
            num_clusters = 1

            with pytest.raises(ValueError):
                WeakLensingData(
                    radial_coordinates=radial_coordinates,
                    redshifts=redshifts,
                    wl_signals=wl_wrong_radial_coordinates,
                    masses = np.ones((5, 3)),
                )

            with pytest.raises(ValueError):
                WeakLensingData(
                    radial_coordinates=radial_coordinates,
                    redshifts=redshifts,
                    wl_signals=wl_wrong_redshifts,
                    masses = np.ones((5, 3)),
                )

            with pytest.raises(ValueError):
                WeakLensingData(
                    radial_coordinates=radial_coordinates,
                    redshifts=redshifts,
                    wl_signals=wl_wrong_both,
                    masses = np.ones((5, 3)),
                )

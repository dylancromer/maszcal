import pytest
from maszcal.data.test import ActTestData


def describe_ActTestData():

    def it_automatically_loads_data():
        act_test = ActTestData(data_dir='data/test-act/')
        assert act_test.radial_coordinates.shape == (22,)

    def it_errors_if_the_files_are_missing():
        with pytest.raises(OSError):
            ActTestData(data_dir='data/test/i_am_not_a_real_directory_hopefully/')


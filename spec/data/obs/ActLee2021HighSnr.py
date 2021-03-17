import pytest
import numpy as np
from maszcal.data.obs import ActLee2021HighSnr


def describe_ActLee2021HighSnr():

    def it_automatically_loads_data():
        act_test = ActLee2021HighSnr(data_dir='data/act-eunseong/high-snr/')
        assert act_test.radial_coordinates.shape == (9,)
        from_arcmin = 2 * np.pi / 360 / 60
        assert np.all(act_test.radial_coordinates < 13*from_arcmin)

    def it_errors_if_the_files_are_missing():
        with pytest.raises(OSError):
            ActLee2021HighSnr(data_dir='data/test/i_am_not_a_real_directory_hopefully/')


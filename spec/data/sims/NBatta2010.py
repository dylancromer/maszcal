import pytest
from maszcal.data.sims import NBatta2010


def describe_nbatta2010():

    def it_automatically_loads_data():
        nbatta = NBatta2010(data_dir='data/NBatta2010/')
        assert nbatta.redshifts.size == 14

    def it_errors_if_the_files_are_missing():
        with pytest.raises(OSError):
            nbatta = NBatta2010(data_dir='data/test/i_am_not_a_real_directory_hopefully/')

import pytest
from maszcal.data import NBatta2010


def describe_nbatta2010():

    def it_automatically_loads_data():
        nbatta = NBatta2010(data_dir='data/test/NBatta2010')

        assert nbatta.redshifts.size == 10

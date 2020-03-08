import pytest
from maszcal.data.sims import NBatta2010


def describe_nbatta2010():

    def it_automatically_loads_data():
        nbatta = NBatta2010(data_dir='data/NBatta2010/')
        assert nbatta.redshifts.size == 14

    def it_errors_if_the_files_are_missing():
        with pytest.raises(OSError):
            nbatta = NBatta2010(data_dir='data/test/i_am_not_a_real_directory_hopefully/')

    def it_can_cut_the_radii():
        nbatta = NBatta2010(data_dir='data/NBatta2010/')
        assert nbatta.radii.size == 40

        nbatta_cut = nbatta.cut_radii(0.125, 3)
        assert nbatta_cut.radii.size == 24

    def describe_from_data():

        @pytest.fixture
        def nbatta():
            return NBatta2010(data_dir='data/NBatta2010/')

        def it_can_init_from_preloaded_data(nbatta):
            NBatta2010.from_data(
                radii=nbatta.radii,
                redshifts=nbatta.redshifts,
                wl_signals=nbatta.wl_signals,
                masses=nbatta.masses,
                cosmology=nbatta.cosmology,
            )

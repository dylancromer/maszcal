import numpy as np
import xarray as xa
from maszcal.model import StackedModel




def test_multiple_concentrations_del_sig_of_m():
    stacked_model = StackedModel()

    stacked_model.mus = xa.DataArray(np.linspace(12, 16, 20), dims=('mu'))

    rs = xa.DataArray(np.logspace(-1, 1, 40), dims=('radius'))
    cons = xa.DataArray(np.linspace(1,3, 40), dims=('concentration'))

    del_sigs = stacked_model.delta_sigma_of_mass(rs, stacked_model.mus, cons)

    assert del_sigs.shape == (40, 20, 40)


def test_multiple_concentrations_del_sig_stack():
    stacked_model = StackedModel()
    a_szs = xa.DataArray(np.linspace(-1, 1, 10), dims=('a_sz'))
    stacked_model.a_sz = a_szs

    stacked_model.mus = xa.DataArray(np.linspace(12, 16, 20), dims=('mu'))

    stacked_model.delta_sigma_of_mass = lambda rs, mus, cons, units: xa.DataArray(
        np.ones((cons.size, rs.size, mus.size)),
        dims=('concentration', 'radius', 'mu')
    )

    stacked_model.dnumber_dlogmass = lambda: xa.DataArray(
        np.ones((stacked_model.zs.size, stacked_model.mus.size)),
        dims=('redshift', 'mu')
    )

    rs = xa.DataArray(np.logspace(-1, 1, 20), dims=('radius'))
    cons = xa.DataArray(np.linspace(1,3, 11), dims=('concentration'))
    stacked_model.concentrations = cons

    stacked_del_sigs = stacked_model.delta_sigma(rs)

    assert stacked_del_sigs.shape == (10, 11, 20)

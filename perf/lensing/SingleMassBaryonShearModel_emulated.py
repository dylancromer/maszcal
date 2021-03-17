import pytest
import numpy as np
import astropy.units as u
from sklearn.gaussian_process.kernels import Matern
import supercubos
from maszcal.lensing import SingleMassShearModel
import maszcal.density
import maszcal.emulate
import maszcal.interpolate


def describe_single_mass_bin():

    @pytest.fixture
    def density_model():
        return maszcal.density.SingleMassGnfw(
            cosmo_params=maszcal.cosmology.CosmoParams(),
            mass_definition='mean',
            delta=200,
            comoving_radii=True,
            nfw_class=maszcal.density.SingleMassNfwModel,
        )

    @pytest.fixture
    def single_mass_model(density_model):
        zs = np.ones(1)
        return SingleMassShearModel(redshifts=zs, rho_func=density_model.rho_tot)

    @pytest.fixture
    def coordinate_samples():
        param_mins = np.array([np.log(1e12), 0.8, 0.05, 1])
        param_maxes = np.array([np.log(8e15), 6, 2, 6])
        return supercubos.LatinSampler().get_sym_sample(param_mins, param_maxes, 100)

    @pytest.fixture
    def wrapped_lensing_func(single_mass_model):
        def _wrapper(params):
            rs = np.logspace(-1, 1, 50)
            mus, cons, alphas, betas = params.T
            gammas = np.ones_like(alphas)*0.2
            return rs[:, None] * single_mass_model.excess_surface_density(rs, mus, cons, alphas, betas, gammas).squeeze()
        return _wrapper

    @pytest.fixture
    def lensing_samples(coordinate_samples, wrapped_lensing_func):
        return wrapped_lensing_func(coordinate_samples)

    @pytest.fixture
    def gp_emulator(coordinate_samples, lensing_samples):
        return maszcal.emulate.PcaEmulator.create_from_data(
            coords=coordinate_samples,
            data=lensing_samples,
            interpolator_class=maszcal.interpolate.GaussianProcessInterpolator,
            interpolator_kwargs={'kernel': Matern()}
        )

    @pytest.fixture
    def rbf_emulator(coordinate_samples, lensing_samples):
        return maszcal.emulate.PcaEmulator.create_from_data(
            coords=coordinate_samples,
            data=lensing_samples,
            interpolator_class=maszcal.interpolate.RbfInterpolator,
        )

    @pytest.fixture
    def param():
        return np.array([[np.log(1.23e14), 2.123, 0.8, 3.8]])

    def the_bare_function_is_fast(wrapped_lensing_func, param, benchmark):
        benchmark(wrapped_lensing_func, param)

    def the_gp_emulator_is_faster(gp_emulator, param, benchmark):
        benchmark(gp_emulator, param)

    def the_rbf_emulator_is_fastest(rbf_emulator, param, benchmark):
        benchmark(rbf_emulator, param)

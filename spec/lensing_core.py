import pytest
import numpy as np
import meso
import maszcal.mathutils as mathutils
import maszcal.lensing._core as core


def rho_with_analytic_result(radii, param_1, param_2):
    total_ndim = radii.ndim+2
    radii = mathutils.atleast_kd(radii, total_ndim)
    param_1 = mathutils.atleast_kd(param_1[:, None], total_ndim, append_dims=False)
    param_2 = mathutils.atleast_kd(param_2, total_ndim, append_dims=False)
    return (radii**2) * param_1 * param_2


def prob_dist_func(r, scale):
    scale = mathutils.atleast_kd(scale, r.ndim+1, append_dims=False)
    return scale*np.ones(r.shape + (1,))


def describe_Miscentering():

    @pytest.fixture
    def miscentering():
        return core.Miscentering(
            rho_func=rho_with_analytic_result,
            misc_distrib=prob_dist_func,
            miscentering_func=meso.Rho().miscenter,
        )

    def it_takes_a_density_function_and_creates_a_miscentered_version(miscentering):
        rs = np.logspace(-2, 2, 30)
        param_1 = np.ones(3)
        param_2 = np.ones(2)
        rho_params = (param_1, param_2)

        miscenter_model = meso.Rho()
        ul = miscenter_model.max_radius
        ll = miscenter_model.min_radius
        interval = ul - ll

        misc_params = (np.array([1/interval, 1/interval]), np.array([0.5, 0.5]))
        rho_cents = rho_with_analytic_result(rs, param_1, param_2)
        rho_miscs = miscentering.rho(rs, misc_params, rho_params).squeeze()

        analytical_answer = 1/3 * (ll**2 + ul**2 + ul*ll + 3*rs**2)
        analytical_answer = analytical_answer[:, None, None] * param_1[None, :, None] * param_2[None, None, :]
        analytical_answer = rho_cents/2 + analytical_answer/2

        assert np.allclose(rho_miscs, analytical_answer, rtol=1e-3)

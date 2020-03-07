import datetime
import pytest
import numpy as np
import pathos.pools as pp
from scipy.interpolate import interp1d
import maszcal.data.sims
import maszcal.lensing
import maszcal.likelihoods
import maszcal.fitutils


NUM_THREADS = 12
NFW_PARAM_MINS = np.array([np.log(1e12), 0])
NFW_PARAM_MAXES = np.array([np.log(5e15), 6])
BARYON_PARAM_MINS = np.array([np.log(1e12), 0, 0.1, 2])
BARYON_PARAM_MAXES = np.array([np.log(5e15), 6, 1.9, 6])
LOWER_RADIUS_CUT = 0.125
UPPER_RADIUS_CUT = 3


def _load_act_covariance():
    ACT_DIR = 'data/act-hsc/'
    act_rs, _, act_errs = np.loadtxt(ACT_DIR + 'deltaSigma.dat', unpack=True, skiprows=1, usecols=[0,3,4])
    full_err_cov = np.loadtxt(ACT_DIR + 'cov_lss_stacked.dat')
    con_err_cov = np.loadtxt(ACT_DIR + 'con_cov.txt')
    cov_full = (act_errs * np.identity(len(act_errs)) * act_errs) + full_err_cov + con_err_cov
    return act_rs, cov_full


def _get_cov_from_act(radii):
    act_rs, act_cov = _load_act_covariance()

    act_diag = np.diag(act_cov)

    _diag_interpolator = interp1d(act_rs, np.log(act_diag), kind='linear')
    diag_interpolator = lambda r: np.exp(_diag_interpolator(r))

    return np.diagflat(diag_interpolator(radii))


def _log_like(params, radii, esd_model_func, esd_data):
    esd_model = esd_model_func(radii, params).flatten()
    esd_data = esd_data.flatten()

    ACT_COVARIANCE = _get_cov_from_act(radii) / 100

    return maszcal.likelihoods.log_gaussian_shape(esd_model, esd_data, ACT_COVARIANCE)


def _get_best_fit(esd_data, radii, esd_model_func, param_mins, param_maxes):
    def func_to_minimize(params): return -_log_like(params, radii, esd_model_func, esd_data)
    return maszcal.fitutils.global_minimize(func_to_minimize, param_mins, param_maxes, 'global-differential-evolution')


def _calculate_nfw_fits(i, z, sim_data):
    nfw_model = maszcal.lensing.SingleMassNfwLensingSignal(
        redshift=np.array([z]),
        delta=500,
        mass_definition='crit',
    )

    def esd_model_func(radii, params): return  nfw_model.esd(radii, params[None, :])

    def _pool_func(data): return _get_best_fit(data, sim_data.radii, esd_model_func, NFW_PARAM_MINS, NFW_PARAM_MAXES)

    pool = pp.ProcessPool(NUM_THREADS)
    best_fit_chunk = np.array(
        pool.map(_pool_func, sim_data.wl_signals[:, i, :].T),
    ).T
    pool.close()
    pool.join()
    pool.terminate()
    pool.restart()

    return best_fit_chunk


def _calculate_baryon_fits(i, z, sim_data):
    baryon_model = maszcal.lensing.SingleBaryonLensingSignal(
        redshift=np.array([z]),
        delta=500,
        mass_definition='crit',
    )

    def esd_model_func(radii, params):
        gamma = np.array([0.2])
        params = np.concatenate((params, gamma))
        return baryon_model.esd(radii, params[None, :])

    def _pool_func(data): return _get_best_fit(data, sim_data.radii, esd_model_func, BARYON_PARAM_MINS, BARYON_PARAM_MAXES)

    pool = pp.ProcessPool(NUM_THREADS//2)
    best_fit_chunk = np.array(
        pool.map(_pool_func, sim_data.wl_signals[:, i, :].T),
    ).T
    pool.close()
    pool.join()
    pool.terminate()
    pool.restart()

    return best_fit_chunk


def _generate_header():
    terminator = '\n'
    configs = [
        f'NFW_PARAM_MINS = {NFW_PARAM_MINS}',
        f'NFW_PARAM_MAXES = {NFW_PARAM_MAXES}',
        f'BARYON_PARAM_MINS = {BARYON_PARAM_MINS}',
        f'BARYON_PARAM_MAXES = {BARYON_PARAM_MAXES}',
        f'LOWER_RADIUS_CUT = {LOWER_RADIUS_CUT}',
        f'UPPER_RADIUS_CUT = {UPPER_RADIUS_CUT}',
    ]
    header = [conf + terminator for conf in configs]
    return ''.join(header)


def _save(array, name):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    filename = name + '_' + timestamp

    DIR = 'data/NBatta2010/single-mass-bin-fits/'

    header = _generate_header()

    nothing = np.array([])

    np.savetxt(DIR + filename + '.header.txt', nothing, header=header)
    np.save(DIR + filename + '.npy', array)


def describe_nbatta_sim():

    def describe_impact_of_baryons():

        @pytest.fixture
        def sim_data():
            return maszcal.data.sims.NBatta2010().cut_radii(LOWER_RADIUS_CUT, UPPER_RADIUS_CUT)

        def using_baryon_reduces_bias(sim_data):
            num_clusters = sim_data.wl_signals.shape[-1]

            nfw_params_shape = (2, num_clusters, sim_data.redshifts.size)
            nfw_fits = np.zeros(nfw_params_shape)

            for i, z in enumerate(sim_data.redshifts):
                nfw_fits[:, :, i] = _calculate_nfw_fits(i, z, sim_data)

            _save(nfw_fits, 'nfw-free-c')

            baryon_params_shape = (4, num_clusters, sim_data.redshifts.size)
            baryon_fits = np.zeros(baryon_params_shape)

            for i, z in enumerate(sim_data.redshifts):
                baryon_fits[:, :, i] = _calculate_baryon_fits(i, z, sim_data)

            _save(baryon_fits, 'bary-free-c')

import datetime
import pytest
import numpy as np
import pathos.pools as pp
import maszcal.data.sims
import maszcal.data.obs
import maszcal.lensing
import maszcal.likelihoods
import maszcal.fitutils
import maszcal.concentration


NUM_THREADS = 8

NFW_PARAM_MINS = np.array([np.log(6e12), 0])
NFW_PARAM_MAXES = np.array([np.log(5e15), 6])

CM_PARAM_MINS = np.array([np.log(6e12)])
CM_PARAM_MAXES = np.array([np.log(5e15)])

BARYON_PARAM_MINS = np.array([np.log(6e12), 0, 2])
BARYON_PARAM_MAXES = np.array([np.log(5e15), 6, 7])

BARYON_CM_PARAM_MINS = np.array([np.log(6e12), 2])
BARYON_CM_PARAM_MAXES = np.array([np.log(5e15), 7])

LOWER_RADIUS_CUT = 0.125
UPPER_RADIUS_CUT = 3

COVARIANCE_REDUCTION_FACTOR = 1/400


def _log_like(params, radii, esd_model_func, esd_data, fisher_matrix):
    if np.any(params) < 0:
        return -np.inf()

    esd_model = esd_model_func(radii, params).flatten()
    esd_data = esd_data.flatten()

    return maszcal.likelihoods.log_gaussian_shape(radii*esd_model, radii*esd_data, fisher_matrix)


def _get_best_fit(esd_data, radii, esd_model_func, fisher_matrix, param_mins, param_maxes):
    def func_to_minimize(params): return -_log_like(params, radii, esd_model_func, esd_data, fisher_matrix)
    return maszcal.fitutils.global_minimize(func_to_minimize, param_mins, param_maxes, 'global-differential-evolution')


def _pool_map(func, array):
    pool = pp.ProcessPool(NUM_THREADS)
    mapped_array = np.array(
        pool.map(func, array),
    ).T
    pool.close()
    pool.join()
    pool.terminate()
    pool.restart()
    return mapped_array


def _calculate_nfw_fits(i, z, sim_data, fisher_matrix):
    nfw_model = maszcal.lensing.SingleMassNfwLensingSignal(
        redshift=np.array([z]),
        delta=500,
        mass_definition='crit',
        cosmo_params=sim_data.cosmology,
    )

    def esd_model_func(radii, params): return  nfw_model.esd(radii, params[None, :])

    def _pool_func(data): return _get_best_fit(data, sim_data.radii, esd_model_func, fisher_matrix, NFW_PARAM_MINS, NFW_PARAM_MAXES)

    return _pool_map(_pool_func, sim_data.wl_signals[:, i, :].T)


def _calculate_cm_fits(i, z, sim_data, fisher_matrix):
    nfw_model = maszcal.lensing.SingleMassNfwLensingSignal(
        redshift=np.array([z]),
        delta=500,
        mass_definition='crit',
        cosmo_params=sim_data.cosmology,
    )

    def esd_model_func(radii, mus):
        masses = np.exp(mus)
        con_model = maszcal.concentration.ConModel('500c', cosmology=sim_data.cosmology)
        cons = con_model.c(masses, np.array([z]), '500c').flatten()
        params = np.stack((mus, cons)).T
        return nfw_model.esd(radii, params)

    def _pool_func(data): return _get_best_fit(data, sim_data.radii, esd_model_func, fisher_matrix, CM_PARAM_MINS, CM_PARAM_MAXES)

    return _pool_map(_pool_func, sim_data.wl_signals[:, i, :].T)


def _calculate_baryon_fits(i, z, sim_data, fisher_matrix):
    baryon_model = maszcal.lensing.SingleBaryonLensingSignal(
        redshift=np.array([z]),
        delta=500,
        mass_definition='crit',
        cosmo_params=sim_data.cosmology,
    )

    def esd_model_func(radii, params):
        alpha = np.array([0.9])
        gamma = np.array([0.2])
        params = np.concatenate((params[:2], alpha, params[2:], gamma))
        return baryon_model.esd(radii, params[None, :])

    def _pool_func(data): return _get_best_fit(data, sim_data.radii, esd_model_func, fisher_matrix, BARYON_PARAM_MINS, BARYON_PARAM_MAXES)

    return _pool_map(_pool_func, sim_data.wl_signals[:, i, :].T)


def _calculate_baryon_cm_fits(i, z, sim_data, fisher_matrix):
    baryon_model = maszcal.lensing.SingleBaryonLensingSignal(
        redshift=np.array([z]),
        delta=500,
        mass_definition='crit',
        cosmo_params=sim_data.cosmology,
    )

    def esd_model_func(radii, params):
        mu = params[0:1]
        alpha = np.array([0.9])
        beta = params[1:2]
        gamma = np.array([0.2])

        mass = np.exp(mu)
        con_model = maszcal.concentration.ConModel('500c', cosmology=sim_data.cosmology)
        con = con_model.c(mass, np.array([z]), '500c').flatten()

        params = np.concatenate((mu, con, alpha, beta, gamma))
        params = params[None, :]
        return baryon_model.esd(radii, params)

    def _pool_func(data): return _get_best_fit(data, sim_data.radii, esd_model_func, fisher_matrix, BARYON_CM_PARAM_MINS, BARYON_CM_PARAM_MAXES)

    return _pool_map(_pool_func, sim_data.wl_signals[:, i, :].T)


def _generate_header():
    terminator = '\n'
    configs = [
        f'NFW_PARAM_MINS = {NFW_PARAM_MINS}',
        f'NFW_PARAM_MAXES = {NFW_PARAM_MAXES}',
        f'CM_PARAM_MINS = {CM_PARAM_MINS}',
        f'CM_PARAM_MAXES = {CM_PARAM_MAXES}',
        f'BARYON_PARAM_MINS = {BARYON_PARAM_MINS}',
        f'BARYON_PARAM_MAXES = {BARYON_PARAM_MAXES}',
        f'BARYON_CM_PARAM_MINS = {BARYON_CM_PARAM_MINS}',
        f'BARYON_CM_PARAM_MAXES = {BARYON_CM_PARAM_MAXES}',
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

            act_covariance = maszcal.data.obs.ActHsc2018.covariance('data/act-hsc/', sim_data.radii) * COVARIANCE_REDUCTION_FACTOR
            act_r_esd_cov = np.diagflat(sim_data.radii).T @ act_covariance @ np.diagflat(sim_data.radii)
            act_fisher = np.linalg.inv(act_r_esd_cov)

            nfw_params_shape = (NFW_PARAM_MINS.size, num_clusters, sim_data.redshifts.size)
            nfw_fits = np.zeros(nfw_params_shape)
            for i, z in enumerate(sim_data.redshifts):
                nfw_fits[:, :, i] = _calculate_nfw_fits(i, z, sim_data, act_fisher)

            _save(nfw_fits, 'nfw-free-c')

            cm_params_shape = (CM_PARAM_MINS.size, num_clusters, sim_data.redshifts.size)
            cm_fits = np.zeros(cm_params_shape)
            for i, z in enumerate(sim_data.redshifts):
                cm_fits[:, :, i] = _calculate_cm_fits(i, z, sim_data, act_fisher)

            _save(cm_fits, 'nfw-cm')

            baryon_params_shape = (BARYON_PARAM_MINS.size, num_clusters, sim_data.redshifts.size)
            baryon_fits = np.zeros(baryon_params_shape)
            for i, z in enumerate(sim_data.redshifts):
                baryon_fits[:, :, i] = _calculate_baryon_fits(i, z, sim_data, act_fisher)

            _save(baryon_fits, 'bary-free-c')

            baryon_cm_params_shape = (BARYON_CM_PARAM_MINS.size, num_clusters, sim_data.redshifts.size)
            baryon_cm_fits = np.zeros(baryon_cm_params_shape)
            for i, z in enumerate(sim_data.redshifts):
                baryon_cm_fits[:, :, i] = _calculate_baryon_cm_fits(i, z, sim_data, act_fisher)

            _save(baryon_cm_fits, 'bary-cm')

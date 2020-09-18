import datetime
import time
import numpy as np
import pathos.pools as pp
import astropy.units as u
import maszcal.data.sims
import maszcal.data.obs
import maszcal.lensing
import maszcal.likelihoods
import maszcal.fitutils
import maszcal.concentration


np.seterr(all='ignore')


NUM_THREADS = 12

CM_PARAM_MINS = np.array([np.log(1e12)])
CM_PARAM_MAXES = np.array([np.log(8e15)])

LOWER_RADIUS_CUT = 0.125
UPPER_RADIUS_CUT = 3

COVARIANCE_REDUCTION_FACTOR = 1/400

FIXED_GAMMA = np.array([0.2])


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def _log_like(params, radii, esd_model_func, esd_data, fisher_matrix):
    if np.any(params) < 0:
        return -np.inf

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


def calculate_cm_fits(i, z, sim_data, fisher_matrix):
    nfw_model = maszcal.lensing.SingleMassNfwShearModel(
        redshifts=np.array([z]),
        delta=500,
        mass_definition='crit',
        cosmo_params=sim_data.cosmology,
    )

    def esd_model_func(radii, mus):
        masses = np.exp(mus)
        con_model = maszcal.concentration.ConModel('500c', cosmology=sim_data.cosmology)
        cons = con_model.c(masses, np.array([z]), '500c').flatten()
        return nfw_model.excess_surface_density(radii, mus, cons)

    def _pool_func(data): return _get_best_fit(data, sim_data.radii, esd_model_func, fisher_matrix, CM_PARAM_MINS, CM_PARAM_MAXES)

    return _pool_map(_pool_func, sim_data.wl_signals[:, i, :].T)


def _generate_header():
    terminator = '\n'
    configs = [
        f'CM_PARAM_MINS = {CM_PARAM_MINS}',
        f'CM_PARAM_MAXES = {CM_PARAM_MAXES}',
        f'LOWER_RADIUS_CUT = {LOWER_RADIUS_CUT}',
        f'UPPER_RADIUS_CUT = {UPPER_RADIUS_CUT}',
    ]
    header = [conf + terminator for conf in configs]
    return ''.join(header)


def save(array, name):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    filename = name + '_' + timestamp

    DIR = 'data/NBatta2010/single-mass-bin-fits/'

    header = _generate_header()

    nothing = np.array([])

    np.savetxt(DIR + filename + '.header.txt', nothing, header=header)
    np.save(DIR + filename + '.npy', array)


def get_sim_data():
    return maszcal.data.sims.NBatta2010().cut_radii(LOWER_RADIUS_CUT, UPPER_RADIUS_CUT)


if __name__ == '__main__':
    start_time = time.time()
    print('Starting single-mass bin fits')


    sim_data = get_sim_data()
    num_clusters = sim_data.wl_signals.shape[-1]

    act_covariance = maszcal.data.obs.ActHsc2018.covariance('data/act-hsc/', sim_data.radii) * COVARIANCE_REDUCTION_FACTOR
    act_r_esd_cov = np.diagflat(sim_data.radii).T @ act_covariance @ np.diagflat(sim_data.radii)
    act_fisher = np.linalg.inv(act_r_esd_cov)


    print('Optimizing with NFW c(m)...')

    start_time = time.time()

    cm_params_shape = (CM_PARAM_MINS.size, num_clusters, sim_data.redshifts.size)
    cm_fits = np.zeros(cm_params_shape)
    for i, z in enumerate(sim_data.redshifts):
        cm_fits[:, :, i] = calculate_cm_fits(i, z, sim_data, act_fisher)

    save(cm_fits, 'nfw-cm')

    delta_t = round(time.time() - start_time)
    print(f'Finished in {delta_t} seconds.')

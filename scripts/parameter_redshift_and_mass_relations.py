import datetime
import time
import numpy as np
import pathos.pools as pp
import maszcal.data.sims
import maszcal.data.obs
import maszcal.lensing
import maszcal.likelihoods
import maszcal.fitutils
import maszcal.concentration


NUM_THREADS = 8

NFW_PARAM_MINS = np.array([0])
NFW_PARAM_MAXES = np.array([10])

BARYON_PARAM_MINS = np.array([0, 0, 0])
BARYON_PARAM_MAXES = np.array([10, 10, 10])

BARYON_CM_PARAM_MINS = np.array([0, 0])
BARYON_CM_PARAM_MAXES = np.array([10, 10])

LOWER_RADIUS_CUT = 0.125
UPPER_RADIUS_CUT = 3

COVARIANCE_REDUCTION_FACTOR = 1/400


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def _log_like(params, radii, esd_model_func, esd_data, true_mass, fisher_matrix):
    if np.any(params) < 0:
        return -np.inf()

    esd_model = esd_model_func(radii, true_mass, params).flatten()
    esd_data = esd_data.flatten()

    return maszcal.likelihoods.log_gaussian_shape(radii*esd_model, radii*esd_data, fisher_matrix)


def _get_best_fit(esd_data, true_mass, radii, esd_model_func, fisher_matrix, param_mins, param_maxes):
    def func_to_minimize(params): return -_log_like(params, radii, esd_model_func, esd_data, true_mass, fisher_matrix)
    return maszcal.fitutils.global_minimize(func_to_minimize, param_mins, param_maxes, 'global-differential-evolution')


def _pool_map(func, *arrays):
    pool = pp.ProcessPool(NUM_THREADS)
    mapped_array = np.array(
        pool.map(func, *arrays),
    ).T
    pool.close()
    pool.join()
    pool.terminate()
    pool.restart()
    return mapped_array


def calculate_nfw_fits(i, z, sim_data, fisher_matrix):
    nfw_model = maszcal.lensing.SingleMassNfwShearModel(
        redshifts=np.array([z]),
        delta=500,
        mass_definition='crit',
        cosmo_params=sim_data.cosmology,
    )

    def esd_model_func(radii, true_mass, params):
        con = params
        mass = np.array([true_mass])
        mu = np.log(mass)
        return  nfw_model.excess_surface_density(radii, mu, con)

    def _pool_func(data, true_mass):
        return _get_best_fit(data, true_mass, sim_data.radii, esd_model_func, fisher_matrix, NFW_PARAM_MINS, NFW_PARAM_MAXES)

    return _pool_map(_pool_func, sim_data.wl_signals[:, i, :].T, sim_data.masses[i, :])


def calculate_baryon_fits(i, z, sim_data, fisher_matrix):
    baryon_model = maszcal.lensing.SingleMassBaryonShearModel(
        redshifts=np.array([z]),
        delta=500,
        mass_definition='crit',
        cosmo_params=sim_data.cosmology,
    )

    def esd_model_func(radii, true_mass, params):
        con = params[0:1]
        alpha = params[1:2]
        beta = params[2:3]
        gamma = np.array([0.2])
        mass = np.array([true_mass])
        mu = np.log(mass)
        return baryon_model.excess_surface_density(radii, mu, con, alpha, beta, gamma)

    def _pool_func(data, true_mass):
        return _get_best_fit(data, true_mass, sim_data.radii, esd_model_func, fisher_matrix, BARYON_PARAM_MINS, BARYON_PARAM_MAXES)

    return _pool_map(_pool_func, sim_data.wl_signals[:, i, :].T, sim_data.masses[i, :])


def calculate_baryon_cm_fits(i, z, sim_data, fisher_matrix):
    baryon_model = maszcal.lensing.SingleMassBaryonShearModel(
        redshifts=np.array([z]),
        delta=500,
        mass_definition='crit',
        cosmo_params=sim_data.cosmology,
    )

    def esd_model_func(radii, true_mass, params):
        alpha = params[0:1]
        beta = params[1:2]
        gamma = np.array([0.2])

        mass = np.array([true_mass])
        mu = np.log(mass)
        con_model = maszcal.concentration.ConModel('500c', cosmology=sim_data.cosmology)
        con = con_model.c(mass, np.array([z]), '500c').flatten()

        return baryon_model.excess_surface_density(radii, mu, con, alpha, beta, gamma)

    def _pool_func(data, true_mass):
        return _get_best_fit(data, true_mass, sim_data.radii, esd_model_func, fisher_matrix, BARYON_CM_PARAM_MINS, BARYON_CM_PARAM_MAXES)

    return _pool_map(_pool_func, sim_data.wl_signals[:, i, :].T, sim_data.masses[i, :])


def _generate_header():
    terminator = '\n'
    configs = [
        f'NFW_PARAM_MINS = {NFW_PARAM_MINS}',
        f'NFW_PARAM_MAXES = {NFW_PARAM_MAXES}',
        f'BARYON_PARAM_MINS = {BARYON_PARAM_MINS}',
        f'BARYON_PARAM_MAXES = {BARYON_PARAM_MAXES}',
        f'BARYON_CM_PARAM_MINS = {BARYON_CM_PARAM_MINS}',
        f'BARYON_CM_PARAM_MAXES = {BARYON_CM_PARAM_MAXES}',
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


def run_validation(sim_data=get_sim_data()):
    num_clusters = sim_data.wl_signals.shape[-1]

    act_covariance = maszcal.data.obs.ActHsc2018.covariance('data/act-hsc/', sim_data.radii) * COVARIANCE_REDUCTION_FACTOR
    act_r_esd_cov = np.diagflat(sim_data.radii).T @ act_covariance @ np.diagflat(sim_data.radii)
    act_fisher = np.linalg.inv(act_r_esd_cov)

    print('Optimizing with NFWs...')

    start_time = time.time()

    nfw_params_shape = (NFW_PARAM_MINS.size, num_clusters, sim_data.redshifts.size)
    nfw_fits = np.zeros(nfw_params_shape)
    for i, z in enumerate(sim_data.redshifts):
        nfw_fits[:, :, i] = calculate_nfw_fits(i, z, sim_data, act_fisher)

    save(nfw_fits, 'nfw-free-c_fixed-true-mass')

    delta_t = round(time.time() - start_time)
    print(f'Finished in {delta_t} seconds.')
    print('Optimizing with baryons...')

    start_time = time.time()

    baryon_params_shape = (BARYON_PARAM_MINS.size, num_clusters, sim_data.redshifts.size)
    baryon_fits = np.zeros(baryon_params_shape)
    for i, z in enumerate(sim_data.redshifts):
        baryon_fits[:, :, i] = calculate_baryon_fits(i, z, sim_data, act_fisher)

    save(baryon_fits, 'bary-free-c_fixed-true-mass')

    delta_t = round(time.time() - start_time)
    print(f'Finished in {delta_t} seconds.')
    print('Optimizing with baryons c(m)...')

    start_time = time.time()

    baryon_cm_params_shape = (BARYON_CM_PARAM_MINS.size, num_clusters, sim_data.redshifts.size)
    baryon_cm_fits = np.zeros(baryon_cm_params_shape)
    for i, z in enumerate(sim_data.redshifts):
        baryon_cm_fits[:, :, i] = calculate_baryon_cm_fits(i, z, sim_data, act_fisher)

    save(baryon_cm_fits, 'bary-cm_fixed-true-mass')

    delta_t = round(time.time() - start_time)
    print(f'Finished in {delta_t} seconds.')


def main():
    start_time = time.time()
    print('Beginning validation:')
    run_validation()
    delta_t = round(time.time() - start_time)
    print(bcolors.OKGREEN + '--- Validation complete ---' + bcolors.ENDC)
    print(f'Total elapsed time: {delta_t} seconds')


if __name__ == '__main__':
    main()

import datetime
import argparse
import time
import numpy as np
import pathos.pools as pp
import supercubos
from sklearn.gaussian_process.kernels import Matern
import emcee
import astropy.units as u
import maszcal.data.sims
import maszcal.data.obs
import maszcal.lensing
import maszcal.likelihoods
import maszcal.fitutils
import maszcal.concentration
import maszcal.emulate
import maszcal.interpolate


np.seterr(all='ignore')


NUM_PROCESSES = 12
DIR = 'data/NBatta2010/single-mass-bin-fits/'
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")

NFW_PARAM_MINS = np.array([np.log(1e12), 1])
NFW_PARAM_MAXES = np.array([np.log(8e15), 6])
CM_PARAM_MINS = np.array([np.log(1e12)])
CM_PARAM_MAXES = np.array([np.log(8e15)])
BARYON_PARAM_MINS = np.array([np.log(1e12), 1, 0.1, 2])
BARYON_PARAM_MAXES = np.array([np.log(8e15), 6, 2, 7])
BARYON_CM_PARAM_MINS = np.array([np.log(1e12), 0.1, 2])
BARYON_CM_PARAM_MAXES = np.array([np.log(8e15), 2, 7])

LOWER_RADIUS_CUT = 0.18
UPPER_RADIUS_CUT = 5

COVARIANCE_REDUCTION_FACTOR = 1/400

FIXED_GAMMA = np.array([0.2])

SAMPLE_SEED = 13
NUM_EMULATION_SAMPLES = 10
NUM_ERRORCHECK_SAMPLES = 10

NSTEPS = 3
NWALKERS = 10
WALKER_DISPERSION = 1e-3


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
    pool = pp.ProcessPool(NUM_PROCESSES)
    mapped_array = np.array(
        pool.map(func, array),
    ).T
    pool.close()
    pool.join()
    pool.clear()
    pool.terminate()
    pool.restart()
    return mapped_array


def get_coordinate_samples():
    return supercubos.LatinSampler(rng=np.random.default_rng(seed=SAMPLE_SEED)).get_sym_sample(
        PARAM_MINS,
        PARAM_MAXES,
        NUM_EMULATION_SAMPLES,
    )


def get_errorcheck_coords():
    return supercubos.LatinSampler(rng=np.random.default_rng(seed=SAMPLE_SEED)).get_sym_sample(
        PARAM_MINS,
        PARAM_MAXES,
        NUM_ERRORCHECK_SAMPLES,
    )


def get_nfw_model(z):
    return maszcal.lensing.SingleMassNfwShearModel(
        redshifts=np.array([z]),
        delta=500,
        mass_definition='crit',
        cosmo_params=sim_data.cosmology,
    )


def get_wrapped_nfw_func(z):
    single_mass_model = get_nfw_model(z)

    def _wrapper(params):
        mus, cons = params.T
        return sim_data.radii[:, None] * single_mass_model.excess_surface_density(sim_data.radii, mus, cons).squeeze()
    return _wrapper


def get_nfw_cm_model(z):
    return maszcal.lensing.SingleMassNfwShearModel(
        redshifts=np.array([z]),
        delta=500,
        mass_definition='crit',
        cosmo_params=sim_data.cosmology,
    )


def get_wrapped_nfw_cm_func(z):
    single_mass_model = get_nfw_cm_model(z)
    con_model = maszcal.concentration.ConModel('500c', cosmology=sim_data.cosmology)
    def _wrapper(mus):
        mus = mus.flatten()
        masses = np.exp(mus)
        cons = con_model.c(masses, np.array([z]), '500c').flatten()
        return sim_data.radii[:, None] * single_mass_model.excess_surface_density(sim_data.radii, mus, cons).squeeze()
    return _wrapper


def get_bary_model(z):
    density_model = maszcal.density.SingleMassGnfw(
        cosmo_params=sim_data.cosmology,
        mass_definition='crit',
        delta=500,
        units=u.Msun/u.pc**2,
        comoving_radii=True,
        nfw_class=maszcal.density.SingleMassNfwModel,
    )
    return maszcal.lensing.SingleMassShearModel(
        redshifts=np.array([z]),
        rho_func=density_model.rho_tot,
    )


def get_wrapped_bary_func(z):
    single_mass_model = get_bary_model(z)
    def _wrapper(params):
        mus, cons, alphas, betas = params.T
        gammas = np.ones_like(alphas) * FIXED_GAMMA
        return sim_data.radii[:, None] * single_mass_model.excess_surface_density(
            sim_data.radii,
            mus,
            cons,
            alphas,
            betas,
            gammas,
        ).squeeze()
    return _wrapper


def get_bary_cm_model(z):
    density_model = maszcal.density.SingleMassGnfw(
        cosmo_params=sim_data.cosmology,
        mass_definition='crit',
        delta=500,
        units=u.Msun/u.pc**2,
        comoving_radii=True,
        nfw_class=maszcal.density.SingleMassNfwModel,
    )
    return maszcal.lensing.SingleMassShearModel(
        redshifts=np.array([z]),
        rho_func=density_model.rho_tot,
    )


def get_wrapped_bary_cm_func(z):
    single_mass_model = get_bary_cm_model(z)
    con_model = maszcal.concentration.ConModel('500c', cosmology=sim_data.cosmology)

    def _wrapper(params):
        mus, alphas, betas = params.T
        masses = np.exp(mus)
        cons = con_model.c(masses, np.array([z]), '500c').flatten()
        gammas = np.ones_like(alphas) * FIXED_GAMMA
        return sim_data.radii[:, None] * single_mass_model.excess_surface_density(
            sim_data.radii,
            mus,
            cons,
            alphas,
            betas,
            gammas,
        ).squeeze()
    return _wrapper


def get_emulator(z, wrapped_lensing_func):
    coordinate_samples = get_coordinate_samples()
    lensing_samples = wrapped_lensing_func(coordinate_samples)
    return maszcal.emulate.PcaEmulator.create_from_data(
        coords=coordinate_samples,
        data=lensing_samples,
        interpolator_class=maszcal.interpolate.RbfInterpolator,
    )


def estimate_emulator_errors(emulator, wrapped_lensing_func):
    errorcheck_coords = get_errorcheck_coords()
    emulated_signal = emulator(errorcheck_coords)
    real_signal = wrapped_lensing_func(errorcheck_coords)
    return (emulated_signal - real_signal)/real_signal


def calculate_best_fits(i, z, sim_data, fisher_matrix, emulator):
    def esd_model_func(radii, params): return emulator(params[None, :])
    def _pool_func(data): return _get_best_fit(data, sim_data.radii, esd_model_func, fisher_matrix, PARAM_MINS, PARAM_MAXES)
    return _pool_map(_pool_func, sim_data.wl_signals[:, i, :].T)


def _generate_header():
    terminator = '\n'
    configs = [
        f'PARAM_MINS = {PARAM_MINS}',
        f'PARAM_MAXES = {PARAM_MAXES}',
        f'LOWER_RADIUS_CUT = {LOWER_RADIUS_CUT}',
        f'UPPER_RADIUS_CUT = {UPPER_RADIUS_CUT}',
        f'SAMPLE_SEED = {SAMPLE_SEED}'
    ]
    header = [conf + terminator for conf in configs]
    return ''.join(header)


def save(array, name):
    filename = name + '_' + TIMESTAMP
    header = _generate_header()
    nothing = np.array([])
    np.savetxt(DIR + filename + '.header.txt', nothing, header=header)
    np.save(DIR + filename + '.npy', array)


def get_sim_data():
    return maszcal.data.sims.NBatta2010().cut_radii(LOWER_RADIUS_CUT, UPPER_RADIUS_CUT)


def log_prob(params, radii, esd_model_func, esd_data, fisher_matrix):
    if np.all(PARAM_MINS < params) and np.all(params < PARAM_MAXES):
        return _log_like(params, radii, esd_model_func, esd_data, fisher_matrix)
    else:
        return - np.inf


def generate_chain_filename(i, j, slug):
    return DIR + slug + f'-mcmc-chains_zbin-{i}_mbin-{j}_' + '.h5'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_slug', help='Either \"nfw-free-c\", \"nfw-cm\", \"bary-free-c\", or \"bary-cm\"')
    args = parser.parse_args()

    get_wrapped_lensing_func = {
        'nfw-free-c': get_wrapped_nfw_func,
        'nfw-cm': get_wrapped_nfw_cm_func,
        'bary-free-c': get_wrapped_bary_func,
        'bary-cm': get_wrapped_bary_cm_func,
    }[args.model_slug]

    PARAM_MINS = {
        'nfw-free-c': NFW_PARAM_MINS,
        'nfw-cm': CM_PARAM_MINS,
        'bary-free-c': BARYON_PARAM_MINS,
        'bary-cm': BARYON_CM_PARAM_MINS,
    }[args.model_slug]

    PARAM_MAXES = {
        'nfw-free-c': NFW_PARAM_MAXES,
        'nfw-cm': CM_PARAM_MAXES,
        'bary-free-c': BARYON_PARAM_MAXES,
        'bary-cm': BARYON_CM_PARAM_MAXES,
    }[args.model_slug]

    start_time = time.time()
    print('Starting single-mass bin fits')

    sim_data = get_sim_data()
    num_clusters = sim_data.wl_signals.shape[-1]
    act_covariance = maszcal.data.obs.ActHsc2018.covariance('data/act-hsc/', sim_data.radii) * COVARIANCE_REDUCTION_FACTOR
    act_r_esd_cov = np.diagflat(sim_data.radii).T @ act_covariance @ np.diagflat(sim_data.radii)
    act_fisher = np.linalg.inv(act_r_esd_cov)

    print('Optimizing...')

    start_time = time.time()

    params_shape = (PARAM_MINS.size, num_clusters, sim_data.redshifts.size)
    best_fits = np.zeros(params_shape)
    ndim = PARAM_MINS.size
    for i, z in enumerate(sim_data.redshifts):
        wrapped_lensing_func = get_wrapped_lensing_func(z)

        emulator = get_emulator(z, wrapped_lensing_func)
        emulator_errors = estimate_emulator_errors(emulator, wrapped_lensing_func)

        save(emulator_errors, f'{args.model_slug}-emulator-errors_redshift-{z}_bin-{i}')

        best_fits[:, :, i] = calculate_best_fits(i, z, sim_data, act_fisher, emulator)

        def esd_model_func(radii, params): return emulator(params[None, :])

        for j, fit in enumerate(best_fits[:, :, i].T):
            initial_position = fit + WALKER_DISPERSION*np.random.randn(NWALKERS, ndim)
            log_prob_args = (sim_data.radii, esd_model_func, sim_data.wl_signals[:, i, j], act_fisher)

            chain_filename = generate_chain_filename(i, j, args.model_slug)
            backend = emcee.backends.HDFBackend(chain_filename)
            backend.reset(NWALKERS, ndim)
            with pp.ProcessPool(NUM_PROCESSES) as pool:
                sampler = emcee.EnsembleSampler(NWALKERS, ndim, log_prob, args=log_prob_args, backend=backend, pool=pool)
                sampler.run_mcmc(initial_position, NSTEPS, progress=True)
                pool.close()
                pool.join()
                pool.clear()
                pool.terminate()
                pool.restart()

    save(best_fits, args.model_slug+'_best-fit')

    delta_t = round(time.time() - start_time)
    print(f'Finished in {delta_t} seconds.')

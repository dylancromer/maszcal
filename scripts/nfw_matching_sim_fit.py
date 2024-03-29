import datetime
import numpy as np
import pathos.pools as pp
import emcee
from sklearn.gaussian_process.kernels import Matern
import multiprocessing
import supercubos
import maszcal.cosmology
import maszcal.corrections
import maszcal.data.sims
import maszcal.data.obs
import maszcal.density
import maszcal.fitutils
import maszcal.lensing
import maszcal.likelihoods
import maszcal.twohalo


PARAM_MINS = np.array([-2, 1])  # a_sz, con
PARAM_MAXES = np.array([2, 6])
LOWER_RADIUS_CUT = 0.1
UPPER_RADIUS_CUT = 5
COV_MAGNITUDE = 1.3
SIM_DATA = maszcal.data.sims.NBatta2010('data/NBatta2010/').cut_radii(LOWER_RADIUS_CUT, UPPER_RADIUS_CUT)
NUM_EMULATOR_SAMPLES = 1002
NUM_ERRORCHECK_SAMPLES = 1000
NUM_PROCESSES = 12
NUM_PRINCIPAL_COMPONENTS = 8
SAMPLE_SEED = 314
NWALKERS = 600
NSTEPS = 6000
WALKER_DISPERSION = 4e-3
DIR = 'data/NBatta2010/matching-model-fits/'
COV_DIR = 'data/NBatta2010/covariance/'
SETUP_SLUG = 'matching-nfw-only'


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def cut_covariance_to_radial_range(covariance):
    sim_data_full_radii = maszcal.data.sims.NBatta2010('data/NBatta2010/').radii
    indices = np.arange(sim_data_full_radii.size)
    cut_indices = indices[
        (LOWER_RADIUS_CUT <= sim_data_full_radii) & (sim_data_full_radii <= UPPER_RADIUS_CUT),
    ]
    return covariance[
        cut_indices[0]:cut_indices[-1]+1,
        cut_indices[0]:cut_indices[-1]+1,
    ]


def get_covariance_and_fisher():
    cov_sn_all_radii = np.loadtxt(COV_DIR + 'cov_sn_stacked_nbatta2010.txt')
    cov_lss_all_radii = np.loadtxt(COV_DIR + 'cov_lss_stacked_nbatta2010.txt')
    cov_sn = cut_covariance_to_radial_range(cov_sn_all_radii)
    cov_lss = cut_covariance_to_radial_range(cov_lss_all_radii)
    cov = cov_lss + cov_sn
    cov = np.diagflat(SIM_DATA.radii).T @ cov @ np.diagflat(SIM_DATA.radii)
    fisher = np.linalg.inv(cov)
    return cov, fisher


def get_density_model():
    return maszcal.density.MatchingNfwModel(
        cosmo_params=SIM_DATA.cosmology,
        mass_definition='crit',
        delta=500,
        comoving=True,
    )


def get_wrapped_lensing_func(density_model):
    def wrapper(rs, zs, mus, cons):
        masses = np.exp(mus)
        return density_model.excess_surface_density(rs, zs, masses, cons)
    return wrapper


def get_shear_model(lensing_func):
    masses = SIM_DATA.masses
    zs = np.repeat(SIM_DATA.redshifts, masses.shape[1])
    masses = masses.flatten()
    weights = np.ones_like(zs)
    return maszcal.lensing.MatchingShearModel(
        sz_masses=masses,
        redshifts=zs,
        lensing_weights=weights,
        lensing_func=lensing_func,
    )

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


def get_emulator_errors(PARAM_MINS, PARAM_MAXES, emulator, esd_func):
    rand = supercubos.LatinSampler().get_rand_sample(
        param_mins=PARAM_MINS,
        param_maxes=PARAM_MAXES,
        num_samples=NUM_ERRORCHECK_SAMPLES,
    )
    esds_check = _pool_map(esd_func, rand)
    data_check = SIM_DATA.radii[:, None] * esds_check
    emulated_data = emulator(rand)
    return (emulated_data - data_check)/data_check


def generate_header():
    terminator = '\n'
    configs = [
        f'PARAM_MINS = {PARAM_MINS}',
        f'PARAM_MAXES = {PARAM_MAXES}',
        f'LOWER_RADIUS_CUT = {LOWER_RADIUS_CUT}',
        f'UPPER_RADIUS_CUT = {UPPER_RADIUS_CUT}',
        f'COV_MAGNITUDE = {COV_MAGNITUDE}',
        f'WALKER_DISPERSION = {WALKER_DISPERSION}',
        f'NUM_EMULATOR_SAMPLES = {NUM_EMULATOR_SAMPLES}',
        f'NUM_PRINCIPAL_COMPONENTS = {NUM_PRINCIPAL_COMPONENTS}',
    ]
    header = [conf + terminator for conf in configs]
    return ''.join(header)


def save_arr(array, name):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    filename = name + '_' + timestamp

    header = generate_header()

    nothing = np.array([])

    np.savetxt(DIR + filename + '.header.txt', nothing, header=header)
    np.save(DIR + filename + '.npy', array)


def generate_chain_filename():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    return DIR + SETUP_SLUG + '-mcmc-chains_' + timestamp + '.h5'


if __name__ == '__main__':
    print('Creating emulator...')

    lh = supercubos.LatinSampler(rng=np.random.default_rng(seed=SAMPLE_SEED)).get_lh_sample(PARAM_MINS, PARAM_MAXES, NUM_EMULATOR_SAMPLES)

    density_model = get_density_model()
    wrapped_nfw_func = get_wrapped_lensing_func(density_model)
    shear_model = get_shear_model(wrapped_nfw_func)

    def wrapped_esd_func(params):
        a_sz = params[0:1]
        con = params[1:2]
        return shear_model.stacked_excess_surface_density(SIM_DATA.radii, a_sz, con).squeeze()

    esds = _pool_map(wrapped_esd_func, lh)

    data = SIM_DATA.radii[:, None] * esds
    emulator = maszcal.emulate.PcaEmulator.create_from_data(
        coords=lh,
        data=data,
        interpolator_class=maszcal.interpolate.RbfInterpolator,
        num_components=NUM_PRINCIPAL_COMPONENTS,
    )

    print('Saving emulator error samples...')

    emulator_errs = get_emulator_errors(PARAM_MINS, PARAM_MAXES, emulator, wrapped_esd_func)

    save_arr(emulator_errs, SETUP_SLUG+'-emulation-errors')

    cov, fisher = get_covariance_and_fisher()
    prefactor = 1/np.log((2*np.pi)**(cov.shape[0]/2) * np.sqrt(np.linalg.det(cov)))

    sim_stack = SIM_DATA.radii * (SIM_DATA.wl_signals.mean(axis=(1, 2)))

    def log_like(params, data):
        model = emulator(params[None, :]).flatten()
        return prefactor + maszcal.likelihoods.log_gaussian_shape(model, data, fisher)

    def log_prob(params, data):
        if np.all(PARAM_MINS < params) and np.all(params < PARAM_MAXES):
            return log_like(params, data)
        else:
            return - np.inf

    print('Obtaining maximum-likelihood optimization...')

    best_fit = maszcal.fitutils.global_minimize(
        lambda p: -log_like(p, sim_stack),
        PARAM_MINS,
        PARAM_MAXES,
        method='global-differential-evolution',
    )
    print(f'Maximum likelihood parameters are: {best_fit}')

    if not ((PARAM_MINS < best_fit).all() and (best_fit < PARAM_MAXES).all()):
        best_fit = (PARAM_MAXES + PARAM_MINS)/2
        print('Best fit is at edge of boundary: using middle of parameter space instead')

    ndim = PARAM_MINS.size
    initial_position = best_fit + WALKER_DISPERSION*np.random.randn(NWALKERS, ndim)

    chain_filename = generate_chain_filename()
    backend = emcee.backends.HDFBackend(chain_filename)
    backend.reset(NWALKERS, ndim)

    print('Beginning MCMC fit...')

    with pp.ProcessPool(NUM_PROCESSES) as pool:
        sampler = emcee.EnsembleSampler(NWALKERS, ndim, log_prob, args=(sim_stack,), backend=backend, pool=pool)
        sampler.run_mcmc(initial_position, NSTEPS, progress=True)
        pool.close()
        pool.join()
        pool.clear()
        pool.terminate()
        pool.restart()
    print(bcolors.OKGREEN + '--- Analysis complete ---' + bcolors.ENDC)

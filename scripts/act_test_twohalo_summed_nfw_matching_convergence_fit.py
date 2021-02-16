import datetime
import numpy as np
import scipy.interpolate
import pathos.pools as pp
import emcee
from sklearn.gaussian_process.kernels import Matern
import multiprocessing
import pixell.enmap
import supercubos
import maszcal.cosmology
import maszcal.corrections
import maszcal.data.test
import maszcal.density
import maszcal.fitutils
import maszcal.lensing
import maszcal.likelihoods
import maszcal.twohalo
import maszcal.filtering


PARAM_MINS = np.array([-2, 1])  # a_sz, con, alpha, beta
PARAM_MAXES = np.array([2, 6])
A_2H_MIN = 0
A_2H_MAX = 5
COSMOLOGY = maszcal.cosmology.CosmoParams()
DATA = maszcal.data.test.ActTestData('data/test-act/')
THETA_GRID = np.geomspace(DATA.radial_coordinates[0]/10, 10*DATA.radial_coordinates[-1], 80)
NUM_A_SZ_SAMPLES = 40
NUM_EMULATOR_SAMPLES = 1000
NUM_ERRORCHECK_SAMPLES = 1000
NUM_PRINCIPAL_COMPONENTS = 10
SAMPLE_SEED = 314
NUM_PROCESSES = 12
NUM_PROCESSES_LH = 12
COV_REDUCTION = 100
NWALKERS = 600
NSTEPS = 10000
WALKER_DISPERSION = 4e-3
ROOT_DIR = 'data/test-act/'
DIR = ROOT_DIR + 'matching-model-fits/'
SETUP_SLUG = 'matching-twohalosum-nfw-reduced-cov'
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
DEBUG = False


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_covariance_and_fisher():
    cov = np.loadtxt(ROOT_DIR + 'bin_2_covmat.txt')/COV_REDUCTION
    fisher = np.linalg.inv(cov)
    return cov, fisher


def get_density_model():
    return maszcal.density.MatchingNfwModel(
        cosmo_params=COSMOLOGY,
        mass_definition='crit',
        delta=500,
        comoving=True,
    )


def get_two_halo_conv():
    model = maszcal.twohalo.TwoHaloConvergenceModel(
        cosmo_params=COSMOLOGY,
        mass_definition='crit',
        delta=500,
    )
    return model.radius_space_convergence


def get_2halo_emulator(two_halo_conv):
    return maszcal.twohalo.TwoHaloEmulator.from_function(
        two_halo_func=two_halo_conv,
        r_grid=np.geomspace(0.0001, 200, 200),
        z_lims=np.array([0, 2.2]),
        mu_lims=np.log(np.array([1e13, 5e15])),
        num_emulator_samples=800,
    ).with_redshift_dependent_radii


def get_convergence_model(lensing_func):
    masses = DATA.masses
    zs = DATA.redshifts
    weights = np.ones_like(zs)
    return maszcal.lensing.MatchingConvergenceModel(
        sz_masses=masses,
        redshifts=zs,
        lensing_weights=weights,
        lensing_func=lensing_func,
        cosmo_params=COSMOLOGY,
    )


def _pool_map(func, array):
    pool = pp.ProcessPool(NUM_PROCESSES_LH)
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
    convs_check = _pool_map(esd_func, rand)
    data_check = THETA_GRID[:, None] * convs_check
    emulated_data = emulator(rand)
    return (emulated_data - data_check)/data_check


def get_2nd_order_emulator_errors(PARAM_MINS, PARAM_MAXES, emulator, esd_func):
    rand = supercubos.LatinSampler().get_rand_sample(
        param_mins=PARAM_MINS,
        param_maxes=PARAM_MAXES,
        num_samples=NUM_ERRORCHECK_SAMPLES,
    )
    data_check = _pool_map(esd_func, rand)
    emulated_data = emulator(rand)
    return (emulated_data - data_check)/data_check


def get_kmask():
    return pixell.enmap.read_map(ROOT_DIR + 'night_bcg_kmask.fits')


def generate_header():
    terminator = '\n'
    configs = [
        f'PARAM_MINS = {PARAM_MINS}',
        f'PARAM_MAXES = {PARAM_MAXES}',
        f'A_2H_MIN = {A_2H_MIN}',
        f'A_2H_MAX = {A_2H_MAX}',
        f'COV_REDUCTION = {COV_REDUCTION}',
        f'WALKER_DISPERSION = {WALKER_DISPERSION}',
        f'NUM_EMULATOR_SAMPLES = {NUM_EMULATOR_SAMPLES}',
        f'NUM_PRINCIPAL_COMPONENTS = {NUM_PRINCIPAL_COMPONENTS}',
    ]
    header = [conf + terminator for conf in configs]
    return ''.join(header)


def save_arr(array, name):
    if not DEBUG:
        filename = name + '_' + TIMESTAMP

        header = generate_header()

        nothing = np.array([])

        np.savetxt(DIR + filename + '.header.txt', nothing, header=header)
        np.save(DIR + filename + '.npy', array)


def generate_chain_filename():
    return DIR + SETUP_SLUG + '-mcmc-chains_' + TIMESTAMP + '.h5'


if __name__ == '__main__':
    print('Creating 2 halo term...')
    two_halo_conv = get_two_halo_conv()
    _two_halo_emulator = get_2halo_emulator(two_halo_conv)
    def two_halo_emulator(*args):
        return np.moveaxis(_two_halo_emulator(*args), 1, 0)
    two_halo_conv_model = get_convergence_model(two_halo_emulator)
    def wrapped_2h_conv(a_sz): return two_halo_conv_model.stacked_convergence(THETA_GRID, a_sz)
    two_halo_term = wrapped_2h_conv(np.zeros(1)).squeeze()
    two_halo_interpolator = scipy.interpolate.interp1d(THETA_GRID, two_halo_term, kind='cubic', axis=0)

    print('Creating 1 halo emulator...')
    one_halo_lh = supercubos.LatinSampler(rng=np.random.default_rng(seed=SAMPLE_SEED)).get_lh_sample(PARAM_MINS, PARAM_MAXES, NUM_EMULATOR_SAMPLES)
    density_model = get_density_model()
    one_halo_conv_model = get_convergence_model(density_model.convergence)

    def wrapped_1h_esd_func(params):
        a_sz = params[0:1]
        con = params[1:2]
        return one_halo_conv_model.stacked_convergence(THETA_GRID, a_sz, con).squeeze()

    one_halo_convs = _pool_map(wrapped_1h_esd_func, one_halo_lh)
    one_halo_data = THETA_GRID[:, None] * one_halo_convs
    one_halo_emulator = maszcal.emulate.PcaEmulator.create_from_data(
        coords=one_halo_lh,
        data=one_halo_data,
        interpolator_class=maszcal.interpolate.GaussianProcessInterpolator,
        interpolator_kwargs={'kernel': Matern()},
        num_components=NUM_PRINCIPAL_COMPONENTS,
    )

    print('Saving 1 halo emulator error samples...')
    one_halo_emulator_errs = get_emulator_errors(PARAM_MINS, PARAM_MAXES, one_halo_emulator, wrapped_1h_esd_func)
    save_arr(one_halo_emulator_errs, SETUP_SLUG+'-1h-emulation-errors')

    def wrapped_1h_function(thetas, params):
        return one_halo_emulator.with_new_radii(THETA_GRID, thetas, params[None, :]).squeeze()/thetas

    kmask = get_kmask()

    filtered_two_halo_term = maszcal.filtering.filter_kappa(two_halo_interpolator, DATA.radial_coordinates, kmask)

    def filtered_1h_function(params):
        return maszcal.filtering.filter_kappa(lambda thetas: wrapped_1h_function(thetas, params), DATA.radial_coordinates, kmask)

    filtered_1_halo_samples = _pool_map(filtered_1h_function, one_halo_lh)

    second_order_1h_emulator = maszcal.emulate.PcaEmulator.create_from_data(
        coords=one_halo_lh,
        data=filtered_1_halo_samples,
        interpolator_class=maszcal.interpolate.GaussianProcessInterpolator,
        interpolator_kwargs={'kernel': Matern()},
        num_components=NUM_PRINCIPAL_COMPONENTS,
    )

    print('Saving 2nd order 1 halo emulator error samples...')
    second_order_1h_emulator_errs = get_2nd_order_emulator_errors(PARAM_MINS, PARAM_MAXES, second_order_1h_emulator, filtered_1h_function)
    save_arr(second_order_1h_emulator_errs, SETUP_SLUG+'-2nd-order-1h-emulation-errors')

    def model_function(params):
        a_2h = params[0]
        one_halo_params = params[1:]
        return second_order_1h_emulator(one_halo_params[None, :]).flatten() + a_2h*filtered_two_halo_term

    cov, fisher = get_covariance_and_fisher()
    prefactor = 1/np.log((2*np.pi)**(cov.shape[0]/2) * np.sqrt(np.linalg.det(cov)))

    sim_stack = DATA.stacked_wl_signal

    def log_like(params, data):
        model = model_function(params).flatten()
        return prefactor + maszcal.likelihoods.log_gaussian_shape(model, data, fisher)

    full_param_mins = np.concatenate((np.array([A_2H_MIN]), PARAM_MINS))
    full_param_maxes = np.concatenate((np.array([A_2H_MAX]), PARAM_MAXES))

    def log_prob(params, data):
        if np.all(full_param_mins < params) and np.all(params < full_param_maxes):
            return log_like(params, data)
        else:
            return - np.inf

    print('Obtaining maximum-likelihood optimization...')

    best_fit = maszcal.fitutils.global_minimize(
        lambda p: -log_like(p, sim_stack),
        full_param_mins,
        full_param_maxes,
        method='global-differential-evolution',
    )

    print(f'Maximum likelihood parameters are: {best_fit}')

    ndim = full_param_mins.size
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

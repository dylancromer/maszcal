import datetime
import numpy as np
import scipy.interpolate
import pathos.pools as pp
import emcee
import dill
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


PARAM_MINS = np.array([-2, 1, 0.1, 2.1])  # a_sz, con, alpha, beta
PARAM_MAXES = np.array([2, 6, 2.0, 8.1])
A_2H_MIN = 0
A_2H_MAX = 5
GAMMA = 0.2
USE_PRIOR = False
MEAN_PRIOR_ALPHA = 0.88
PRIOR_ALPHA_STD = 0.3
COSMOLOGY = maszcal.cosmology.CosmoParams()
DATA = maszcal.data.test.ActTestData('data/test-act/')
THETA_GRID = np.geomspace(DATA.radial_coordinates[0]/10, 10*DATA.radial_coordinates[-1], 80)
NUM_A_SZ_SAMPLES = 40
NUM_EMULATOR_SAMPLES = 2000
NUM_ERRORCHECK_SAMPLES = 1000
NUM_PRINCIPAL_COMPONENTS = 8
SAMPLE_SEED = 314
NUM_PROCESSES = 12
NWALKERS = 600
NSTEPS = 10000
WALKER_DISPERSION = 4e-3
ROOT_DIR = 'data/test-act/'
DIR = ROOT_DIR + 'matching-model-fits/'
SETUP_SLUG = 'matching-twohalosum-baryons'
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")


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
    cov = np.loadtxt(ROOT_DIR + 'bin_2_covmat.txt')
    fisher = np.linalg.inv(cov)
    return cov, fisher


def get_density_model():
    return maszcal.density.MatchingGnfw(
        cosmo_params=COSMOLOGY,
        mass_definition='crit',
        delta=500,
        comoving_radii=True,
        nfw_class=maszcal.density.MatchingNfwModel,
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
    data_check = THETA_GRID[:, None] * esds_check
    emulated_data = emulator(rand)
    return (emulated_data - data_check)/data_check


def get_kmask():
    return pixell.enmap.read_map(ROOT_DIR + 'night_bcg_kmask.fits')


def generate_header():
    terminator = '\n'
    configs = [
        f'PARAM_MINS = {PARAM_MINS}'
        f'PARAM_MAXES = {PARAM_MAXES}'
        f'A_2H_MIN = {A_2H_MIN}'
        f'A_2H_MAX = {A_2H_MAX}'
        f'GAMMA = {GAMMA}'
        f'USE_PRIOR = {USE_PRIOR}'
        f'MEAN_PRIOR_ALPHA = {MEAN_PRIOR_ALPHA}'
        f'PRIOR_ALPHA_STD  = {PRIOR_ALPHA_STD}'
        f'WALKER_DISPERSION = {WALKER_DISPERSION}'
        f'NUM_EMULATOR_SAMPLES = {NUM_EMULATOR_SAMPLES}'
        f'NUM_PRINCIPAL_COMPONENTS = {NUM_PRINCIPAL_COMPONENTS}'
    ]
    header = [conf + terminator for conf in configs]
    return ''.join(header)


def save_arr(array, name):
    filename = name + '_' + TIMESTAMP

    header = generate_header()

    nothing = np.array([])

    np.savetxt(DIR + filename + '.header.txt', nothing, header=header)
    np.save(DIR + filename + '.npy', array)


def save_emulator(emulator, name):
    filename = name + '_' + TIMESTAMP

    header = generate_header()

    nothing = np.array([])

    np.savetxt(DIR + filename + '.header.txt', nothing, header=header)
    with open(DIR + filename + '.pickle', 'wb') as file:
        dill.dump(emulator, file)


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
    two_halo_convs = wrapped_2h_conv(np.zeros(1))

    save_arr(THETA_GRID, SETUP_SLUG+'_theta-grid')
    save_arr(two_halo_convs, SETUP_SLUG+'_2-halo_term')

    print('Creating 1 halo emulator...')
    one_halo_lh = supercubos.LatinSampler(rng=np.random.default_rng(seed=SAMPLE_SEED)).get_lh_sample(PARAM_MINS, PARAM_MAXES, NUM_EMULATOR_SAMPLES)
    density_model = get_density_model()
    one_halo_conv_model = get_convergence_model(density_model.convergence)

    def wrapped_1h_esd_func(params):
        a_sz = params[0:1]
        con = params[1:2]
        alpha = params[2:3]
        beta = params[3:4]
        gamma = np.array([GAMMA])
        return one_halo_conv_model.stacked_convergence(THETA_GRID, a_sz, con, alpha, beta, gamma).squeeze()

    one_halo_esds = _pool_map(wrapped_1h_esd_func, one_halo_lh)
    one_halo_data = THETA_GRID[:, None] * one_halo_esds

    save_arr(one_halo_esds, SETUP_SLUG+'_1h-esds_sample')
    save_arr(one_halo_lh, SETUP_SLUG+'_1h-latin-hypercube')
    print(bcolors.OKGREEN + '--- Data for emulation saved ---' + bcolors.ENDC)

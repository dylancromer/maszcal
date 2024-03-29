import datetime
import numpy as np
import pathos.pools as pp
from sklearn.gaussian_process.kernels import Matern
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


PARAM_MINS = np.array([-2, 1, 0.1, 0.1])  # a_sz, con, alpha, beta
PARAM_MAXES = np.array([2, 6, 2.1, 8.1])
GAMMA = 0.2

MIN_MU = np.log(3e13)
MAX_MU = np.log(3e14)

Z_MIN = 0.2
Z_MAX = 1

RADII = np.geomspace(1e-1, 20, 100)

SEED = 13

NUM_CLUSTERS = 400
NUM_ERRORCHECK_SAMPLES = 1000
NUM_PROCESSES = 12

MIN_SAMPLES = 200
MAX_SAMPLES = 2000
SAMPLE_STEP_SIZE = 200

FID_SAMPLE_SIZE = 1000
FID_PRINCIPAL_COMPONENTS = 8

TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
DIR = 'data/emulator/'
SETUP_SLUG = 'emulator-errors_bary'


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_density_model():
    return maszcal.density.MatchingGnfw(
        cosmo_params=maszcal.cosmology.CosmoParams(),
        mass_definition='crit',
        delta=500,
        comoving_radii=True,
        nfw_class=maszcal.density.MatchingNfwModel,
    )


def get_shear_model(lensing_func):
    rng = np.random.default_rng(seed=SEED)
    sz_mus = (MAX_MU - MIN_MU) * rng.random(size=NUM_CLUSTERS) + MIN_MU
    sz_masses = np.exp(sz_mus)
    zs = Z_MAX*rng.random(size=NUM_CLUSTERS) + Z_MIN
    weights = np.ones(NUM_CLUSTERS)
    return maszcal.lensing.MatchingShearModel(
        sz_masses=sz_masses,
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
    data_check = RADII[:, None] * esds_check
    emulated_data = emulator(rand)
    return (emulated_data - data_check)/data_check


def generate_header():
    terminator = '\n'
    configs = [
        f'PARAM_MINS = {PARAM_MINS}'
        f'PARAM_MAXES = {PARAM_MAXES}'
        f'GAMMA = {GAMMA}'
        f'MIN_MU = {MIN_MU}'
        f'MAX_MU = {MAX_MU}'
        f'Z_MIN = {Z_MIN}'
        f'Z_MAX = {Z_MAX}'
        f'RADII = {RADII}'
        f'SEED = {SEED}'
        f'NUM_CLUSTERS = {NUM_CLUSTERS}'
        f'NUM_ERRORCHECK_SAMPLES = {NUM_ERRORCHECK_SAMPLES}'
        f'MIN_SAMPLES = {MIN_SAMPLES}'
        f'MAX_SAMPLES = {MAX_SAMPLES}'
        f'SAMPLE_STEP_SIZE = {SAMPLE_STEP_SIZE}'
        f'FID_SAMPLE_SIZE = {FID_SAMPLE_SIZE}'
        f'FID_PRINCIPAL_COMPONENTS = {FID_PRINCIPAL_COMPONENTS}'
    ]
    header = [conf + terminator for conf in configs]
    return ''.join(header)


def save_arr(array, name):
    filename = name + '_' + TIMESTAMP

    header = generate_header()

    nothing = np.array([])

    np.savetxt(DIR + filename + '.header.txt', nothing, header=header)
    np.save(DIR + filename + '.npy', array)


def do_estimation(sample_size, num_pcs):
    rng = np.random.default_rng(seed=SEED)
    lh = supercubos.LatinSampler(rng=rng).get_lh_sample(PARAM_MINS, PARAM_MAXES, sample_size)

    density_model = get_density_model()
    shear_model = get_shear_model(density_model.excess_surface_density)

    def wrapped_esd_func(params):
        a_sz = params[0:1]
        con = params[1:2]
        alpha = params[2:3]
        beta = params[3:4]
        gamma = np.array([GAMMA])
        return shear_model.stacked_excess_surface_density(RADII, a_sz, con, alpha, beta, gamma).squeeze()

    esds = _pool_map(wrapped_esd_func, lh)

    data = RADII[:, None] * esds
    emulator = maszcal.emulate.PcaEmulator.create_from_data(
        coords=lh,
        data=data,
        interpolator_class=maszcal.interpolate.GaussianProcessInterpolator,
        interpolator_kwargs={'kernel': Matern()},
        num_components=num_pcs,
    )

    return get_emulator_errors(PARAM_MINS, PARAM_MAXES, emulator, wrapped_esd_func)


if __name__ == '__main__':
    sample_sizes = np.arange(MIN_SAMPLES, MAX_SAMPLES+SAMPLE_STEP_SIZE, SAMPLE_STEP_SIZE)

    print('Calculating errors as a function of LH size...')
    for num_samples in sample_sizes:
        emulator_errs = do_estimation(num_samples, FID_PRINCIPAL_COMPONENTS)
        save_arr(emulator_errs, SETUP_SLUG+f'_nsamples={num_samples}')
        print(f'Finished calculation for LH of size {num_samples}')

    print('Calculating errors as a function of principal components...')
    num_component_range = np.arange(3, 13, 1)
    for num_components in num_component_range:
        emulator_errs = do_estimation(FID_SAMPLE_SIZE, num_components)
        save_arr(emulator_errs, SETUP_SLUG+f'_ncomponents={num_components}')
        print(f'Finished calculation for {num_components} PCs')

    print(bcolors.OKGREEN + '--- Analysis complete ---' + bcolors.ENDC)

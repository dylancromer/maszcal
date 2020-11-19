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


PARAM_MINS = np.array([-2, 0, 1, 0.1, 2.2])  # a_sz, con, alpha, beta
PARAM_MAXES = np.array([2, 5, 6, 2.1, 8.1])
GAMMA = 0.2

MIN_MU = np.log(3e13)
MAX_MU = np.log(3e14)

Z_MIN = 0.2
Z_MAX = 1

FROM_ARCMIN = 2 * np.pi / 360 / 60
TO_ARCMIN = 1/FROM_ARCMIN
THETAS = np.linspace(0.01*FROM_ARCMIN, 20*FROM_ARCMIN, 88)

SEED = 13

NUM_CLUSTERS = 101
NUM_ERRORCHECK_SAMPLES = 10
NUM_PROCESSES = 1

MIN_SAMPLES = 200
MAX_SAMPLES = 2000
SAMPLE_STEP_SIZE = 200

FID_SAMPLE_SIZE = 1000
FID_PRINCIPAL_COMPONENTS = 8

TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
DIR = 'data/emulator/'
SETUP_SLUG = 'emulator-errors_bary-2h_convergence'


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
    return maszcal.density.Gnfw(
        cosmo_params=maszcal.cosmology.CosmoParams(),
        mass_definition='mean',
        delta=200,
        comoving_radii=True,
        nfw_class=maszcal.density.NfwModel,
    )


def get_hmf_interp():
    return maszcal.tinker.HmfInterpolator(
        mu_samples=np.log(np.geomspace(1e12, 1e16, 600)),
        redshift_samples=np.linspace(0.01, 4, 120),
        delta=200,
        mass_definition='mean',
        cosmo_params=maszcal.cosmology.CosmoParams(),
    )


def get_two_halo_convergence():
    model = maszcal.twohalo.TwoHaloConvergenceModel(
        cosmo_params=maszcal.cosmology.CosmoParams(),
        mass_definition='mean',
        delta=200,
    )
    return model.radius_space_convergence


def get_2h_emulator(two_halo_conv):
    return maszcal.twohalo.TwoHaloEmulator.from_function(
        two_halo_func=two_halo_conv,
        r_grid=np.geomspace(1e-4, 100, 120),
        z_lims=np.array([0, Z_MAX+0.01]),
        mu_lims=np.log(np.array([1e13, 3e15])),
        num_emulator_samples=800,
        separate_mu_and_z_axes=True,
    ).with_redshift_dependent_radii


def get_corrected_lensing_func(density_model, two_halo_emulator):
    return maszcal.corrections.TwoHaloCorrection(
        one_halo_func=density_model.convergence,
        two_halo_func=two_halo_emulator,
    ).corrected_profile


def get_convergence_model(hmf_interp, lensing_func):
    rng = np.random.default_rng(seed=SEED)
    sz_mus = (MAX_MU - MIN_MU) * rng.random(size=NUM_CLUSTERS) + MIN_MU
    sz_masses = np.exp(sz_mus)
    zs = Z_MAX*rng.random(size=NUM_CLUSTERS) + Z_MIN
    weights = np.ones(NUM_CLUSTERS)
    cosmo_params = maszcal.cosmology.CosmoParams()
    return maszcal.lensing.ScatteredMatchingConvergenceModel(
        sz_masses=sz_masses,
        redshifts=zs,
        lensing_weights=weights,
        cosmo_params=cosmo_params,
        lensing_func=lensing_func,
        logmass_prob_dist_func=hmf_interp,
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


def get_emulator_errors(PARAM_MINS, PARAM_MAXES, emulator, conv_func):
    rand = supercubos.LatinSampler().get_rand_sample(
        param_mins=PARAM_MINS,
        param_maxes=PARAM_MAXES,
        num_samples=NUM_ERRORCHECK_SAMPLES,
    )
    convs_check = _pool_map(conv_func, rand)
    data_check = convs_check
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
        f'THETAS = {THETAS}'
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
    two_halo_conv = get_two_halo_convergence()
    two_halo_emulator = get_2h_emulator(two_halo_conv)
    corrected_lensing_func = get_corrected_lensing_func(density_model, two_halo_emulator)
    hmf_interp = get_hmf_interp()
    conv_model = get_convergence_model(hmf_interp, corrected_lensing_func)

    def wrapped_conv_func(params):
        a_sz = params[0:1]
        a_2h = params[1:2]
        con = params[2:3]
        alpha = params[3:4]
        beta = params[4:5]
        gamma = np.array([GAMMA])
        conv = conv_model.stacked_convergence(THETAS, a_sz, a_2h, con, alpha, beta, gamma).squeeze()
        return THETAS * conv

    convs = _pool_map(wrapped_conv_func, lh)

    emulator = maszcal.emulate.PcaEmulator.create_from_data(
        coords=lh,
        data=convs,
        interpolator_class=maszcal.interpolate.GaussianProcessInterpolator,
        interpolator_kwargs={'kernel': Matern()},
        num_components=num_pcs,
    )

    return get_emulator_errors(PARAM_MINS, PARAM_MAXES, emulator, wrapped_conv_func)


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

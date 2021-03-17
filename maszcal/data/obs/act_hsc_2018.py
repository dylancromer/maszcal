import numpy as np
import scipy.interpolate


class ActHsc2018:
    # TODO check if there are any little-h factors in here to be taken care of
    @classmethod
    def _load_covariance(cls, directory):
        act_rs, _, act_errs = np.loadtxt(directory + 'deltaSigma.dat', unpack=True, skiprows=1, usecols=[0, 3, 4])
        full_err_cov = np.loadtxt(directory + 'cov_lss_stacked.dat')
        con_err_cov = np.loadtxt(directory + 'con_cov.txt')
        cov_full = (act_errs * np.identity(len(act_errs)) * act_errs) + full_err_cov + con_err_cov
        return act_rs, cov_full

    @classmethod
    def _interpolate_covariance(cls, directory, radii):
        act_rs, act_cov = cls._load_covariance(directory)
        act_diag = np.diag(act_cov)
        _diag_interpolator = scipy.interpolate.interp1d(act_rs, np.log(act_diag), kind='linear')
        def diag_interpolator(r): return np.exp(_diag_interpolator(r))
        return np.diagflat(diag_interpolator(radii))

    @classmethod
    def covariance(cls, directory, radii):
        return cls._interpolate_covariance(directory, radii)

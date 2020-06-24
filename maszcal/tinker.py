from dataclasses import dataclass
import numpy as np
import astropy.units as u
from scipy.interpolate import InterpolatedUnivariateSpline as iuSpline
import scipy.integrate
#np.seterr(divide='ignore', invalid='ignore')


tinker_data = np.transpose([[float(x) for x in line.split()]
                            for line in
    """200 0.186 1.47 2.57 1.19
       300 0.200 1.52 2.25 1.27
       400 0.212 1.56 2.05 1.34
       600 0.218 1.61 1.87 1.45
       800 0.248 1.87 1.59 1.58
      1200 0.255 2.13 1.51 1.80
      1600 0.260 2.30 1.46 1.97
      2400 0.260 2.53 1.44 2.24
      3200 0.260 2.66 1.41 2.44""".split('\n')])

tinker_splines = None


def tinker_params_spline(delta, z=None):
    global tinker_splines
    if tinker_splines is None:
        tinker_splines = []
        D, data = np.log(tinker_data[0]), tinker_data[1:]
        for y in data:
            # Extend to large delta
            p = np.polyfit(D[-2:], y[-2:], 1)
            x = np.hstack((D, D[-1]+3.))
            y = np.hstack((y, np.polyval(p, x[-1])))
            tinker_splines.append(iuSpline(x, y, k=2))
    A0, a0, b0, c0 = [ts(np.log(delta)) for ts in tinker_splines]
    if z is None:
        return A0, a0, b0, c0

    z = np.asarray(z)
    A = A0 * (1 + z)**(-.14)
    a = a0 * (1 + z)**(-.06)
    alpha = 10.**(-((.75/np.log10(delta/75.))**1.2))
    b = b0 * (1 + z)**(-alpha)
    c = np.zeros(np.shape(z)) + c0
    return A, a, b, c


def tinker_params_analytic(delta, z=None):
    alpha = None
    if np.asarray(delta).ndim == 0:  # scalar delta.
        A0, a0, b0, c0 = [p[0] for p in
                          tinker_params(np.array([delta]), z=None)]
        if z is not None:
            if delta < 75.:
                alpha = 1.
            else:
                alpha = 10.**(-((.75 / np.log10(delta/75.))**1.2))
    else:
        log_delta = np.log10(delta)
        A0 = 0.1*log_delta - 0.05
        a0 = 1.43 + (log_delta - 2.3)**(1.5)
        b0 = 1.0 + (log_delta - 1.6)**(-1.5)
        c0 = log_delta - 2.35
        A0[delta > 1600] = .26
        a0[log_delta < 2.3] = 1.43
        b0[log_delta < 1.6] = 1.0
        c0[c0 < 0] = 0
        c0 = 1.2 + c0**1.6
    if z is None:
        return A0, a0, b0, c0
    A = A0 * (1 + z)**-.14
    a = a0 * (1 + z)**-.06
    if alpha is None:
        alpha = 10**(-((.75/np.log10(delta/75))**1.2))
        alpha[delta < 75] = 1
    b = b0 * (1 + z)**-alpha
    c = np.zeros(np.shape(z)) + c0
    return A, a, b, c


tinker_params = tinker_params_spline


def tinker_f(sigma, params):
    A, a, b, c = params
    return A * ((sigma/b)**(-a) + 1) * np.exp(-c/sigma**2)


def top_hatf(kr):
    """
    Returns the Fourier transform of the spherical top-hat function
    evaluated at a given k*R.
    """
    return np.nan_to_num(3*(np.sin(kr) - (kr)*np.cos(kr))) / ((kr)**3)


def sigma_sq_integral(rs, power_spect, ks):
    """
    Determines the sigma^2 parameter over the m-z grid by integrating
    over k.
    """
    integrand = top_hatf(rs[None, ...] * ks[:, None, None])**2 * power_spect.T[:, None, :] * ks[:, None, None]**2
    return scipy.integrate.simps(integrand/(2 * np.pi**2), x=ks, axis=0)


def fnl_correction(sigma2, fnl):
    d_c = 1.686
    S3 = 3.15e-4 * fnl / (sigma2**(0.838/2.0))
    del_cor = np.sqrt(1 - d_c*S3/3.0)
    return np.exp(S3 * d_c**3/(sigma2*6.0))*(d_c**2/(6.0*del_cor)*(-0.838*S3)+del_cor)


def dsigma_dkmax_dM(M, z, rho, k, P, comoving=False):
    """
    M      is  (nM)  or  (nM, nz)
    z      is  (nz)
    rho    is  (nz)
    delta  is  (nz)  or  scalar
    k      is  (nk)
    P      is  (nz,nk)

    Somewhat awkwardly, k and P are comoving.  rho really isn't.

    return is  (nM,nz)
    """
    if M.ndim == 1:
        M = M[:, None]

    R = radius_from_mass(M, rho)
    if not comoving:
        R = R * np.transpose(1+z)

    sigma_k = np.zeros(len(k)-3)
    kmax_out = np.zeros(len(k)-3)

    for ii in range(len(k)-3):
        iii = ii + 3
        sigma_k[iii] = sigma_sq_integral(R, P[:iii], k[:iii])**.5
        kmax_out[iii] = k[iii]

    return kmax_out, sigma_k


def tinker_bias_params(delta):
    y = np.log10(delta)
    big_a = 1 + 0.24*y*np.exp(-(4/y)**4)
    a = 0.44*y - 0.88
    big_b = 0.183
    b = 1.5
    big_c = 0.019 + 0.107*y + 0.19*np.exp(-(4./y)**4)
    c = 2.4
    return big_a, a, big_b, b, big_c, c


@dataclass
class TinkerModel:
    delta: int
    mass_definition: str
    astropy_cosmology: object
    comoving: bool

    def __post_init__(self):
        if self.mass_definition not in ('mean', 'crit'):
            raise ValueError('mass_definition must be \'mean\' or \'crit\'.')

    def _rho(self, zs):
        if self.comoving:
            rhos = (
                self.astropy_cosmology.Om0
                * self.astropy_cosmology.critical_density0
                * np.ones(zs.shape)
            ).to(u.Msun/u.Mpc**3).value
        else:
            rhos = (
                self.astropy_cosmology.Om(zs)
                * self.astropy_cosmology.critical_density(zs)
            ).to(u.Msun/u.Mpc**3).value

        return rhos

    def radius_from_mass(self, masses, zs):
        """
        Convert mass M to radius R assuming density rho.
        """
        rhos = self._rho(zs)
        return (3*masses / (4*np.pi*rhos))**(1/3)

    def _get_delta_means(self, zs):
        if self.mass_definition == 'mean':
            delta_means = self.delta * np.ones(zs.shape)
        elif self.mass_definition == 'crit':
            delta_means = self.delta / self.astropy_cosmology.Om(zs)
        return delta_means


class TinkerHmf(TinkerModel):
    def dn_dlnm(self, masses, zs, ks, power_spect):
        delta_means = self._get_delta_means(zs)  # converts delta_c to delta_m such that mass is the same
        return self._dn_dlnm(masses, zs, delta_means, ks, power_spect)

    def _dn_dlnm(self, masses, zs, delta_means, ks, power_spectrum):
        """
        M      is  (nM)  or  (nM, nz)
        z      is  (nz)
        rho    is  (nz)
        delta  is  (nz)  or  scalar
        k      is  (nk)
        P      is  (nz,nk)

        Somewhat awkwardly, k and P are comoving.  rho really isn't.

        return is  (nM,nz)
        """
        if masses.ndim == 1:
            masses = masses[:, None]

        radii = self.radius_from_mass(masses, zs)

        sigma = np.sqrt(sigma_sq_integral(radii, power_spectrum, ks))

        if radii.shape[-1] == 1:
            dlogs = -np.gradient(np.log(sigma[..., 0]))[:, None]
        else:
            dlogs = -np.gradient(np.log(sigma))[0]

        tp = tinker_params(delta_means, zs)
        tf = tinker_f(sigma, tp)

        if masses.shape[-1] == 1:
            dmasses = np.gradient(np.log(masses[:, 0]))[:, None] * masses
        else:
            dmasses = np.gradient(np.log(masses))[0] * masses

        return tf * self._rho(zs) * dlogs / dmasses


class TinkerBias(TinkerModel):
    def sigma_sq_integral(self, rs, power_spect, ks):
        """
        Determines the sigma^2 parameter over the m-z grid by integrating
        over k.
        """
        integrand = top_hatf(rs[None, ...] * ks[:, None])**2 * power_spect.T * ks[:, None]**2
        return scipy.integrate.simps(integrand/(2 * np.pi**2), x=ks, axis=0)

    def _bias_from_sigma(self, sigma, delta):
        big_a, a, big_b, b, big_c, c = tinker_bias_params(delta)
        delta_collapse = 1.686
        nu = delta_collapse/sigma
        return 1 - big_a*nu**a / (nu**a + delta_collapse**a) + big_b*nu**b + big_c*nu**c

    def bias(self, masses, zs, ks, power_spectrum):
        delta_means = self._get_delta_means(zs)  # converts delta_c to delta_m such that mass is the same

        radii = self.radius_from_mass(masses, zs)
        sigmas = np.sqrt(self.sigma_sq_integral(radii, power_spectrum, ks))

        return self._bias_from_sigma(sigmas, delta_means)

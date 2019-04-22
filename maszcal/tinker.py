from __future__ import division
from builtins import zip
from builtins import range
from past.utils import old_div
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as iuSpline
from scipy.integrate import simps
np.seterr(divide='ignore', invalid='ignore')

# Tinker stuff
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
            # Extend to large Delta
            p = np.polyfit(D[-2:], y[-2:], 1)
            x = np.hstack((D, D[-1]+3.))
            y = np.hstack((y, np.polyval(p, x[-1])))
            tinker_splines.append(iuSpline(x, y, k=2))
    A0, a0, b0, c0 = [ts(np.log(delta)) for ts in tinker_splines]
    if z is None:
        return A0, a0, b0, c0

    z = np.asarray(z)
    A = A0 * (1 + z)**-.14
    a = a0 * (1 + z)**-.06
    alpha = 10.**(-(((old_div(.75,np.log10(old_div(delta,75.)))))**1.2))
    b = b0 * (1 + z)**-alpha
    c = np.zeros(np.shape(z)) + c0
    return A, a, b, c


def tinker_params_analytic(delta, z=None):
    alpha = None
    if np.asarray(delta).ndim == 0: # scalar delta.
        A0, a0, b0, c0 = [p[0] for p in
                          tinker_params(np.array([delta]), z=None)]
        if z != None:
            if delta < 75.:
                alpha = 1.
            else:
                alpha = 10.**(-(((old_div(.75,np.log10(old_div(delta,75.)))))**1.2))
    else:
        log_delta = np.log10(delta)
        A0 = 0.1*log_delta - 0.05
        a0 = 1.43 + (log_delta - 2.3)**(1.5)
        b0 = 1.0 + (log_delta - 1.6)**(-1.5)
        c0 = log_delta - 2.35
        A0[delta>1600] = .26
        a0[log_delta < 2.3] = 1.43
        b0[log_delta < 1.6] = 1.0
        c0[c0<0] = 0.
        c0 = 1.2 + c0**1.6
    if z is None:
        return A0, a0, b0, c0
    A = A0 * (1 + z)**-.14
    a = a0 * (1 + z)**-.06
    if alpha is None:
        alpha = 10.**(-(((old_div(.75,np.log10(old_div(delta,75.)))))**1.2))
        alpha[delta<75.] = 1.
    b = b0 * (1 + z)**-alpha
    c = np.zeros(np.shape(z)) + c0
    return A, a, b, c


tinker_params = tinker_params_spline


def tinker_f(sigma, params):
    A, a, b, c = params
    return A * ( (old_div(sigma,b))**-a + 1 ) * np.exp(old_div(-c,sigma**2))


def radius_from_mass(M, rho):
    """
    Convert mass M to radius R assuming density rho.
    """
    return (3.*M / (4.*np.pi*rho))**(old_div(1,3.))


def top_hatf(kR):
    """
    Returns the Fourier transform of the spherical top-hat function
    evaluated at a given k*R.
    """
    out = old_div(np.nan_to_num(3*(np.sin(kR) - (kR)*np.cos(kR))),((kR)**3))
    return out


def sigma_sq_integral(R_grid, power_spt, k_val):
    #TODO: what the fuck is going on here
    """
    Determines the sigma^2 parameter over the m-z grid by integrating
    over k.

    Notes:
    -------
    * Fastest python solution I have found for this. There is probably a
      smarter way using numpy arrays.

    """
    to_integ = np.array(
        [
            top_hatf(R_grid * k)**2 * np.tile(power_spt[:,i], (R_grid.shape[0],1)) * k**2
            for k,i in zip(k_val,np.arange(len(k_val)))
        ]
    )

    return simps(to_integ/(2*np.pi**2), x=k_val, axis=0)


def fnl_correction(sigma2,fnl):
    d_c = 1.686
    S3 = 3.15e-4 * fnl / (sigma2**(old_div(0.838,2.0)))
    del_cor = np.sqrt(1 - d_c*S3/3.0)
    ans = np.exp(S3 * d_c**3/(sigma2*6.0))*(d_c**2/(6.0*del_cor)*(-0.838*S3)+del_cor)
    return ans


def dn_dlogM(M, z, rho, delta, k, P, comoving=False):
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
        M = M[:,None]

    R = radius_from_mass(M, rho)
    if not comoving:
        R =  R * np.transpose(1+z)

    sigma = np.sqrt(sigma_sq_integral(R, P, k))

    if R.shape[-1] == 1:
        dlogs = -np.gradient(np.log(sigma[...,0]))[:,None]
    else:
        dlogs = -np.gradient(np.log(sigma))[0]

    tp = tinker_params(delta, z)
    tf = tinker_f(sigma, tp)

    if M.shape[-1] == 1:
        dM = np.gradient(np.log(M[:,0]))[:,None] * M
    else:
        dM = np.gradient(np.log(M))[0] * M

    return tf * rho * dlogs / dM


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
        M = M[:,None]

    R = radius_from_mass(M, rho)
    if not comoving:
        R =  R * np.transpose(1+z)

    sigma_k = np.zeros(len(k)-3)
    kmax_out = np.zeros(len(k)-3)

    for ii in range(len(k)-3):
        iii = ii + 3
        sigma_k[iii] = sigma_sq_integral(R, P[:iii], k[:iii])**.5
        kmax_out[iii] = k[iii]

    return kmax_out, sigma_k


def tinker_bias_params(Delta):
    y = np.log10(Delta)
    A = 1.0 + 0.24*y*np.exp(-(old_div(4.,y))**4.)
    a = 0.44*y - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107*y + 0.19*np.exp(-(old_div(4.,y))**4.)
    c = 2.4

    return A,a,B,b,C,c


def tinker_bias(sig,Delta):
    A,a,B,b,C,c = tinker_bias_params(Delta)
    delc = 1.686
    nu = old_div(delc, sig)
    ans = 1. - A*nu**a / (nu**a + delc**a) + B*nu**b + C*nu**c

    return ans

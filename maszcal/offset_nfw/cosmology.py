
import numpy
import astropy.units as u

def sigma_crit_inverse(cosmo, z_lens, z_source):
    z_lens = numpy.atleast_1d(z_lens)
    z_source = numpy.atleast_1d(z_source)
    if False:
        if z_lens.shape != z_source.shape:
            if numpy.product(z_lens.shape)==1:
                z_lens = numpy.full(shape=z_source.shape, fill_value=z_lens[0])
            elif numpy.product(z_source.shape)==1:
                z_source = numpy.full(shape=z_lens.shape, fill_value=z_source[0])
        if z_lens.shape != z_source.shape:
            raise RuntimeError("Can only compute sigma crit inverse for z_lens and z_source arrays of the same size")
    zl, zs = numpy.broadcast_arrays(z_lens, z_source)
    sci = numpy.zeros(zl.shape)*u.m*u.m/u.kg
    mask = zl<zs
    if numpy.any(mask):
        sci[mask] = 9.33168061E-27*u.m/u.kg*((
            cosmo.angular_diameter_distance(zl[mask])*cosmo.angular_diameter_distance_z1z2(zl[mask], zs[mask]))
            /cosmo.angular_diameter_distance(zs[mask]))
    return sci

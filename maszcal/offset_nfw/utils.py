"""utils.py

Includes utility functions for OffsetNFW: at this time, decorators to ensure inputs to the
user-facing functions in OffsetNFW are broadcastable.  These aren't as thoroughly documented as
the user-facing functions in other files, but they should still at least be readable."""

import numpy
import astropy.units as u
from inspect import getargspec




def _form_iterables(*args):
    """ Make input arrays broadcastable in the way we want.  We can't just use meshgrid since we may
    have an arbitrary number of vectors of the same length that all need to be tiled."""
    original_args = args
    r = numpy.atleast_1d(args[0])
    args = args[1:]
    # Don't do anything if everything non-r is a scalar, or if r is a scalar
    is_iterable = [hasattr(a, '__iter__') and len(a)>1 for a in args]
    if sum(is_iterable)==0 or (len(r)==1 and not hasattr(original_args[0], "__iter__")):
        return [numpy.array(o) if hasattr(o,'__iter__') else o for o in original_args]
    # Check everything that isn't r is the same shape
    obj_shapes = []
    for arg, iter in zip(args, is_iterable):
        if iter:
            obj_shapes.append(numpy.array(arg).shape)
    if len(set(obj_shapes))>1:
        raise RuntimeError("All iterable non-r parameters must have same shape")
    # Add an extra axis to the non-r arguments so they're column vectors instead (or the analogue
    # for multi-dimensional arrays, lol).
    args = [numpy.atleast_1d(a) if hasattr(a, '__len__') else a for a in args]
    args = [a[:,numpy.newaxis] if isinstance(a, numpy.ndarray) else a for a in args]
    return (r,)+tuple(args)


def _form_iterables_multisource(nargs, *args):
    """ Make input arrays broadcastable in the way we want, in the case where we have source
    redshifts to contend with.  Argument nargs is the number of args passed to func; this is
    assumed to INCLUDE a self argument that won't get passed to this decorator."""
    original_args = args
    r = args[0]
    if len(args)==nargs-1:
        z_source = args[-2]
        args = args[1:-2]
    else:
        z_source = args[-1]
        args = args[1:-1]
    is_iterable = [hasattr(a, '__iter__') and len(a)>1 for a in args]
    r_is_iterable = hasattr(r, '__iter__')
    sz_is_iterable = hasattr(z_source, '__iter__')

    # If only max one of these arguments is iterable, no need to broadcast
    if r_is_iterable+sz_is_iterable+sum(is_iterable)<2:
        if len(original_args)==nargs-1:
            # Ensure consistency with the rest of this fn, which ignores z_source_pdf
            return [numpy.array(o) if hasattr(o,'__iter__') else o for o in original_args[:-1]]
        else:
            return [numpy.array(o) if hasattr(o,'__iter__') else o for o in original_args]
    # or if z_source is not iterable, we can use the function for the 2d broadcasting
    elif sum(is_iterable)==0 or not sz_is_iterable:
        if len(original_args)==nargs-1:
            return _form_iterables(*original_args[:-1])
        else:
            return _form_iterables(*original_args)
    # Check that there's only array shape in the non-r, non-z_source arrays
    obj_shapes = []
    for arg, iter in zip(args, is_iterable):
        if iter:
            obj_shapes.append(numpy.array(arg).shape)
    if len(set(obj_shapes))>1:
        raise RuntimeError("All iterable non-r parameters must have same shape")
    r = numpy.atleast_1d(r)
    if len(r)==1 and not hasattr(original_args[0], '__iter__'):
        # Okay. So r is a scalar, but not the other arguments. That means we need an extra axis
        # ONLY for the source_z array.
        return (r[0],)+tuple(args)+(z_source[:, numpy.newaxis],)
    # Everything's iterable (or at least one thing in the non-r, non-z_source arrays).
    # Make the args column-like and z_source hypercolumn-like.
    z_source = numpy.atleast_1d(z_source)
    args = [numpy.atleast_1d(a) if hasattr(a, '__len__') else a for a in args]
    args = [a[:,numpy.newaxis] if isinstance(a, numpy.ndarray) else a for a in args]
    z_source = z_source[:,numpy.newaxis,numpy.newaxis]
    return (r,)+tuple(args)+(z_source,)


def reshape(func):
    """This is a decorator to handle reforming input vectors into something broadcastable for easy
    multiplication or interpolation table calls.  Arrays can generally be arbitrary shapes.
    Pass the kwarg 'skip_reformat' to skip this process (mainly
    used for OffsetNFW object methods that reference other methods)."""
    def wrap_shapes(self, *args, **kwargs):
        skip_reformat = kwargs.pop('skip_reformat', False)
        if skip_reformat:
            return func(self, *args, **kwargs)
        new_args = _form_iterables(*args)
        return func(self, *new_args, **kwargs)
    return wrap_shapes


def reshape_multisource(func):
    """This is a decorator to handle reforming input vectors into something broadcastable for easy
    multiplication or interpolation table calls, in the case where we have a list of source
    redshifts (args[-2]) and potentially source redshift pdfs (args[-1], which may be None
    to return the full array).  Arrays can generally be arbitrary shapes, but source redshift
    pdfs must be one-dimensional.  Pass the kwarg 'skip_reformat' to skip this process (mainly
    used for OffsetNFW object methods that reference other methods)."""
    def wrap_shapes(self, *args, **kwargs):
        skip_reformat = kwargs.pop('skip_reformat', False)
        if skip_reformat:
            return func(self, *args, **kwargs)
        nargs = len(getargspec(func).args)
        new_args = _form_iterables_multisource(nargs, *args)
        result = func(self, *new_args, **kwargs)

        if (len(args)==nargs-1 and args[-1] is not None) or 'z_source_pdf' in kwargs:
            if 'z_source_pdf' in kwargs:
                zs = kwargs['z_source_pdf']
            else:
                zs = args[-1]
            if hasattr(zs, '__iter__'):
                return numpy.sum(zs[:, numpy.newaxis, numpy.newaxis]*result, axis=-1)
            else:
                return zs*result
        else:
            return result
    return wrap_shapes


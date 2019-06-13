import sys
import numpy as np

from scipy import linalg
from scipy._lib.six import callable, get_method_function, get_function_code
from scipy.special import xlogy
from scipy.spatial.distance import cdist, pdist, squareform
from maszcal.nothing import NoSavedRbf

__all__ = ['Rbf']





class Rbf(object):
    """
    Taken from scipy v1.3.0
    """
    # Available radial basis functions that can be selected as strings;
    # they all start with _h_ (self._init_function relies on that)
    def _h_multiquadric(self, r):
        return np.sqrt((1.0/self.epsilon*r)**2 + 1)

    def _h_inverse_multiquadric(self, r):
        return 1.0/np.sqrt((1.0/self.epsilon*r)**2 + 1)

    def _h_gaussian(self, r):
        return np.exp(-(1.0/self.epsilon*r)**2)

    def _h_linear(self, r):
        return r

    def _h_cubic(self, r):
        return r**3

    def _h_quintic(self, r):
        return r**5

    def _h_thin_plate(self, r):
        return xlogy(r**2, r)

    # Setup self._function and do smoke test on initial r
    def _init_function(self, r):
        if isinstance(self.function, str):
            self.function = self.function.lower()
            _mapped = {'inverse': 'inverse_multiquadric',
                       'inverse multiquadric': 'inverse_multiquadric',
                       'thin-plate': 'thin_plate'}
            if self.function in _mapped:
                self.function = _mapped[self.function]

            func_name = "_h_" + self.function
            if hasattr(self, func_name):
                self._function = getattr(self, func_name)
            else:
                functionlist = [x[3:] for x in dir(self)
                                if x.startswith('_h_')]
                raise ValueError("function must be a callable or one of " +
                                 ", ".join(functionlist))
            self._function = getattr(self, "_h_"+self.function)
        elif callable(self.function):
            allow_one = False
            if hasattr(self.function, 'func_code') or \
               hasattr(self.function, '__code__'):
                val = self.function
                allow_one = True
            elif hasattr(self.function, "im_func"):
                val = get_method_function(self.function)
            elif hasattr(self.function, "__call__"):
                val = get_method_function(self.function.__call__)
            else:
                raise ValueError("Cannot determine number of arguments to "
                                 "function")

            argcount = get_function_code(val).co_argcount
            if allow_one and argcount == 1:
                self._function = self.function
            elif argcount == 2:
                if sys.version_info[0] >= 3:
                    self._function = self.function.__get__(self, Rbf)
                else:
                    import new
                    self._function = new.instancemethod(self.function, self,
                                                        Rbf)
            else:
                raise ValueError("Function argument must take 1 or 2 "
                                 "arguments.")

        a0 = self._function(r)
        if a0.shape != r.shape:
            raise ValueError("Callable must take array and return array of "
                             "the same shape")
        return a0

    def __init__(self, *args, **kwargs):
        # `args` can be a variable number of arrays; we flatten them and store
        # them as a single 2-D array `xi` of shape (n_args-1, array_size),
        # plus a 1-D array `di` for the values.
        # All arrays must have the same number of elements
        saved_rbf = kwargs.pop('saved_rbf', NoSavedRbf())
        if not isinstance(saved_rbf, NoSavedRbf):
            self.norm = saved_rbf.norm
            self.function = saved_rbf.function
            self.di = saved_rbf.data
            self.xi = saved_rbf.coords
            self.N = self.xi.shape[-1]
            self.epsilon = saved_rbf.epsilon
            self.smooth = saved_rbf.smoothness
            self.nodes = saved_rbf.nodes

            func_name = "_h_" + self.function
            self._function = getattr(self, func_name)
            return


        self.xi = np.asarray([np.asarray(a, dtype=np.float_).flatten()
                              for a in args[:-1]])
        self.N = self.xi.shape[-1]
        self.di = np.asarray(args[-1]).flatten()

        if not all([x.size == self.di.size for x in self.xi]):
            raise ValueError("All arrays must be equal length.")

        self.norm = kwargs.pop('norm', 'euclidean')
        self.epsilon = kwargs.pop('epsilon', None)
        if self.epsilon is None:
            # default epsilon is the "the average distance between nodes" based
            # on a bounding hypercube
            ximax = np.amax(self.xi, axis=1)
            ximin = np.amin(self.xi, axis=1)
            edges = ximax - ximin
            edges = edges[np.nonzero(edges)]
            self.epsilon = np.power(np.prod(edges)/self.N, 1.0/edges.size)

        self.smooth = kwargs.pop('smooth', 0.0)
        self.function = kwargs.pop('function', 'multiquadric')

        # attach anything left in kwargs to self for use by any user-callable
        # function or to save on the object returned.
        for item, value in kwargs.items():
            setattr(self, item, value)

        self.nodes = linalg.solve(self.A, self.di)

    @property
    def A(self):
        # this only exists for backwards compatibility: self.A was available
        # and, at least technically, public.

        r = squareform(pdist(self.xi.T, self.norm))  # Pairwise norm
        return self._init_function(r) - np.eye(self.N)*self.smooth

    def _call_norm(self, x1, x2):
        return cdist(x1.T, x2.T, self.norm)

    def __call__(self, *args):
        args = [np.asarray(x) for x in args]
        if not all([x.shape == y.shape for x in args for y in args]):
            raise ValueError("Array lengths must be equal")

        shp = args[0].shape
        xa = np.asarray([a.flatten() for a in args], dtype=np.float_)
        r = self._call_norm(xa, self.xi)
        return np.dot(self._function(r), self.nodes).reshape(shp)

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np


ext_modules = []
ext_modules.append(
    Extension('cmaszcal.lensing', ['cmaszcal/lensing.pyx'], include_dirs=[np.get_include()])
)
ext_modules.append(
    Extension('cmaszcal.nfw', ['cmaszcal/nfw.pyx'], include_dirs=[np.get_include()])
)
ext_modules.append(
    Extension('cmaszcal.integrate', ['cmaszcal/integrate.pyx'], include_dirs=[np.get_include()])
)


setup(
    name='maszcal',
    version='0.1',
    py_modules=['maszcal'],
    ext_modules = cythonize(ext_modules, include_path=[np.get_include()], language_level=3.7),
    include_dirs=[np.get_include()],
)

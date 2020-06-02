from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np


extension = Extension('cmaszcal', ['cmaszcal/*.pyx'], include_dirs=[np.get_include()])


setup(
    name='maszcal',
    version='0.1',
    py_modules=['maszcal'],
    ext_modules = cythonize(extension, include_path=[np.get_include()]),
    include_dirs=[np.get_include()],
)

from setuptools import setup
from Cython.Build import cythonize
import numpy
import pythran

setup(
    name="DL_NM",
    ext_modules=cythonize(['*.pyx'], annotate=True),
    include_dirs=[numpy.get_include(), pythran.get_include()],
    extra_compile_args=["-O3"]
)

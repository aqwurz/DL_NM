from setuptools import setup
from Cython.Build import cythonize
import numpy
import pythran

setup(
    name="DL-NM",
    ext_modules=cythonize('utils.pyx', annotate=True),
    include_dirs=[numpy.get_include(), pythran.get_include()],
    extra_compile_args=["-O3"]
)

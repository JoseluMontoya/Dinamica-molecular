import numpy
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

nameFile = "verletCy"

ext_modules=[
    Extension(nameFile,
              [nameFile+".pyx"],
              libraries=["m"],
              extra_compile_args = ["-O3", "-ffast-math"],
              include_dirs=[numpy.get_include()],
              ) 
]

setup( 
  name = nameFile,
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules,
)

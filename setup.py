from distutils.core import Extension
from Cython.Distutils import build_ext
from numpy.distutils.core import setup

import numpy

pyx_ext = Extension('statlib.ffbs', ['statlib/src/ffbs.pyx'],
                    include_dirs=[numpy.get_include()])

setup(name='statlib.ffbs',
      description='FFBS implemented in Cython',
      ext_modules=[pyx_ext],
      cmdclass = {
          'build_ext' : build_ext
      })

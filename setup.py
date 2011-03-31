#!/usr/bin/env python

# from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration
from numpy.distutils.system_info import get_info

import numpy as np

def configuration(parent_package='', top_path=None):
    blas_info = get_info('blas_opt', 0)

    config = Configuration(None, parent_package, top_path)

    ext_kwds = {'include_dirs' : [np.get_include(), 'statlib/src'],
                'extra_info' : blas_info}

    config.add_extension('tokyo', sources=['statlib/src/tokyo.pyx'],
                         **ext_kwds)
    config.add_extension('statlib.ffbs', sources=['statlib/src/ffbs.pyx'],
                         **ext_kwds)
    config.add_extension('statlib.filter', sources=['statlib/src/filter.pyx'],
                         **ext_kwds)

    return config

setup(
    name='Statlib extensions',
    cmdclass={'build_ext': build_ext},
    configuration=configuration,
)

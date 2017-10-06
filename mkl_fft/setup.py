from __future__ import division, print_function, absolute_import
import sys
from os.path import join, exists, abspath, dirname
from os import getcwd

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info
    config = Configuration('mkl_fft', parent_package, top_path)

    pdir = dirname(__file__)
    wdir = join(pdir, 'src')
    mkl_info = get_info('mkl')
    libs = mkl_info.get('libraries', ['mkl_rt'])

    try:
        from Cython.Build import cythonize
        sources = [join(pdir, '_pydfti.pyx')]
        have_cython = True
    except ImportError as e:
        have_cython = False
        sources = [join(pdir, '_pydfti.c')]
        if not exists(sources[0]):
            raise ValueError(str(e) + '. ' + 
                             'Cython is required to build the initial .c file.')

    config.add_extension(
        '_pydfti',
        sources = [
            join(wdir, 'mklfft.c.src'),
            join(wdir, 'multi_iter.c'),
        ] + sources,
        depends = [
            join(wdir, 'mklfft.h'),
            join(wdir, 'multi_iter.h')
        ],
        include_dirs = [wdir],
        libraries = libs,
        extra_compile_args = [
            '-DNDEBUG',
            # '-g', '-O0', '-Wall', '-Wextra',
        ]
    )

    config.add_data_dir('tests')

    if have_cython:
        config.ext_modules = cythonize(config.ext_modules, include_path=[pdir, wdir])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)

#!/usr/bin/env python
# Copyright (c) 2017, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
        name = '_pydfti',
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
            # '-g', '-O0', '-Wall', '-Wextra', '-DDEBUG',
        ]
    )

    config.add_data_dir('tests')

    if have_cython:
        config.ext_modules = cythonize(config.ext_modules, include_path=[pdir, wdir])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)

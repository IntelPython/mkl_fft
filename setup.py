#!/usr/bin/env python
# Copyright (c) 2017-2025, Intel Corporation
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

import io
import os
import re
from os.path import join
import Cython.Build
from setuptools import setup, Extension
import numpy as np
from _vendored.conv_template import process_file as process_c_file

with io.open('mkl_fft/_version.py', 'rt', encoding='utf8') as f:
    version = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)

with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()

VERSION = version

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved
Programming Language :: C
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3.10
Programming Language :: Python :: 3.11
Programming Language :: Python :: 3.12
Programming Language :: Python :: Implementation :: CPython
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

def extensions():
    mkl_root = os.environ.get('MKLROOT', None)
    if mkl_root:
        mkl_info = {
            'include_dirs': [join(mkl_root, 'include')],
            'library_dirs': [join(mkl_root, 'lib'), join(mkl_root, 'lib', 'intel64')],
            'libraries': ['mkl_rt']
        }
    else:
        try:
            mkl_info = get_info('mkl')
        except:
            mkl_info = dict()

    mkl_include_dirs = mkl_info.get('include_dirs', [])
    mkl_library_dirs = mkl_info.get('library_dirs', [])
    mkl_libraries = mkl_info.get('libraries', ['mkl_rt'])

    mklfft_templ = os.path.join("mkl_fft", "src", "mklfft.c.src")
    processed_mklfft_fn = os.path.join("mkl_fft", "src", "mklfft.c")
    src_processed = process_c_file(mklfft_templ)

    with open(processed_mklfft_fn, 'w') as fid:
        fid.write(src_processed)

    return [
        Extension(
            "mkl_fft._pydfti",
            [
                os.path.join("mkl_fft", "_pydfti.pyx"),
                os.path.join("mkl_fft", "src", "mklfft.c"),
            ],
            depends = [
                os.path.join("mkl_fft", "src", 'mklfft.h'),
                os.path.join("mkl_fft", "src", "multi_iter.h")
            ],
            include_dirs = [os.path.join("mkl_fft", "src"), np.get_include()] + mkl_include_dirs,
            libraries = mkl_libraries,
            library_dirs = mkl_library_dirs,
            extra_compile_args = [
                '-DNDEBUG',
                # '-ggdb', '-O0', '-Wall', '-Wextra', '-DDEBUG',
            ],
            define_macros=[("NPY_NO_DEPRECATED_API", None), ("PY_ARRAY_UNIQUE_SYMBOL", "mkl_fft_ext")]
        )
    ]


setup(
    name = "mkl_fft",
    maintainer = "Intel Corp.",
    maintainer_email = "scripting@intel.com",
    description = "MKL-based FFT transforms for NumPy arrays",
    version = version,
    cmdclass={'build_ext': Cython.Build.build_ext},
    packages=[
        "mkl_fft",
        "mkl_fft.interfaces",
    ],
    package_data={"mkl_fft": ["tests/*.py"]},
    include_package_data=True,
    ext_modules=extensions(),
    zip_safe=False,
    long_description = long_description,
    long_description_content_type="text/markdown",
    url = "http://github.com/IntelPython/mkl_fft",
    author = "Intel Corporation",
    download_url = "http://github.com/IntelPython/mkl_fft",
    license = "BSD",
    classifiers = [_f for _f in CLASSIFIERS.split('\n') if _f],
    platforms = ["Windows", "Linux", "Mac OS-X"],
    test_suite = "pytest",
    python_requires = '>=3.7',
    setup_requires=["Cython",],
    install_requires = ["numpy >=1.16", "mkl"],
    keywords=["DFTI", "FFT", "Fourier", "MKL",],
)

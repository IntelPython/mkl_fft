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

import sys
import os
from os.path import join
import Cython.Build
from setuptools import setup, Extension
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))  # Ensures local imports work
from _vendored.conv_template import process_file as process_c_file  # noqa: E402


def extensions():
    mkl_root = os.environ.get("MKLROOT", None)
    if mkl_root:
        mkl_info = {
            "include_dirs": [join(mkl_root, "include")],
            "library_dirs": [join(mkl_root, "lib"), join(mkl_root, "lib", "intel64")],
            "libraries": ["mkl_rt"],
        }
    else:
        raise ValueError("MKLROOT environment variable not set.")

    mkl_include_dirs = mkl_info.get("include_dirs", [])
    mkl_library_dirs = mkl_info.get("library_dirs", [])
    mkl_libraries = mkl_info.get("libraries", ["mkl_rt"])

    mklfft_templ = join("mkl_fft", "src", "mklfft.c.src")
    processed_mklfft_fn = join("mkl_fft", "src", "mklfft.c")
    src_processed = process_c_file(mklfft_templ)

    with open(processed_mklfft_fn, "w") as fid:
        fid.write(src_processed)

    return [
        Extension(
            "mkl_fft._pydfti",
            sources=[
                join("mkl_fft", "_pydfti.pyx"),
                join("mkl_fft", "src", "mklfft.c"),
            ],
            depends=[
                join("mkl_fft", "src", "mklfft.h"),
                join("mkl_fft", "src", "multi_iter.h"),
            ],
            include_dirs=[join("mkl_fft", "src"), np.get_include()] + mkl_include_dirs,
            libraries=mkl_libraries,
            library_dirs=mkl_library_dirs,
            extra_compile_args=[
                "-DNDEBUG",
                # '-ggdb', '-O0', '-Wall', '-Wextra', '-DDEBUG',
            ],
            define_macros=[
                ("NPY_NO_DEPRECATED_API", None),
                ("PY_ARRAY_UNIQUE_SYMBOL", "mkl_fft_ext"),
            ],
        )
    ]


setup(
    cmdclass={"build_ext": Cython.Build.build_ext},
    ext_modules=extensions(),
    zip_safe=False,
)

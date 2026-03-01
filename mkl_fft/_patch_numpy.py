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

"""Define functions for patching NumPy with MKL-based NumPy interface."""

from contextlib import ContextDecorator
from threading import local as threading_local

import numpy as np

import mkl_fft.interfaces.numpy_fft as _nfft

_tls = threading_local()


class _Patch:
    """Internal object for patching NumPy with mkl_fft interfaces."""

    _is_patched = False
    __patched_functions__ = _nfft.__all__
    _restore_dict = {}

    def _register_func(self, name, func):
        if name not in self.__patched_functions__:
            raise ValueError("%s not an mkl_fft function." % name)
        f = getattr(np.fft, name)
        self._restore_dict[name] = f
        setattr(np.fft, name, func)

    def _restore_func(self, name, verbose=False):
        if name not in self.__patched_functions__:
            raise ValueError("%s not an mkl_fft function." % name)
        try:
            val = self._restore_dict[name]
        except KeyError:
            if verbose:
                print("failed to restore")
            return
        else:
            if verbose:
                print("found and restoring...")
            setattr(np.fft, name, val)

    def restore(self, verbose=False):
        for name in self._restore_dict.keys():
            self._restore_func(name, verbose=verbose)
        self._is_patched = False

    def do_patch(self):
        for f in self.__patched_functions__:
            self._register_func(f, getattr(_nfft, f))
        self._is_patched = True

    def is_patched(self):
        return self._is_patched


def _initialize_tls():
    _tls.patch = _Patch()
    _tls.initialized = True


def _is_tls_initialized():
    return (getattr(_tls, "initialized", None) is not None) and (
        _tls.initialized is True
    )


def patch_numpy_fft(verbose=False):
    if verbose:
        print("Now patching NumPy FFT submodule with mkl_fft NumPy interface.")
        print(
            "Please direct bug reports to https://github.com/IntelPython/mkl_fft"
        )
    if not _is_tls_initialized():
        _initialize_tls()
    _tls.patch.do_patch()


def restore_numpy_fft(verbose=False):
    if verbose:
        print("Now restoring original NumPy FFT submodule.")
    if not _is_tls_initialized():
        _initialize_tls()
    _tls.patch.restore(verbose=verbose)


def is_patched():
    if not _is_tls_initialized():
        _initialize_tls()
    return _tls.patch.is_patched()


class mkl_fft(ContextDecorator):
    def __enter__(self):
        patch_numpy_fft()
        return self

    def __exit__(self, *exc):
        restore_numpy_fft()
        return False

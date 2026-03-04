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
from threading import Lock, local

import numpy as np

import mkl_fft.interfaces.numpy_fft as _nfft


class _GlobalPatch:
    def __init__(self):
        self._lock = Lock()
        self._patch_count = 0
        self._restore_dict = {}
        # make _patched_functions a tuple (immutable)
        self._patched_functions = tuple(_nfft.__all__)
        self._tls = local()

    def _register_func(self, name, func):
        if name not in self._patched_functions:
            raise ValueError(f"{name} not an mkl_fft function.")
        if name not in self._restore_dict:
            self._restore_dict[name] = getattr(np.fft, name)
        setattr(np.fft, name, func)

    def _restore_func(self, name, verbose=False):
        if name not in self._patched_functions:
            raise ValueError(f"{name} not an mkl_fft function.")
        try:
            val = self._restore_dict[name]
        except KeyError:
            if verbose:
                print(f"failed to restore {name}")
            return
        else:
            if verbose:
                print(f"found and restoring {name}...")
            setattr(np.fft, name, val)

    def do_patch(self, verbose=False):
        with self._lock:
            local_count = getattr(self._tls, "local_count", 0)
            if self._patch_count == 0:
                if verbose:
                    print(
                        "Now patching NumPy FFT submodule with mkl_fft NumPy "
                        "interface."
                    )
                    print(
                        "Please direct bug reports to "
                        "https://github.com/IntelPython/mkl_fft"
                    )
                for f in self._patched_functions:
                    self._register_func(f, getattr(_nfft, f))
            self._patch_count += 1
            self._tls.local_count = local_count + 1

    def do_restore(self, verbose=False):
        with self._lock:
            local_count = getattr(self._tls, "local_count", 0)
            if local_count <= 0:
                if verbose:
                    print(
                        "Warning: restore_numpy_fft called more times than "
                        "patch_numpy_fft in this thread."
                    )
                return
            self._tls.local_count -= 1
            self._patch_count -= 1
            if self._patch_count == 0:
                if verbose:
                    print("Now restoring original NumPy FFT submodule.")
                for name in tuple(self._restore_dict):
                    self._restore_func(name, verbose=verbose)
                self._restore_dict.clear()

    def is_patched(self):
        with self._lock:
            return self._patch_count > 0


_patch = _GlobalPatch()


def patch_numpy_fft(verbose=False):
    """
    Patch NumPy's fft submodule with mkl_fft's numpy_interface.

    Parameters
    ----------
    verbose : bool, optional
        print message when starting the patching process.

    Notes
    -----
    This function uses reference-counted semantics. Each call increments a
    global patch counter. Restoration requires a matching number of calls
    between `patch_numpy_fft` and `restore_numpy_fft`.

    In multi-threaded programs, prefer the `mkl_fft` context manager.

    """
    _patch.do_patch(verbose=verbose)


def restore_numpy_fft(verbose=False):
    """
    Restore NumPy's fft submodule to its original implementations.

    Parameters
    ----------
    verbose : bool, optional
        print message when starting restoration process.

    Notes
    -----
    This function uses reference-counted semantics. Each call decrements a
    global patch counter. Restoration requires a matching number of calls
    between `patch_numpy_fft` and `restore_numpy_fft`.

    In multi-threaded programs, prefer the `mkl_fft` context manager.

    """
    _patch.do_restore(verbose=verbose)


def is_patched():
    """Return True if NumPy's fft submodule is currently patched by mkl_fft."""
    return _patch.is_patched()


class mkl_fft(ContextDecorator):
    """
    Context manager and decorator to temporarily patch NumPy fft submodule
    with MKL-based implementations.

    Examples
    --------
    >>> import mkl_fft
    >>> mkl_fft.is_patched()
    # False

    >>> with mkl_fft.mkl_fft():  # Enable mkl_fft in Numpy
    >>>     print(mkl_fft.is_patched())
    # True

    >>> mkl_fft.is_patched()
    # False

    """

    def __enter__(self):
        patch_numpy_fft()
        return self

    def __exit__(self, *exc):
        restore_numpy_fft()
        return False

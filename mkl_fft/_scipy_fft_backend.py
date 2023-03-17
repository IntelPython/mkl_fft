#!/usr/bin/env python
# Copyright (c) 2019-2023, Intel Corporation
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

from . import _pydfti
from . import _float_utils
import mkl

from numpy.core import (take, sqrt, prod)
import contextvars
import contextlib
import operator
import os


__doc__ = """
This module implements interfaces mimicing `scipy.fft` module.

It also provides DftiBackend class which can be used to set mkl_fft to be used
via `scipy.fft` namespace.

:Example:
    import scipy.fft
    import mkl_fft._scipy_fft_backend as be
    # Set mkl_fft to be used as backend of SciPy's FFT functions.
    scipy.fft.set_global_backend(be)
"""

class _cpu_max_threads_count:
    def __init__(self):
        self.cpu_count = None
        self.max_threads_count = None

    def get_cpu_count(self):
        if self.cpu_count is None:
            max_threads = self.get_max_threads_count()
            self.cpu_count = max_threads
        return self.cpu_count

    def get_max_threads_count(self):
        if self.max_threads_count is None:
            self.max_threads_count = mkl.get_max_threads()

        return self.max_threads_count


class _workers_data:
    def __init__(self, workers=None):
        if workers:
            self.workers_ = workers
        else:
            self.workers_ = _cpu_max_threads_count().get_cpu_count()
        self.workers_ = operator.index(self.workers_)

    @property
    def workers(self):
        return self.workers_

    @workers.setter
    def workers(self, workers_val):
        self.workerks_ = operator.index(workers_val)


_workers_global_settings = contextvars.ContextVar('scipy_backend_workers', default=_workers_data())


def get_workers():
    "Gets the number of workers used by mkl_fft by default"
    return _workers_global_settings.get().workers


@contextlib.contextmanager
def set_workers(n_workers):
    "Set the value of workers used by default, returns the previous value"
    nw = operator.index(n_workers)
    token = None
    try:
        new_wd = _workers_data(nw)
        token = _workers_global_settings.set(new_wd)
        yield
    finally:
        if token:
            _workers_global_settings.reset(token)
        else:
            raise ValueError


__all__ = ['fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn',
           'rfft', 'irfft', 'rfft2', 'irfft2', 'rfftn', 'irfftn',
           'get_workers', 'set_workers', 'DftiBackend']

__ua_domain__ = "numpy.scipy.fft"

def __ua_function__(method, args, kwargs):
    """Fetch registered UA function."""
    fn = globals().get(method.__name__, None)
    if fn is None:
        return NotImplemented
    return fn(*args, **kwargs)


class DftiBackend:
    __ua_domain__ = "numpy.scipy.fft"
    @staticmethod
    def __ua_function__(method, args, kwargs):
        """Fetch registered UA function."""
        fn = globals().get(method.__name__, None)
        if fn is None:
            return NotImplemented
        return fn(*args, **kwargs)


def _cook_nd_args(a, s=None, axes=None, invreal=0):
    if s is None:
        shapeless = 1
        if axes is None:
            s = list(a.shape)
        else:
            s = take(a.shape, axes)
    else:
        shapeless = 0
    s = list(s)
    if axes is None:
        axes = list(range(-len(s), 0))
    if len(s) != len(axes):
        raise ValueError("Shape and axes have different lengths.")
    if invreal and shapeless:
        s[-1] = (a.shape[axes[-1]] - 1) * 2
    return s, axes


def _workers_to_num_threads(w):
    """Handle conversion of workers to a positive number of threads in the
    same way as scipy.fft.helpers._workers.
    """
    if w is None:
        return _workers_global_settings.get().workers
    _w = operator.index(w)
    if (_w == 0):
        raise ValueError("Number of workers must not be zero")
    if (_w < 0):
        ub = os.cpu_count()
        _w += ub + 1
        if _w <= 0:
            raise ValueError("workers value out of range; got {}, must not be"
                             " less than {}".format(w, -ub))
    return _w


class Workers:
    def __init__(self, workers):
        self.workers = workers
        self.n_threads = _workers_to_num_threads(workers)

    def __enter__(self):
        try:
            self.prev_num_threads = mkl.set_num_threads_local(self.n_threads)
        except:
            raise ValueError("Class argument {} result in invalid number of threads {}".format(self.workers, self.n_threads))

    def __exit__(self, *args):
        # restore old value
        mkl.set_num_threads_local(self.prev_num_threads)


def _check_norm(norm):
    if norm not in (None, "ortho", "forward", "backward"):
        raise ValueError(
            ("Invalid norm value {} should be None, "
             "\"ortho\", \"forward\", or \"backward\".").format(norm))

def _check_plan(plan):
    if plan is None:
        return
    raise NotImplementedError(
        f"Passing a precomputed plan with value={plan} is currently not supported"
    )


def _frwd_sc_1d(n, s):
    nn = n if n else s
    return 1/nn if nn != 0 else 1


def _frwd_sc_nd(s, axes, x_shape):
    ss = s if s is not None else x_shape
    if axes is not None:
        nn = prod([ss[ai] for ai in axes])
    else:
        nn = prod(ss)
    return 1/nn if nn != 0 else 1


def _ortho_sc_1d(n, s):
    return sqrt(_frwd_sc_1d(n, s))


def _compute_1d_forward_scale(norm, n, s):
    if norm in (None, "backward"):
        fsc = 1.0
    elif norm == "forward":
        fsc = _frwd_sc_1d(n, s)
    elif norm == "ortho":
        fsc = _ortho_sc_1d(n, s)
    else:
        _check_norm(norm)
    return fsc


def _compute_nd_forward_scale(norm, s, axes, x_shape):
    if norm in (None, "backward"):
        fsc = 1.0
    elif norm == "forward":
        fsc = _frwd_sc_nd(s, axes, x_shape)
    elif norm == "ortho":
        fsc = sqrt(_frwd_sc_nd(s, axes, x_shape))
    else:
        _check_norm(norm)
    return fsc


def fft(a, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, plan=None):
    try:
        x = _float_utils.__supported_array_or_not_implemented(a)
    except ValueError:
        return NotImplemented
    if x is NotImplemented:
        return x
    fsc = _compute_1d_forward_scale(norm, n, x.shape[axis])
    _check_plan(plan)
    with Workers(workers):
        output = _pydfti.fft(x, n=n, axis=axis, overwrite_x=overwrite_x, forward_scale=fsc)
    return output


def ifft(a, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, plan=None):
    try:
        x = _float_utils.__supported_array_or_not_implemented(a)
    except ValueError:
        return NotImplemented
    if x is NotImplemented:
        return x
    fsc = _compute_1d_forward_scale(norm, n, x.shape[axis])
    _check_plan(plan)
    with Workers(workers):
        output = _pydfti.ifft(x, n=n, axis=axis, overwrite_x=overwrite_x, forward_scale=fsc)
    return output


def fft2(a, s=None, axes=(-2,-1), norm=None, overwrite_x=False, workers=None, plan=None):
    try:
        x = _float_utils.__supported_array_or_not_implemented(a)
    except ValueError:
        return NotImplemented
    if x is NotImplemented:
        return x
    fsc = _compute_nd_forward_scale(norm, s, axes, x.shape)
    _check_plan(plan)
    with Workers(workers):
        output = _pydfti.fftn(x, shape=s, axes=axes, overwrite_x=overwrite_x, forward_scale=fsc)
    return output


def ifft2(a, s=None, axes=(-2,-1), norm=None, overwrite_x=False, workers=None, plan=None):
    try:
        x = _float_utils.__supported_array_or_not_implemented(a)
    except ValueError:
        return NotImplemented
    if x is NotImplemented:
        return x
    fsc = _compute_nd_forward_scale(norm, s, axes, x.shape)
    _check_plan(plan)
    with Workers(workers):
        output = _pydfti.ifftn(x, shape=s, axes=axes, overwrite_x=overwrite_x, forward_scale=fsc)
    return output


def fftn(a, s=None, axes=None, norm=None, overwrite_x=False, workers=None, plan=None):
    try:
        x = _float_utils.__supported_array_or_not_implemented(a)
    except ValueError:
        return NotImplemented
    if x is NotImplemented:
        return x
    fsc = _compute_nd_forward_scale(norm, s, axes, x.shape)
    _check_plan(plan)
    with Workers(workers):
        output = _pydfti.fftn(x, shape=s, axes=axes, overwrite_x=overwrite_x, forward_scale=fsc)
    return output


def ifftn(a, s=None, axes=None, norm=None, overwrite_x=False, workers=None, plan=None):
    try:
        x = _float_utils.__supported_array_or_not_implemented(a)
    except ValueError:
        return NotImplemented
    if x is NotImplemented:
        return x
    fsc = _compute_nd_forward_scale(norm, s, axes, x.shape)
    _check_plan(plan)
    with Workers(workers):
        output = _pydfti.ifftn(x, shape=s, axes=axes, overwrite_x=overwrite_x, forward_scale=fsc)
    return output


def rfft(a, n=None, axis=-1, norm=None, workers=None, plan=None):
    try:
        x = _float_utils.__supported_array_or_not_implemented(a)
    except ValueError:
        return NotImplemented
    if x is NotImplemented:
        return x
    fsc = _compute_1d_forward_scale(norm, n, x.shape[axis])
    _check_plan(plan)
    with Workers(workers):
        output = _pydfti.rfft_numpy(x, n=n, axis=axis, forward_scale=fsc)
    return output


def irfft(a, n=None, axis=-1, norm=None, workers=None, plan=None):
    try:
        x = _float_utils.__supported_array_or_not_implemented(a)
    except ValueError:
        return NotImplemented
    if x is NotImplemented:
        return x
    nn = n if n else 2*(x.shape[axis]-1)
    fsc = _compute_1d_forward_scale(norm, nn, x.shape[axis])
    _check_plan(plan)
    with Workers(workers):
        output = _pydfti.irfft_numpy(x, n=n, axis=axis, forward_scale=fsc)
    return output


def _compute_nd_forward_scale_for_rfft(norm, s, axes, x, invreal=False):
    if norm in (None, "backward"):
        fsc = 1.0
    elif norm == "forward":
        s, axes = _cook_nd_args(x, s, axes, invreal=invreal)
        fsc = _frwd_sc_nd(s, axes, x.shape)
    elif norm == "ortho":
        s, axes = _cook_nd_args(x, s, axes, invreal=invreal)
        fsc = sqrt(_frwd_sc_nd(s, axes, x.shape))
    else:
        _check_norm(norm)
    return s, axes, fsc


def rfft2(a, s=None, axes=(-2, -1), norm=None, workers=None, plan=None):
    try:
        x = _float_utils.__supported_array_or_not_implemented(a)
    except ValueError:
        return NotImplemented
    if x is NotImplemented:
        return x
    s, axes, fsc = _compute_nd_forward_scale_for_rfft(norm, s, axes, x)
    _check_plan(plan)
    with Workers(workers):
        output = _pydfti.rfftn_numpy(x, s, axes, forward_scale=fsc)
    return output


def irfft2(a, s=None, axes=(-2, -1), norm=None, workers=None, plan=None):
    try:
        x = _float_utils.__supported_array_or_not_implemented(a)
    except ValueError:
        return NotImplemented
    if x is NotImplemented:
        return x
    s, axes, fsc = _compute_nd_forward_scale_for_rfft(norm, s, axes, x, invreal=True)
    _check_plan(plan)
    with Workers(workers):
        output = _pydfti.irfftn_numpy(x, s, axes, forward_scale=fsc)
    return output


def rfftn(a, s=None, axes=None, norm=None, workers=None, plan=None):
    try:
        x = _float_utils.__supported_array_or_not_implemented(a)
    except ValueError:
        return NotImplemented
    if x is NotImplemented:
        return x
    s, axes, fsc = _compute_nd_forward_scale_for_rfft(norm, s, axes, x)
    _check_plan(plan)
    with Workers(workers):
        output = _pydfti.rfftn_numpy(x, s, axes, forward_scale=fsc)
    return output


def irfftn(a, s=None, axes=None, norm=None, workers=None, plan=None):
    try:
        x = _float_utils.__supported_array_or_not_implemented(a)
    except ValueError:
        return NotImplemented
    if x is NotImplemented:
        return x
    s, axes, fsc = _compute_nd_forward_scale_for_rfft(norm, s, axes, x, invreal=True)
    _check_plan(plan)
    with Workers(workers):
        output = _pydfti.irfftn_numpy(x, s, axes, forward_scale=fsc)
    return output

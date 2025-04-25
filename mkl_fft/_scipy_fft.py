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

"""
An interface for FFT module of Scipy (`scipy.fft`) that uses OneMKL FFT
in the backend.

"""

import contextlib
import contextvars
import operator
import os

import mkl
import numpy as np

from . import _pydfti as mkl_fft  # pylint: disable=no-name-in-module
from ._fft_utils import _compute_fwd_scale, _swap_direction
from ._float_utils import _supported_array_or_not_implemented

__doc__ = """
This module implements interfaces mimicking `scipy.fft` module.

It also provides DftiBackend class which can be used to set mkl_fft to be used
via `scipy.fft` namespace.

:Example:
    import scipy.fft
    import mkl_fft._scipy_fft as be
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
            # pylint: disable=no-member
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


_workers_global_settings = contextvars.ContextVar(
    "scipy_backend_workers", default=_workers_data()
)


def get_workers():
    """Gets the number of workers used by mkl_fft by default"""
    return _workers_global_settings.get().workers


@contextlib.contextmanager
def set_workers(n_workers):
    """Set the value of workers used by default, returns the previous value"""
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


__all__ = [
    "fft",
    "ifft",
    "fft2",
    "ifft2",
    "fftn",
    "ifftn",
    "rfft",
    "irfft",
    "rfft2",
    "irfft2",
    "rfftn",
    "irfftn",
    "hfft",
    "ihfft",
    "hfft2",
    "ihfft2",
    "hfftn",
    "ihfftn",
    "get_workers",
    "set_workers",
    "DftiBackend",
]

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


def _workers_to_num_threads(w):
    """Handle conversion of workers to a positive number of threads in the
    same way as scipy.fft.helpers._workers.
    """
    if w is None:
        return _workers_global_settings.get().workers
    _w = operator.index(w)
    if _w == 0:
        raise ValueError("Number of workers must not be zero")
    if _w < 0:
        ub = os.cpu_count()
        _w += ub + 1
        if _w <= 0:
            raise ValueError(
                "workers value out of range; got {}, must not be"
                " less than {}".format(w, -ub)
            )
    return _w


class Workers:
    def __init__(self, workers):
        self.workers = workers
        self.n_threads = _workers_to_num_threads(workers)

    def __enter__(self):
        try:
            # pylint: disable=no-member
            self.prev_num_threads = mkl.set_num_threads_local(self.n_threads)
        except Exception as e:
            raise ValueError(
                "Class argument {} result in invalid number of threads {}".format(
                    self.workers, self.n_threads
                )
            ) from e

    def __exit__(self, *args):
        # restore old value
        # pylint: disable=no-member
        mkl.set_num_threads_local(self.prev_num_threads)


def _check_plan(plan):
    if plan is not None:
        raise NotImplementedError(
            f"Passing a precomputed plan with value={plan} is currently not supported"
        )


def _check_overwrite_x(overwrite_x):
    if overwrite_x:
        raise NotImplementedError(
            "Overwriting the content of `x` is currently not supported"
        )


def _cook_nd_args(x, s=None, axes=None, invreal=False):
    if s is None:
        shapeless = True
        if axes is None:
            s = list(x.shape)
        else:
            s = np.take(x.shape, axes)
    else:
        shapeless = False
    s = list(s)
    if axes is None:
        axes = list(range(-len(s), 0))
    if len(s) != len(axes):
        raise ValueError("Shape and axes have different lengths.")
    if invreal and shapeless:
        s[-1] = (x.shape[axes[-1]] - 1) * 2
    return s, axes


def _validate_input(x):
    try:
        x = _supported_array_or_not_implemented(x)
    except ValueError:
        raise NotImplementedError

    return x


def fft(
    x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *, plan=None
):
    """
    Compute the 1-D discrete Fourier Transform.

    For full documentation refer to `scipy.fft.fft`.

    """
    _check_plan(plan)
    x = _validate_input(x)
    fsc = _compute_fwd_scale(norm, n, x.shape[axis])

    with Workers(workers):
        return mkl_fft.fft(
            x, n=n, axis=axis, overwrite_x=overwrite_x, fwd_scale=fsc
        )


def ifft(
    x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *, plan=None
):
    """
    Compute the 1-D inverse discrete Fourier Transform.

    For full documentation refer to `scipy.fft.ifft`.

    """
    _check_plan(plan)
    x = _validate_input(x)
    fsc = _compute_fwd_scale(norm, n, x.shape[axis])

    with Workers(workers):
        return mkl_fft.ifft(
            x, n=n, axis=axis, overwrite_x=overwrite_x, fwd_scale=fsc
        )


def fft2(
    x,
    s=None,
    axes=(-2, -1),
    norm=None,
    overwrite_x=False,
    workers=None,
    *,
    plan=None,
):
    """
    Compute the 2-D discrete Fourier Transform.

    For full documentation refer to `scipy.fft.fft2`.

    """
    return fftn(
        x,
        s=s,
        axes=axes,
        norm=norm,
        overwrite_x=overwrite_x,
        workers=workers,
        plan=plan,
    )


def ifft2(
    x,
    s=None,
    axes=(-2, -1),
    norm=None,
    overwrite_x=False,
    workers=None,
    *,
    plan=None,
):
    """
    Compute the 2-D inverse discrete Fourier Transform.

    For full documentation refer to `scipy.fft.ifft2`.

    """
    return ifftn(
        x,
        s=s,
        axes=axes,
        norm=norm,
        overwrite_x=overwrite_x,
        workers=workers,
        plan=plan,
    )


def fftn(
    x,
    s=None,
    axes=None,
    norm=None,
    overwrite_x=False,
    workers=None,
    *,
    plan=None,
):
    """
    Compute the N-D discrete Fourier Transform.

    For full documentation refer to `scipy.fft.fftn`.

    """
    _check_plan(plan)
    x = _validate_input(x)
    fsc = _compute_fwd_scale(norm, s, x.shape)

    with Workers(workers):
        return mkl_fft.fftn(
            x, s=s, axes=axes, overwrite_x=overwrite_x, fwd_scale=fsc
        )


def ifftn(
    x,
    s=None,
    axes=None,
    norm=None,
    overwrite_x=False,
    workers=None,
    *,
    plan=None,
):
    """
    Compute the N-D inverse discrete Fourier Transform.

    For full documentation refer to `scipy.fft.ifftn`.

    """
    _check_plan(plan)
    x = _validate_input(x)
    fsc = _compute_fwd_scale(norm, s, x.shape)

    with Workers(workers):
        return mkl_fft.ifftn(
            x, s=s, axes=axes, overwrite_x=overwrite_x, fwd_scale=fsc
        )


def rfft(
    x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *, plan=None
):
    """
    Compute the 1-D discrete Fourier Transform for real input..

    For full documentation refer to `scipy.fft.rfft`.

    Limitation
    -----------
    The kwarg `overwrite_x` is only supported with its default value.

    """
    _check_plan(plan)
    _check_overwrite_x(overwrite_x)
    x = _validate_input(x)
    fsc = _compute_fwd_scale(norm, n, x.shape[axis])

    with Workers(workers):
        return mkl_fft.rfft(x, n=n, axis=axis, fwd_scale=fsc)


def irfft(
    x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *, plan=None
):
    """
    Compute the inverse of `rfft`.

    For full documentation refer to `scipy.fft.irfft`.

    Limitation
    -----------
    The kwarg `overwrite_x` is only supported with its default value.

    """
    _check_plan(plan)
    _check_overwrite_x(overwrite_x)
    x = _validate_input(x)
    fsc = _compute_fwd_scale(norm, n, 2 * (x.shape[axis] - 1))

    with Workers(workers):
        return mkl_fft.irfft(x, n=n, axis=axis, fwd_scale=fsc)


def rfft2(
    x,
    s=None,
    axes=(-2, -1),
    overwrite_x=False,
    norm=None,
    workers=None,
    *,
    plan=None,
):
    """
    Compute the 2-D discrete Fourier Transform for real input.

    For full documentation refer to `scipy.fft.rfft2`.

    Limitation
    -----------
    The kwarg `overwrite_x` is only supported with its default value.

    """
    return rfftn(
        x,
        s=s,
        axes=axes,
        norm=norm,
        overwrite_x=overwrite_x,
        workers=workers,
        plan=plan,
    )


def irfft2(
    x,
    s=None,
    axes=(-2, -1),
    norm=None,
    overwrite_x=False,
    workers=None,
    *,
    plan=None,
):
    """
    Compute the inverse of `rfft2`.

    For full documentation refer to `scipy.fft.irfft2`.

    Limitation
    -----------
    The kwarg `overwrite_x` is only supported with its default value.

    """
    return irfftn(
        x,
        s=s,
        axes=axes,
        norm=norm,
        overwrite_x=overwrite_x,
        workers=workers,
        plan=plan,
    )


def rfftn(
    x,
    s=None,
    axes=None,
    norm=None,
    overwrite_x=False,
    workers=None,
    *,
    plan=None,
):
    """
    Compute the N-D discrete Fourier Transform for real input.

    For full documentation refer to `scipy.fft.rfftn`.

    Limitation
    -----------
    The kwarg `overwrite_x` is only supported with its default value.

    """
    _check_plan(plan)
    _check_overwrite_x(overwrite_x)
    x = _validate_input(x)
    s, axes = _cook_nd_args(x, s, axes)
    fsc = _compute_fwd_scale(norm, s, x.shape)

    with Workers(workers):
        return mkl_fft.rfftn(x, s, axes, fwd_scale=fsc)


def irfftn(
    x,
    s=None,
    axes=None,
    norm=None,
    overwrite_x=False,
    workers=None,
    *,
    plan=None,
):
    """
    Compute the inverse of `rfftn`.

    For full documentation refer to `scipy.fft.irfftn`.

    Limitation
    -----------
    The kwarg `overwrite_x` is only supported with its default value.

    """
    _check_plan(plan)
    _check_overwrite_x(overwrite_x)
    x = _validate_input(x)
    s, axes = _cook_nd_args(x, s, axes, invreal=True)
    fsc = _compute_fwd_scale(norm, s, x.shape)

    with Workers(workers):
        return mkl_fft.irfftn(x, s, axes, fwd_scale=fsc)


def hfft(
    x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *, plan=None
):
    """
    Compute the FFT of a signal that has Hermitian symmetry,
    i.e., a real spectrum.

    For full documentation refer to `scipy.fft.hfft`.

    Limitation
    -----------
    The kwarg `overwrite_x` is only supported with its default value.

    """
    _check_plan(plan)
    _check_overwrite_x(overwrite_x)
    x = _validate_input(x)
    norm = _swap_direction(norm)
    x = np.array(x, copy=True)
    np.conjugate(x, out=x)
    fsc = _compute_fwd_scale(norm, n, 2 * (x.shape[axis] - 1))

    with Workers(workers):
        return mkl_fft.irfft(x, n=n, axis=axis, fwd_scale=fsc)


def ihfft(
    x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *, plan=None
):
    """
    Compute the inverse FFT of a signal that has Hermitian symmetry.

    For full documentation refer to `scipy.fft.ihfft`.

    Limitation
    -----------
    The kwarg `overwrite_x` is only supported with its default value.

    """
    _check_plan(plan)
    _check_overwrite_x(overwrite_x)
    x = _validate_input(x)
    norm = _swap_direction(norm)
    fsc = _compute_fwd_scale(norm, n, x.shape[axis])

    with Workers(workers):
        result = mkl_fft.rfft(x, n=n, axis=axis, fwd_scale=fsc)

    np.conjugate(result, out=result)
    return result


def hfft2(
    x,
    s=None,
    axes=(-2, -1),
    norm=None,
    overwrite_x=False,
    workers=None,
    *,
    plan=None,
):
    """
    Compute the 2-D FFT of a Hermitian complex array.

    For full documentation refer to `scipy.fft.hfft2`.

    Limitation
    -----------
    The kwarg `overwrite_x` is only supported with its default value.

    """
    return hfftn(
        x,
        s=s,
        axes=axes,
        norm=norm,
        overwrite_x=overwrite_x,
        workers=workers,
        plan=plan,
    )


def ihfft2(
    x,
    s=None,
    axes=(-2, -1),
    norm=None,
    overwrite_x=False,
    workers=None,
    *,
    plan=None,
):
    """
    Compute the 2-D inverse FFT of a real spectrum.

    For full documentation refer to `scipy.fft.ihfft2`.

    Limitation
    -----------
    The kwarg `overwrite_x` is only supported with its default value.

    """
    return ihfftn(
        x,
        s=s,
        axes=axes,
        norm=norm,
        overwrite_x=overwrite_x,
        workers=workers,
        plan=plan,
    )


def hfftn(
    x,
    s=None,
    axes=None,
    norm=None,
    overwrite_x=False,
    workers=None,
    *,
    plan=None,
):
    """
    Compute the N-D FFT of Hermitian symmetric complex input, i.e., a signal with a real spectrum.

    For full documentation refer to `scipy.fft.hfftn`.

    Limitation
    -----------
    The kwarg `overwrite_x` is only supported with its default value.

    """
    _check_plan(plan)
    _check_overwrite_x(overwrite_x)
    x = _validate_input(x)
    norm = _swap_direction(norm)
    x = np.array(x, copy=True)
    np.conjugate(x, out=x)
    s, axes = _cook_nd_args(x, s, axes, invreal=True)
    fsc = _compute_fwd_scale(norm, s, x.shape)

    with Workers(workers):
        return mkl_fft.irfftn(x, s, axes, fwd_scale=fsc)


def ihfftn(
    x,
    s=None,
    axes=None,
    norm=None,
    overwrite_x=False,
    workers=None,
    *,
    plan=None,
):
    """
    Compute the N-D inverse discrete Fourier Transform for a real spectrum.

    For full documentation refer to `scipy.fft.ihfftn`.

    Limitation
    -----------
    The kwarg `overwrite_x` is only supported with its default value.

    """
    _check_plan(plan)
    _check_overwrite_x(overwrite_x)
    x = _validate_input(x)
    norm = _swap_direction(norm)
    s, axes = _cook_nd_args(x, s, axes)
    fsc = _compute_fwd_scale(norm, s, x.shape)

    with Workers(workers):
        result = mkl_fft.rfftn(x, s, axes, fwd_scale=fsc)

    np.conjugate(result, out=result)
    return result

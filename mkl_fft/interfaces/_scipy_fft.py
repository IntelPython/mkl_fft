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
An interface for FFT module of SciPy (`scipy.fft`) that uses OneMKL FFT
in the backend.
"""

import contextlib
import contextvars
import operator
from numbers import Number

import mkl
import numpy as np

import mkl_fft

from .._fft_utils import _compute_fwd_scale, _swap_direction
from ._float_utils import _supported_array_or_not_implemented

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
]


class _workers_data:
    def __init__(self, workers=None):
        if workers is not None:  # workers = 0 should be handled
            self.workers_ = _workers_to_num_threads(workers)
        else:
            # Unlike SciPy, the default value is maximum number of threads
            self.workers_ = mkl.get_max_threads()  # pylint: disable=no-member
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


def _workers_to_num_threads(w):
    """
    Handle conversion of workers to a positive number of threads in the
    same way as scipy.fft._pocketfft.helpers._workers.
    """
    if w is None:
        return _workers_global_settings.get().workers
    _w = operator.index(w)
    if _w == 0:
        raise ValueError("Number of workers must not be zero")
    if _w < 0:
        # SciPy uses os.cpu_count()
        _cpu_count = mkl.get_max_threads()  # pylint: disable=no-member
        _w += _cpu_count + 1
        if _w <= 0:
            raise ValueError(
                f"workers value out of range; got {w}, must not be less "
                f"than {-_cpu_count}"
            )
    return _w


class _Workers:
    def __init__(self, workers):
        self.workers = workers
        self.n_threads = _workers_to_num_threads(workers)

    def __enter__(self):
        try:
            # mkl.set_num_threads_local sets the number of threads to the
            # given input number, and returns the previous number of threads
            # pylint: disable=no-member
            self.prev_num_threads = mkl.set_num_threads_local(self.n_threads)
        except Exception as e:
            raise ValueError(
                f"Class argument {self.workers} results in invalid number of "
                f"threads {self.n_threads}"
            ) from e
        return self

    def __exit__(self, *args):
        # restore old value
        # pylint: disable=no-member
        mkl.set_num_threads_local(self.prev_num_threads)


def _check_plan(plan):
    if plan is not None:
        raise NotImplementedError(
            f"Passing a precomputed plan with value={plan} is currently not supported"
        )


# copied from scipy.fft._pocketfft.helper
# https://github.com/scipy/scipy/blob/main/scipy/fft/_pocketfft/helper.py
def _iterable_of_int(x, name=None):
    if isinstance(x, Number):
        x = (x,)

    try:
        x = [operator.index(a) for a in x]
    except TypeError as e:
        name = name or "value"
        raise ValueError(
            f"{name} must be a scalar or iterable of integers"
        ) from e

    return x


# copied and modified from scipy.fft._pocketfft.helper
# https://github.com/scipy/scipy/blob/main/scipy/fft/_pocketfft/helper.py
def _init_nd_shape_and_axes(x, shape, axes, invreal=False):
    noshape = shape is None
    noaxes = axes is None

    if not noaxes:
        axes = _iterable_of_int(axes, "axes")
        axes = [a + x.ndim if a < 0 else a for a in axes]

        if any(a >= x.ndim or a < 0 for a in axes):
            raise ValueError("axes exceeds dimensionality of input")
        if len(set(axes)) != len(axes):
            raise ValueError("all axes must be unique")

    if not noshape:
        shape = _iterable_of_int(shape, "shape")

        if axes and len(axes) != len(shape):
            raise ValueError(
                "when given, axes and shape arguments"
                " have to be of the same length"
            )
        if noaxes:
            if len(shape) > x.ndim:
                raise ValueError("shape requires more axes than are present")
            axes = range(x.ndim - len(shape), x.ndim)

        shape = [x.shape[a] if s == -1 else s for s, a in zip(shape, axes)]
    elif noaxes:
        shape = list(x.shape)
        axes = range(x.ndim)
    else:
        shape = [x.shape[a] for a in axes]

    if noshape and invreal:
        shape[-1] = (x.shape[axes[-1]] - 1) * 2

    if any(s < 1 for s in shape):
        raise ValueError(f"invalid number of data points ({shape}) specified")

    return tuple(shape), list(axes)


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

    with _Workers(workers):
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

    with _Workers(workers):
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
    s, axes = _init_nd_shape_and_axes(x, s, axes)
    fsc = _compute_fwd_scale(norm, s, x.shape)

    with _Workers(workers):
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
    s, axes = _init_nd_shape_and_axes(x, s, axes)
    fsc = _compute_fwd_scale(norm, s, x.shape)

    with _Workers(workers):
        return mkl_fft.ifftn(
            x, s=s, axes=axes, overwrite_x=overwrite_x, fwd_scale=fsc
        )


def rfft(
    x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *, plan=None
):
    """
    Compute the 1-D discrete Fourier Transform for real input..

    For full documentation refer to `scipy.fft.rfft`.

    """
    _check_plan(plan)
    x = _validate_input(x)
    fsc = _compute_fwd_scale(norm, n, x.shape[axis])

    with _Workers(workers):
        # Note: overwrite_x is not utilized
        return mkl_fft.rfft(x, n=n, axis=axis, fwd_scale=fsc)


def irfft(
    x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *, plan=None
):
    """
    Compute the inverse of `rfft`.

    For full documentation refer to `scipy.fft.irfft`.

    """
    _check_plan(plan)
    x = _validate_input(x)
    fsc = _compute_fwd_scale(norm, n, 2 * (x.shape[axis] - 1))

    with _Workers(workers):
        # Note: overwrite_x is not utilized
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

    """
    _check_plan(plan)
    x = _validate_input(x)
    s, axes = _init_nd_shape_and_axes(x, s, axes)
    fsc = _compute_fwd_scale(norm, s, x.shape)

    with _Workers(workers):
        # Note: overwrite_x is not utilized
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

    """
    _check_plan(plan)
    x = _validate_input(x)
    s, axes = _init_nd_shape_and_axes(x, s, axes, invreal=True)
    fsc = _compute_fwd_scale(norm, s, x.shape)

    with _Workers(workers):
        # Note: overwrite_x is not utilized
        return mkl_fft.irfftn(x, s, axes, fwd_scale=fsc)


def hfft(
    x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *, plan=None
):
    """
    Compute the FFT of a signal that has Hermitian symmetry,
    i.e., a real spectrum.

    For full documentation refer to `scipy.fft.hfft`.

    """
    _check_plan(plan)
    x = _validate_input(x)
    norm = _swap_direction(norm)
    x = np.array(x, copy=True)
    np.conjugate(x, out=x)
    fsc = _compute_fwd_scale(norm, n, 2 * (x.shape[axis] - 1))

    with _Workers(workers):
        # Note: overwrite_x is not utilized
        return mkl_fft.irfft(x, n=n, axis=axis, fwd_scale=fsc)


def ihfft(
    x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *, plan=None
):
    """
    Compute the inverse FFT of a signal that has Hermitian symmetry.

    For full documentation refer to `scipy.fft.ihfft`.

    """
    _check_plan(plan)
    x = _validate_input(x)
    norm = _swap_direction(norm)
    fsc = _compute_fwd_scale(norm, n, x.shape[axis])

    with _Workers(workers):
        # Note: overwrite_x is not utilized
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
    Compute the N-D FFT of Hermitian symmetric complex input,
    i.e., a signal with a real spectrum.

    For full documentation refer to `scipy.fft.hfftn`.

    """
    _check_plan(plan)
    x = _validate_input(x)
    norm = _swap_direction(norm)
    x = np.array(x, copy=True)
    np.conjugate(x, out=x)
    s, axes = _init_nd_shape_and_axes(x, s, axes, invreal=True)
    fsc = _compute_fwd_scale(norm, s, x.shape)

    with _Workers(workers):
        # Note: overwrite_x is not utilized
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

    """
    _check_plan(plan)
    x = _validate_input(x)
    norm = _swap_direction(norm)
    s, axes = _init_nd_shape_and_axes(x, s, axes)
    fsc = _compute_fwd_scale(norm, s, x.shape)

    with _Workers(workers):
        # Note: overwrite_x is not utilized
        result = mkl_fft.rfftn(x, s, axes, fwd_scale=fsc)

    np.conjugate(result, out=result)
    return result


def get_workers():
    """
    Gets the number of workers used by mkl_fft by default.

    For full documentation refer to `scipy.fft.get_workers`.

    """
    return _workers_global_settings.get().workers


@contextlib.contextmanager
def set_workers(workers):
    """
    Set the value of workers used by default, returns the previous value.

    For full documentation refer to `scipy.fft.set_workers`.

    """
    nw = operator.index(workers)
    token = None
    try:
        new_wd = _workers_data(nw)
        token = _workers_global_settings.set(new_wd)
        yield
    finally:
        if token is not None:
            _workers_global_settings.reset(token)

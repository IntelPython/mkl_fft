#!/usr/bin/env python
# Copyright (c) 2025, Intel Corporation
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

import numpy as np

# pylint: disable=no-name-in-module
from ._pydfti import _direct_fftnd, _fft1d_impl

__all__ = [
    "_check_norm",
    "_compute_fwd_scale",
    "_cook_nd_args",
    "_trim_array",
    "_pad_array",
    "_remove_axis",
    "_fftnd_impl",
]


def _check_norm(norm):
    if norm not in (None, "ortho", "forward", "backward"):
        raise ValueError(
            f"Invalid norm value {norm} should be None, 'ortho', 'forward', "
            "or 'backward'."
        )


def _compute_fwd_scale(norm, n, shape):
    _check_norm(norm)
    if norm in (None, "backward"):
        return 1.0

    ss = n if n is not None else shape
    nn = np.prod(ss)
    fsc = 1 / nn if nn != 0 else 1
    if norm == "forward":
        return fsc
    else:  # norm == "ortho"
        return np.sqrt(fsc)


def _cook_nd_args(a, s=None, axes=None, invreal=0):
    if s is None:
        shapeless = 1
        if axes is None:
            s = list(a.shape)
        else:
            try:
                s = [a.shape[i] for i in axes]
            except IndexError:
                # fake s designed to trip the ValueError further down
                s = range(len(axes) + 1)
                pass
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


def _trim_array(arr, s, axes):
    """
    Forms a view into subarray of arr if any element of shape parameter s is
    smaller than the corresponding element of the shape of the input array arr,
    otherwise returns the input array.
    """

    arr_shape = arr.shape
    no_trim = True
    ind = [slice(None, None, None)] * len(arr_shape)
    for si, ai in zip(s, axes):
        try:
            shp_i = arr_shape[ai]
        except IndexError:
            raise ValueError("Invalid axis (%d) specified" % ai)
        if si < shp_i:
            no_trim = False
            ind[ai] = slice(None, si, None)
    if no_trim:
        return arr
    return arr[tuple(ind)]


def _pad_array(arr, s, axes):
    """Pads array arr with zeros to attain shape s associated with axes"""
    arr_shape = arr.shape
    no_padding = True
    pad_widths = [(0, 0)] * len(arr_shape)
    for si, ai in zip(s, axes):
        try:
            shp_i = arr_shape[ai]
        except IndexError:
            raise ValueError("Invalid axis (%d) specified" % ai)
        if si > shp_i:
            no_padding = False
            pad_widths[ai] = (0, si - shp_i)
    if no_padding:
        return arr
    return np.pad(arr, tuple(pad_widths), "constant")


def _remove_axis(s, axes, axis_to_remove):
    lens = len(s)
    axes_normalized = tuple(lens + ai if ai < 0 else ai for ai in axes)
    a2r = lens + axis_to_remove if axis_to_remove < 0 else axis_to_remove

    ss = s[:a2r] + s[a2r + 1 :]
    pivot = axes_normalized[a2r]
    aa = tuple(
        ai if ai < pivot else ai - 1 for ai in axes_normalized[:a2r]
    ) + tuple(ai if ai < pivot else ai - 1 for ai in axes_normalized[a2r + 1 :])
    return ss, aa


def _datacopied(arr: np.ndarray, orig: object) -> int:
    """
    Strict check for `arr` not sharing any data with `original`,
    under the assumption that arr = np.asarray(original).
    """
    if not isinstance(orig, np.ndarray) and hasattr(orig, "__array__"):
        return 0
    if isinstance(orig, np.ndarray) and arr is orig:
        return 0
    return 1 if arr.base is None else 0


# copied from scipy.fftpack.helper
def _init_nd_shape_and_axes(x, shape, axes):
    """Handle shape and axes arguments for n-dimensional transforms.
    Returns the shape and axes in a standard form, taking into account negative
    values and checking for various potential errors.
    Parameters
    ----------
    x : array_like
        The input array.
    shape : int or array_like of ints or None
        The shape of the result.  If both `shape` and `axes` (see below) are
        None, `shape` is ``x.shape``; if `shape` is None but `axes` is
        not None, then `shape` is ``scipy.take(x.shape, axes, axis=0)``.
        If `shape` is -1, the size of the corresponding dimension of `x` is
        used.
    axes : int or array_like of ints or None
        Axes along which the calculation is computed.
        The default is over all axes.
        Negative indices are automatically converted to their positive
        counterpart.
    Returns
    -------
    shape : array
        The shape of the result. It is a 1D integer array.
    axes : array
        The shape of the result. It is a 1D integer array.
    """
    x = np.asarray(x)
    noshape = shape is None
    noaxes = axes is None

    if noaxes:
        axes = np.arange(x.ndim, dtype=np.intc)
    else:
        axes = np.atleast_1d(axes)

    if axes.size == 0:
        axes = axes.astype(np.intc)

    if not axes.ndim == 1:
        raise ValueError("when given, axes values must be a scalar or vector")
    if not np.issubdtype(axes.dtype, np.integer):
        raise ValueError("when given, axes values must be integers")

    axes = np.where(axes < 0, axes + x.ndim, axes)

    if axes.size != 0 and (axes.max() >= x.ndim or axes.min() < 0):
        raise ValueError("axes exceeds dimensionality of input")
    if axes.size != 0 and np.unique(axes).shape != axes.shape:
        raise ValueError("all axes must be unique")

    if not noshape:
        shape = np.atleast_1d(shape)
    elif np.isscalar(x):
        shape = np.array([], dtype=np.intc)
    elif noaxes:
        shape = np.array(x.shape, dtype=np.intc)
    else:
        shape = np.take(x.shape, axes)

    if shape.size == 0:
        shape = shape.astype(np.intc)

    if shape.ndim != 1:
        raise ValueError("when given, shape values must be a scalar or vector")
    if not np.issubdtype(shape.dtype, np.integer):
        raise ValueError("when given, shape values must be integers")
    if axes.shape != shape.shape:
        raise ValueError(
            "when given, axes and shape arguments"
            " have to be of the same length"
        )

    shape = np.where(shape == -1, np.array(x.shape)[axes], shape)

    if shape.size != 0 and (shape < 1).any():
        raise ValueError(
            "invalid number of data points ({0}) specified".format(shape)
        )

    return shape, axes


def _iter_fftnd(
    a,
    s=None,
    axes=None,
    direction=+1,
    overwrite_x=False,
    scale_function=lambda n, ind: 1.0,
):
    a = np.asarray(a)
    s, axes = _init_nd_shape_and_axes(a, s, axes)
    ovwr = overwrite_x
    for ii in reversed(range(len(axes))):
        a = _fft1d_impl(
            a,
            n=s[ii],
            axis=axes[ii],
            overwrite_x=ovwr,
            direction=direction,
            fsc=scale_function(s[ii], ii),
        )
        ovwr = True
    return a


def _flat_to_multi(ind, shape):
    nd = len(shape)
    m_ind = [-1] * nd
    j = ind
    for i in range(nd):
        si = shape[nd - 1 - i]
        q = j // si
        r = j - si * q
        m_ind[nd - 1 - i] = r
        j = q
    return m_ind


def _iter_complementary(x, axes, func, kwargs, result):
    if axes is None:
        return func(x, **kwargs)
    x_shape = x.shape
    nd = x.ndim
    r = list(range(nd))
    sl = [slice(None, None, None)] * nd
    if not np.iterable(axes):
        axes = (axes,)
    for ai in axes:
        r[ai] = None
    size = 1
    sub_shape = []
    dual_ind = []
    for ri in r:
        if ri is not None:
            size *= x_shape[ri]
            sub_shape.append(x_shape[ri])
            dual_ind.append(ri)

    for ind in range(size):
        m_ind = _flat_to_multi(ind, sub_shape)
        for k1, k2 in zip(dual_ind, m_ind):
            sl[k1] = k2
        np.copyto(result[tuple(sl)], func(x[tuple(sl)], **kwargs))

    return result


def _check_shapes_for_direct(xs, shape, axes):
    if len(axes) > 7:  # Intel MKL supports up to 7D
        return False
    if not (len(xs) == len(shape)):
        return False
    if not (len(set(axes)) == len(axes)):
        return False
    for xsi, ai in zip(xs, axes):
        try:
            sh_ai = shape[ai]
        except IndexError:
            raise ValueError("Invalid axis (%d) specified" % ai)

        if not (xsi == sh_ai):
            return False
    return True


def _output_dtype(dt):
    if dt == np.double:
        return np.cdouble
    if dt == np.single:
        return np.csingle
    return dt


def _fftnd_impl(x, s=None, axes=None, overwrite_x=False, direction=+1, fsc=1.0):
    if direction not in [-1, +1]:
        raise ValueError("Direction of FFT should +1 or -1")

    # _direct_fftnd requires complex type, and full-dimensional transform
    if isinstance(x, np.ndarray) and x.size != 0 and x.ndim > 1:
        _direct = s is None and axes is None
        if _direct:
            _direct = x.ndim <= 7  # Intel MKL only supports FFT up to 7D
        if not _direct:
            xs, xa = _cook_nd_args(x, s, axes)
            if _check_shapes_for_direct(xs, x.shape, xa):
                _direct = True
        _direct = _direct and x.dtype in [
            np.complex64,
            np.complex128,
            np.float32,
            np.float64,
        ]
    else:
        _direct = False

    if _direct:
        return _direct_fftnd(
            x, overwrite_x=overwrite_x, direction=direction, fsc=fsc
        )
    else:
        if s is None and x.dtype in [
            np.csingle,
            np.cdouble,
            np.single,
            np.double,
        ]:
            x = np.asarray(x)
            res = np.empty(x.shape, dtype=_output_dtype(x.dtype))
            return _iter_complementary(
                x,
                axes,
                _direct_fftnd,
                {
                    "overwrite_x": overwrite_x,
                    "direction": direction,
                    "fsc": fsc,
                },
                res,
            )
        else:
            return _iter_fftnd(
                x,
                s=s,
                axes=axes,
                overwrite_x=overwrite_x,
                scale_function=lambda n, i: fsc if i == 0 else 1.0,
                direction=direction,
            )

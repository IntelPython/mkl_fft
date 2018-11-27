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

from __future__ import absolute_import
import numpy as np
cimport numpy as cnp
try:
    from numpy.core.multiarray_tests import internal_overlap
except ImportError:
    # Module has been renamed in NumPy 1.15
    from numpy.core._multiarray_tests import internal_overlap

from libc.string cimport memcpy

from threading import Lock
_lock = Lock()

cdef extern from "Python.h":
    ctypedef int size_t

    void* PyMem_Malloc(size_t n)
    void PyMem_Free(void* buf)

    int PyErr_Occurred()
    void PyErr_Clear()
    long PyInt_AsLong(object ob)
    int PyObject_HasAttrString(object, char*)


# These are commented out in the numpy support we cimported above.
# Here I have declared them as taking void* instead of PyArrayDescr
# and object. In this file, only NULL is passed to these parameters.
cdef extern from *:
    cnp.ndarray PyArray_CheckFromAny(object, void*, int, int, int, void*)
    object PyArray_BASE(cnp.ndarray)

cdef extern from "src/mklfft.h":
    int cdouble_mkl_fft1d_in(cnp.ndarray, int, int)
    int cfloat_mkl_fft1d_in(cnp.ndarray, int, int)
    int float_cfloat_mkl_fft1d_out(cnp.ndarray, int, int, cnp.ndarray, int)
    int cfloat_cfloat_mkl_fft1d_out(cnp.ndarray, int, int, cnp.ndarray)
    int double_cdouble_mkl_fft1d_out(cnp.ndarray, int, int, cnp.ndarray, int)
    int cdouble_cdouble_mkl_fft1d_out(cnp.ndarray, int, int, cnp.ndarray)

    int cdouble_mkl_ifft1d_in(cnp.ndarray, int, int)
    int cfloat_mkl_ifft1d_in(cnp.ndarray, int, int)
    int float_cfloat_mkl_ifft1d_out(cnp.ndarray, int, int, cnp.ndarray, int)
    int cfloat_cfloat_mkl_ifft1d_out(cnp.ndarray, int, int, cnp.ndarray)
    int double_cdouble_mkl_ifft1d_out(cnp.ndarray, int, int, cnp.ndarray, int)
    int cdouble_cdouble_mkl_ifft1d_out(cnp.ndarray, int, int, cnp.ndarray)

    int double_mkl_rfft_in(cnp.ndarray, int, int)
    int double_mkl_irfft_in(cnp.ndarray, int, int)
    int float_mkl_rfft_in(cnp.ndarray, int, int)
    int float_mkl_irfft_in(cnp.ndarray, int, int)

    int double_double_mkl_rfft_out(cnp.ndarray, int, int, cnp.ndarray)
    int double_double_mkl_irfft_out(cnp.ndarray, int, int, cnp.ndarray)
    int float_float_mkl_rfft_out(cnp.ndarray, int, int, cnp.ndarray)
    int float_float_mkl_irfft_out(cnp.ndarray, int, int, cnp.ndarray)

    int cdouble_double_mkl_irfft_out(cnp.ndarray, int, int, cnp.ndarray)
    int cfloat_float_mkl_irfft_out(cnp.ndarray, int, int, cnp.ndarray)

    int cdouble_cdouble_mkl_fftnd_in(cnp.ndarray)
    int cdouble_cdouble_mkl_ifftnd_in(cnp.ndarray)
    int cfloat_cfloat_mkl_fftnd_in(cnp.ndarray)
    int cfloat_cfloat_mkl_ifftnd_in(cnp.ndarray)

    int cdouble_cdouble_mkl_fftnd_out(cnp.ndarray, cnp.ndarray)
    int cdouble_cdouble_mkl_ifftnd_out(cnp.ndarray, cnp.ndarray)
    int cfloat_cfloat_mkl_fftnd_out(cnp.ndarray, cnp.ndarray)
    int cfloat_cfloat_mkl_ifftnd_out(cnp.ndarray, cnp.ndarray)

    int float_cfloat_mkl_fftnd_out(cnp.ndarray, cnp.ndarray)
    int double_cdouble_mkl_fftnd_out(cnp.ndarray, cnp.ndarray)
    int float_cfloat_mkl_ifftnd_out(cnp.ndarray, cnp.ndarray)
    int double_cdouble_mkl_ifftnd_out(cnp.ndarray, cnp.ndarray)
    char * mkl_dfti_error(int)

# Initialize numpy
cnp.import_array()


cdef int _datacopied(cnp.ndarray arr, object orig):
    """
    Strict check for `arr` not sharing any data with `original`,
    under the assumption that arr = asarray(original)
    """
    if not cnp.PyArray_Check(orig) and PyObject_HasAttrString(orig, '__array__'):
        return 0
    if isinstance(orig, np.ndarray) and (arr is (<cnp.ndarray> orig)):
        return 0
    arr_obj = <object> arr
    return 1 if (arr_obj.base is None) else 0


def fft(x, n=None, axis=-1, overwrite_x=False):
    return _fft1d_impl(x, n=n, axis=axis, overwrite_arg=overwrite_x, direction=+1)


def ifft(x, n=None, axis=-1, overwrite_x=False):
    return _fft1d_impl(x, n=n, axis=axis, overwrite_arg=overwrite_x, direction=-1)


cdef cnp.ndarray pad_array(cnp.ndarray x_arr, cnp.npy_intp n, int axis, int realQ):
    "Internal utility to zero-pad input array along given axis"
    cdef cnp.ndarray b_arr "b_arrayObject"
    cdef int x_type, b_type, b_ndim, x_arr_is_fortran
    cdef cnp.npy_intp *b_shape

    x_type = cnp.PyArray_TYPE(x_arr)
    if realQ:
        b_type = x_type
    else:
        b_type = cnp.NPY_CFLOAT if (x_type is cnp.NPY_FLOAT or
                                    x_type is cnp.NPY_CFLOAT) else cnp.NPY_CDOUBLE
    b_ndim = cnp.PyArray_NDIM(x_arr)

    b_shape = <cnp.npy_intp*> PyMem_Malloc(b_ndim * sizeof(cnp.npy_intp))
    memcpy(b_shape, cnp.PyArray_DIMS(x_arr), b_ndim * sizeof(cnp.npy_intp))
    b_shape[axis] = n

    # allocating temporary buffer
    x_arr_is_fortran = cnp.PyArray_CHKFLAGS(x_arr, cnp.NPY_F_CONTIGUOUS)
    b_arr = <cnp.ndarray> cnp.PyArray_EMPTY(
        b_ndim, b_shape, <cnp.NPY_TYPES> b_type, x_arr_is_fortran) # 0 for C-contiguous
    PyMem_Free(b_shape)

    ind = [slice(0, None, None), ] * b_ndim
    ind[axis] = slice(0, cnp.PyArray_DIM(x_arr, axis), None)
    bo = <object> b_arr
    xo = <object> x_arr
    bo[tuple(ind)] = xo
    ind[axis] = slice(cnp.PyArray_DIM(x_arr, axis), None, None)
    bo[tuple(ind)] = 0.0

    return b_arr


cdef cnp.ndarray  __process_arguments(object x, object n, object axis,
                                      object overwrite_arg, object direction,
                                      long *axis_, long *n_, int *in_place,
                                      int *xnd, int *dir_, int realQ):
    "Internal utility to validate and process input arguments of 1D FFT functions"
    cdef int err
    cdef long n_max = 0
    cdef cnp.ndarray x_arr "xx_arrayObject"

    if direction not in [-1, +1]:
        raise ValueError("Direction of FFT should +1 or -1")
    else:
        dir_[0] = -1 if direction is -1 else +1

    in_place[0] = 1 if overwrite_arg is True else 0

    # convert x to ndarray, ensure that strides are multiples of itemsize
    x_arr = PyArray_CheckFromAny(
          x, NULL, 0, 0,
          cnp.NPY_ELEMENTSTRIDES | cnp.NPY_ENSUREARRAY | cnp.NPY_NOTSWAPPED,
          NULL)

    if <void *> x_arr is NULL:
        raise ValueError("An input argument x is not an array-like object")

    if _datacopied(x_arr, x):
        in_place[0] = 1  # a copy was made, so we can work in place.

    xnd[0] = cnp.PyArray_NDIM(x_arr) # tensor-rank of the array

    err = 0
    axis_[0] = PyInt_AsLong(axis)
    if (axis_[0] == -1 and PyErr_Occurred()):
        PyErr_Clear()
        err = 1
    elif not (-xnd[0] <= axis_[0] < xnd[0]):
        err = 1

    if err:
        raise ValueError("Invalid axis (%d) specified." % axis)

    axis_[0] = axis_[0] if axis_[0] >= 0 else xnd[0] + axis_[0]

    if n is None:
        n_[0] = x_arr.shape[axis_[0]]
    else:
        try:
            n_[0] = PyInt_AsLong(n)
        except:
            err = 1

    if not err:
        n_max = <long> cnp.PyArray_DIM(x_arr, axis_[0])
        if n_[0] < 1:
            err = 1
        elif n_[0] > n_max:
            in_place[0] = 1  # we must copy to pad and will work in-place
            x_arr = pad_array(x_arr, n_[0], axis_[0], realQ)

    if err:
        raise ValueError("Dimension n should be a positive integer not "
                         "larger than the shape of the array along the chosen axis")

    return x_arr


cdef cnp.ndarray __allocate_result(cnp.ndarray x_arr, long n_, long axis_, int f_type):
    """
    An internal utility to allocate an empty array for output of not-in-place FFT.
    """
    cdef cnp.npy_intp *f_shape
    cdef cnp.ndarray f_arr "ff_arrayObject"
    cdef int x_arr_is_fortran

    f_ndim = cnp.PyArray_NDIM(x_arr)

    f_shape = <cnp.npy_intp*> PyMem_Malloc(f_ndim * sizeof(cnp.npy_intp))
    memcpy(f_shape, cnp.PyArray_DIMS(x_arr), f_ndim * sizeof(cnp.npy_intp))
    # if dimension is negative, do not alter the dimension
    if n_ > 0:
        f_shape[axis_] = n_

    # allocating output buffer
    x_arr_is_fortran = cnp.PyArray_CHKFLAGS(x_arr, cnp.NPY_F_CONTIGUOUS)
    f_arr = <cnp.ndarray> cnp.PyArray_EMPTY(
        f_ndim, f_shape, <cnp.NPY_TYPES> f_type, x_arr_is_fortran) # 0 for C-contiguous
    PyMem_Free(f_shape);

    return f_arr


# this routine implements complex forward/backward FFT
# Float/double inputs are not cast to complex, but are effectively
# treated as complexes with zero imaginary parts.
# All other types are cast to complex double.
def _fft1d_impl(x, n=None, axis=-1, overwrite_arg=False, direction=+1):
    """
    Uses MKL to perform 1D FFT on the input array x along the given axis.
    """
    cdef cnp.ndarray x_arr "x_arrayObject"
    cdef cnp.ndarray f_arr "f_arrayObject"
    cdef int xnd, err, n_max = 0, in_place, dir_
    cdef long n_, axis_
    cdef int x_type, f_type, status = 0
    cdef int ALL_HARMONICS = 1
    cdef char * c_error_msg = NULL
    cdef bytes py_error_msg

    x_arr = __process_arguments(x, n, axis, overwrite_arg, direction,
                                &axis_, &n_, &in_place, &xnd, &dir_, 0)

    x_type = cnp.PyArray_TYPE(x_arr)

    if x_type is cnp.NPY_CFLOAT or x_type is cnp.NPY_CDOUBLE:
        # we can operate in place if requested.
        if in_place:
           if not cnp.PyArray_ISONESEGMENT(x_arr):
              in_place = 0 if internal_overlap(x_arr) else 1;
    elif x_type is cnp.NPY_FLOAT or x_type is cnp.NPY_DOUBLE:
        # to work in place we need to cast the input to complex,
        # which may be more expensive than creating the output using MKL
        in_place = 0
    else:
        # we must cast the input and allocate the output,
        # so we cast to complex double and operate in place
        try:
            x_arr = <cnp.ndarray> cnp.PyArray_FROM_OTF(
                x_arr, cnp.NPY_CDOUBLE, cnp.NPY_BEHAVED)
        except:
            raise ValueError("First argument must be a complex or real sequence of single or double precision")
        x_type = cnp.PyArray_TYPE(x_arr)
        in_place = 1

    if in_place:
        with _lock:
            if x_type is cnp.NPY_CDOUBLE:
                if dir_ < 0:
                   status = cdouble_mkl_ifft1d_in(x_arr, n_, <int> axis_)
                else:
                   status = cdouble_mkl_fft1d_in(x_arr, n_, <int> axis_)
            elif x_type is cnp.NPY_CFLOAT:
                if dir_ < 0:
                   status = cfloat_mkl_ifft1d_in(x_arr, n_, <int> axis_)
                else:
                   status = cfloat_mkl_fft1d_in(x_arr, n_, <int> axis_)
            else:
                status = 1

        if status:
            c_error_msg = mkl_dfti_error(status)
            py_error_msg = c_error_msg
            raise ValueError("Internal error occurred: {}".format(py_error_msg))

        n_max = <long> cnp.PyArray_DIM(x_arr, axis_)
        if (n_ < n_max):
            ind = [slice(0, None, None), ] * xnd
            ind[axis_] = slice(0, n_, None)
            x_arr = x_arr[tuple(ind)]

        return x_arr
    else:
        f_type = cnp.NPY_CFLOAT if (x_type is cnp.NPY_FLOAT or
                                    x_type is cnp.NPY_CFLOAT) else cnp.NPY_CDOUBLE
        f_arr = __allocate_result(x_arr, n_, axis_, f_type);

        # call out-of-place FFT
        with _lock:
            if f_type is cnp.NPY_CDOUBLE:
                if x_type is cnp.NPY_DOUBLE:
                    if dir_ < 0:
                       status = double_cdouble_mkl_ifft1d_out(
                           x_arr, n_, <int> axis_, f_arr, ALL_HARMONICS)
                    else:
                       status = double_cdouble_mkl_fft1d_out(
                           x_arr, n_, <int> axis_, f_arr, ALL_HARMONICS)
                elif x_type is cnp.NPY_CDOUBLE:
                    if dir_ < 0:
                        status = cdouble_cdouble_mkl_ifft1d_out(
                            x_arr, n_, <int> axis_, f_arr)
                    else:
                        status = cdouble_cdouble_mkl_fft1d_out(
                            x_arr, n_, <int> axis_, f_arr)
            else:
                if x_type is cnp.NPY_FLOAT:
                    if dir_ < 0:
                       status = float_cfloat_mkl_ifft1d_out(
                           x_arr, n_, <int> axis_, f_arr, ALL_HARMONICS)
                    else:
                       status = float_cfloat_mkl_fft1d_out(
                           x_arr, n_, <int> axis_, f_arr, ALL_HARMONICS)
                elif x_type is cnp.NPY_CFLOAT:
                    if dir_ < 0:
                        status = cfloat_cfloat_mkl_ifft1d_out(
                            x_arr, n_, <int> axis_, f_arr)
                    else:
                        status = cfloat_cfloat_mkl_fft1d_out(
                            x_arr, n_, <int> axis_, f_arr)

        if (status):
            c_error_msg = mkl_dfti_error(status)
            py_error_msg = c_error_msg
            raise ValueError("Internal error occurred: {}".format(py_error_msg))

        return f_arr


def rfft(x, n=None, axis=-1, overwrite_x=False):
    """Packed real-valued harmonics of FFT of a real sequence x"""
    return _rrfft1d_impl(x, n=n, axis=axis, overwrite_arg=overwrite_x, direction=+1)


def irfft(x, n=None, axis=-1, overwrite_x=False):
    """Inverse FFT of a real sequence, takes packed real-valued harmonics of FFT"""
    return _rrfft1d_impl(x, n=n, axis=axis, overwrite_arg=overwrite_x, direction=-1)


def _rrfft1d_impl(x, n=None, axis=-1, overwrite_arg=False, direction=+1):
    """
    Uses MKL to perform real packed 1D FFT on the input array x along the given axis.
    """
    cdef cnp.ndarray x_arr "x_arrayObject"
    cdef cnp.ndarray f_arr "f_arrayObject"
    cdef int xnd, err, n_max = 0, in_place, dir_
    cdef long n_, axis_
    cdef int x_type, status
    cdef char * c_error_msg = NULL
    cdef bytes py_error_msg

    x_arr = __process_arguments(x, n, axis, overwrite_arg, direction,
                                &axis_, &n_, &in_place, &xnd, &dir_, 1)

    x_type = cnp.PyArray_TYPE(x_arr)

    if x_type is cnp.NPY_FLOAT or x_type is cnp.NPY_DOUBLE:
        # we can operate in place if requested.
        if in_place:
           if not cnp.PyArray_ISONESEGMENT(x_arr):
              in_place = 0 if internal_overlap(x_arr) else 1;
    elif x_type is cnp.NPY_CFLOAT or x_type is cnp.NPY_CDOUBLE:
        raise TypeError("1st argument must be a real sequence")
    else:
        # we must cast the input and allocate the output,
        # so we cast to double and operate in place
        try:
            x_arr = <cnp.ndarray> cnp.PyArray_FROM_OTF(
                x_arr, cnp.NPY_DOUBLE, cnp.NPY_BEHAVED)
        except:
            raise TypeError("1st argument must be a real sequence")
        x_type = cnp.PyArray_TYPE(x_arr)
        in_place = 1

    if in_place:
        with _lock:
            if x_type is cnp.NPY_DOUBLE:
                if dir_ < 0:
                   status = double_mkl_irfft_in(x_arr, n_, <int> axis_)
                else:
                   status = double_mkl_rfft_in(x_arr, n_, <int> axis_)
            elif x_type is cnp.NPY_FLOAT:
                if dir_ < 0:
                   status = float_mkl_irfft_in(x_arr, n_, <int> axis_)
                else:
                   status = float_mkl_rfft_in(x_arr, n_, <int> axis_)
            else:
                status = 1

        if status:
            c_error_msg = mkl_dfti_error(status)
            py_error_msg = c_error_msg
            raise ValueError("Internal error occurred: {}".format(py_error_msg))

        n_max = <long> cnp.PyArray_DIM(x_arr, axis_)
        if (n_ < n_max):
            ind = [slice(0, None, None), ] * xnd
            ind[axis_] = slice(0, n_, None)
            x_arr = x_arr[tuple(ind)]

        return x_arr
    else:
        f_arr = __allocate_result(x_arr, n_, axis_, x_type);

        # call out-of-place FFT
        with _lock:
            if x_type is cnp.NPY_DOUBLE:
                if dir_ < 0:
                   status = double_double_mkl_irfft_out(x_arr, n_, <int> axis_, f_arr)
                else:
                   status = double_double_mkl_rfft_out(x_arr, n_, <int> axis_, f_arr)
            else:
                if dir_ < 0:
                   status = float_float_mkl_irfft_out(x_arr, n_, <int> axis_, f_arr)
                else:
                   status = float_float_mkl_rfft_out(x_arr, n_, <int> axis_, f_arr)

        if (status):
            c_error_msg = mkl_dfti_error(status)
            py_error_msg = c_error_msg
            raise ValueError("Internal error occurred: {}".format(py_error_msg))

        return f_arr

# this routine is functionally equivalent to numpy.fft.rfft
def _rc_fft1d_impl(x, n=None, axis=-1, overwrite_arg=False):
    """
    Uses MKL to perform 1D FFT on the real input array x along the given axis,
    producing complex output, but giving only half of the harmonics.

    cf. numpy.fft.rfft
    """
    cdef cnp.ndarray x_arr "x_arrayObject"
    cdef cnp.ndarray f_arr "f_arrayObject"
    cdef int xnd, err, n_max = 0, in_place, dir_
    cdef long n_, axis_
    cdef int x_type, f_type, status, requirement
    cdef int HALF_HARMONICS = 0 # give only positive index harmonics
    cdef int direction = 1 # dummy, only used for the sake of arg-processing
    cdef char * c_error_msg = NULL
    cdef bytes py_error_msg

    x_arr = __process_arguments(x, n, axis, overwrite_arg, direction,
                                &axis_, &n_, &in_place, &xnd, &dir_, 1)

    x_type = cnp.PyArray_TYPE(x_arr)

    if x_type is cnp.NPY_CFLOAT or x_type is cnp.NPY_CDOUBLE or x_type is cnp.NPY_CLONGDOUBLE:
        raise TypeError("1st argument must be a real sequence 1")
    elif x_type is cnp.NPY_FLOAT or x_type is cnp.NPY_DOUBLE:
        pass
    else:
        # we must cast the input to doubles and allocate the output,
        try:
            requirement = cnp.NPY_BEHAVED
            if x_type is cnp.NPY_LONGDOUBLE:
                requirement = requirement | cnp.NPY_FORCECAST
            x_arr = <cnp.ndarray> cnp.PyArray_FROM_OTF(
                x_arr, cnp.NPY_DOUBLE, requirement)
            x_type = cnp.PyArray_TYPE(x_arr)
        except:
            raise TypeError("1st argument must be a real sequence 2")

    # in_place is ignored here.
    # it can be done only if 2*(n_ // 2 + 1)  <= x_arr.shape[axis_] which is not
    # the common usage
    f_type = cnp.NPY_CFLOAT if x_type is cnp.NPY_FLOAT else cnp.NPY_CDOUBLE
    f_arr = __allocate_result(x_arr, n_ // 2 + 1, axis_, f_type);

    # call out-of-place FFT
    if x_type is cnp.NPY_FLOAT:
        with _lock:
            status = float_cfloat_mkl_fft1d_out(x_arr, n_, <int> axis_, f_arr, HALF_HARMONICS)
    else:
        with _lock:
            status = double_cdouble_mkl_fft1d_out(x_arr, n_, <int> axis_, f_arr, HALF_HARMONICS)

    if (status):
        c_error_msg = mkl_dfti_error(status)
        py_error_msg = c_error_msg
        raise ValueError("Internal error occurred: {}".format(str(py_error_msg)))

    return f_arr


cdef int _is_integral(object num):
    cdef long n
    cdef int _integral
    if num is None:
        return 0
    try:
        n = PyInt_AsLong(num)
        _integral = 1 if n > 0 else 0
    except:
        _integral = 0

    return _integral


# this routine is functionally equivalent to numpy.fft.rfft
def _rc_ifft1d_impl(x, n=None, axis=-1, overwrite_arg=False):
    """
    Uses MKL to perform 1D FFT on the real input array x along the given axis,
    producing complex output, but giving only half of the harmonics.

    cf. numpy.fft.rfft
    """
    cdef cnp.ndarray x_arr "x_arrayObject"
    cdef cnp.ndarray f_arr "f_arrayObject"
    cdef int xnd, err, n_max = 0, in_place, dir_, int_n
    cdef long n_, axis_
    cdef int x_type, f_type, status
    cdef int direction = 1 # dummy, only used for the sake of arg-processing
    cdef char * c_error_msg = NULL
    cdef bytes py_error_msg

    int_n = _is_integral(n)
    # nn gives the number elements along axis of the input that we use
    nn = (n // 2 + 1) if int_n and n > 0 else n
    x_arr = __process_arguments(x, nn, axis, overwrite_arg, direction,
                                &axis_, &n_, &in_place, &xnd, &dir_, 0)
    n_ = 2*(n_ - 1)
    if int_n and (n % 2 == 1):
        n_ += 1

    x_type = cnp.PyArray_TYPE(x_arr)

    if x_type is cnp.NPY_CFLOAT or x_type is cnp.NPY_CDOUBLE:
        # we can operate in place if requested.
        if in_place:
           if not cnp.PyArray_ISONESEGMENT(x_arr):
              in_place = 0 if internal_overlap(x_arr) else 1;
    else:
        # we must cast the input and allocate the output,
        # so we cast to complex double and operate in place
        try:
            x_arr = <cnp.ndarray> cnp.PyArray_FROM_OTF(
                x_arr, cnp.NPY_CDOUBLE, cnp.NPY_BEHAVED)
        except:
            ValueError("First argument should be a real or a complex sequence of single or double precision")
        x_type = cnp.PyArray_TYPE(x_arr)
        in_place = 1

    in_place = 0
    if in_place:
        # TODO: Provide in-place functionality
        pass
    else:
        f_type = cnp.NPY_FLOAT if x_type is cnp.NPY_CFLOAT else cnp.NPY_DOUBLE
        f_arr = __allocate_result(x_arr, n_, axis_, f_type);

        # call out-of-place FFT
        if x_type is cnp.NPY_CFLOAT:
            with _lock:
                status = cfloat_float_mkl_irfft_out(x_arr, n_, <int> axis_, f_arr)
        else:
            with _lock:
                status = cdouble_double_mkl_irfft_out(x_arr, n_, <int> axis_, f_arr)

        if (status):
            c_error_msg = mkl_dfti_error(status)
            py_error_msg = c_error_msg
            raise ValueError("Internal error occurred: {}".format(str(py_error_msg)))

        return f_arr


def rfft_numpy(x, n=None, axis=-1):
    return _rc_fft1d_impl(x, n=n, axis=axis)


def irfft_numpy(x, n=None, axis=-1):
    return _rc_ifft1d_impl(x, n=n, axis=axis)


# ============================== ND ====================================== #

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
        raise ValueError("when given, axes and shape arguments"
                         " have to be of the same length")

    shape = np.where(shape == -1, np.array(x.shape)[axes], shape)

    if shape.size != 0 and (shape < 1).any():
        raise ValueError(
            "invalid number of data points ({0}) specified".format(shape))

    return shape, axes


def _cook_nd_args(a, s=None, axes=None, invreal=0):
    if s is None:
        shapeless = 1
        if axes is None:
            s = list(a.shape)
        else:
            try:
                s = [ a.shape[i] for i in axes ]
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


def _iter_fftnd(a, s=None, axes=None, function=fft, overwrite_arg=False):
    a = np.asarray(a)
    s, axes = _init_nd_shape_and_axes(a, s, axes)
    ovwr = overwrite_arg
    for ii in reversed(range(len(axes))):
        a = function(a, n = s[ii], axis = axes[ii], overwrite_x=ovwr)
        ovwr = True
    return a


def _direct_fftnd(x, overwrite_arg=False, direction=+1):
    """Perform n-dimensional FFT over all axes"""
    cdef int err
    cdef long n_max = 0
    cdef cnp.ndarray x_arr "xxnd_arrayObject"
    cdef cnp.ndarray f_arr "ffnd_arrayObject"
    cdef int xnd, dir_, in_place, x_type, f_type

    if direction not in [-1, +1]:
        raise ValueError("Direction of FFT should +1 or -1")
    else:
        dir_ = -1 if direction is -1 else +1

    in_place = 1 if overwrite_arg is True else 0

    # convert x to ndarray, ensure that strides are multiples of itemsize
    x_arr = PyArray_CheckFromAny(
          x, NULL, 0, 0,
          cnp.NPY_ELEMENTSTRIDES | cnp.NPY_ENSUREARRAY | cnp.NPY_NOTSWAPPED,
          NULL)

    if <void *> x_arr is NULL:
       raise ValueError("An input argument x is not an array-like object")

    if _datacopied(x_arr, x):
       in_place = 1  # a copy was made, so we can work in place.

    x_type = cnp.PyArray_TYPE(x_arr)
    assert( x_type == cnp.NPY_CDOUBLE or x_type == cnp.NPY_CFLOAT or x_type == cnp.NPY_DOUBLE or x_type == cnp.NPY_FLOAT);

    if in_place:
        in_place = 1 if x_type == cnp.NPY_CDOUBLE or x_type == cnp.NPY_CFLOAT else 0

    if in_place:
        if x_type == cnp.NPY_CDOUBLE:
            if dir_ == 1:
                err = cdouble_cdouble_mkl_fftnd_in(x_arr)
            else:
                err = cdouble_cdouble_mkl_ifftnd_in(x_arr)
        elif x_type == cnp.NPY_CFLOAT:
            if dir_ == 1:
                err = cfloat_cfloat_mkl_fftnd_in(x_arr)
            else:
                err = cfloat_cfloat_mkl_ifftnd_in(x_arr)
        else:
            raise ValueError("An input argument x is not complex type array")

        return x_arr
    else:
        f_type = cnp.NPY_CDOUBLE if x_type == cnp.NPY_CDOUBLE or x_type == cnp.NPY_DOUBLE else cnp.NPY_CFLOAT
        f_arr = __allocate_result(x_arr, -1, 0, f_type);
        if x_type == cnp.NPY_CDOUBLE:
            if dir_ == 1:
                err = cdouble_cdouble_mkl_fftnd_out(x_arr, f_arr)
            else:
                err = cdouble_cdouble_mkl_ifftnd_out(x_arr, f_arr)
        elif x_type == cnp.NPY_CFLOAT:
            if dir_ == 1:
                err = cfloat_cfloat_mkl_fftnd_out(x_arr, f_arr)
            else:
                err = cfloat_cfloat_mkl_ifftnd_out(x_arr, f_arr)
        elif x_type == cnp.NPY_DOUBLE:
            if dir_ == 1:
                err = double_cdouble_mkl_fftnd_out(x_arr, f_arr)
            else:
                err = double_cdouble_mkl_ifftnd_out(x_arr, f_arr)
        elif x_type == cnp.NPY_FLOAT:
            if dir_ == 1:
                err = float_cfloat_mkl_fftnd_out(x_arr, f_arr)
            else:
                err = float_cfloat_mkl_ifftnd_out(x_arr, f_arr)
        else:
            raise ValueError("An input argument x is not complex type array")

        return f_arr

def _check_shapes_for_direct(xs, shape, axes):
    if len(axes) > 7: # Intel MKL supports up to 7D
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


def _fftnd_impl(x, shape=None, axes=None, overwrite_x=False, direction=+1):
    if direction not in [-1, +1]:
        raise ValueError("Direction of FFT should +1 or -1")

    # _direct_fftnd requires complex type, and full-dimensional transform
    if isinstance(x, np.ndarray) and x.size != 0 and x.ndim > 1:
        _direct = shape is None and axes is None
        if _direct:
            _direct = x.ndim <= 7 # Intel MKL only supports FFT up to 7D
        if not _direct:
            xs, xa = _cook_nd_args(x, shape, axes)
            if _check_shapes_for_direct(xs, x.shape, xa):
                _direct = True
        _direct = _direct and x.dtype in [np.complex64, np.complex128, np.float32, np.float64]
    else:
        _direct = False

    if _direct:
        return _direct_fftnd(x, overwrite_arg=overwrite_x, direction=direction)
    else:
        return _iter_fftnd(x, s=shape, axes=axes, 
                           overwrite_arg=overwrite_x, 
                           function=fft if direction == 1 else ifft)


def fft2(x, shape=None, axes=(-2,-1), overwrite_x=False):
    return _fftnd_impl(x, shape=shape, axes=axes, overwrite_x=overwrite_x, direction=+1)


def ifft2(x, shape=None, axes=(-2,-1), overwrite_x=False):
    return _fftnd_impl(x, shape=shape, axes=axes, overwrite_x=overwrite_x, direction=-1)


def fftn(x, shape=None, axes=None, overwrite_x=False):
    return _fftnd_impl(x, shape=shape, axes=axes, overwrite_x=overwrite_x, direction=+1)


def ifftn(x, shape=None, axes=None, overwrite_x=False):
    return _fftnd_impl(x, shape=shape, axes=axes, overwrite_x=overwrite_x, direction=-1)


def rfft2_numpy(x, s=None, axes=(-2,-1)):
    return rfftn_numpy(x, s=s, axes=axes)


def irfft2_numpy(x, s=None, axes=(-2,-1)):
    return irfftn_numpy(x, s=s, axes=axes)


def _remove_axis(s, axes, axis_to_remove):
    lens = len(s)
    axes_normalized = tuple(lens + ai if ai < 0 else ai for ai in axes)
    a2r = lens + axis_to_remove if axis_to_remove < 0 else axis_to_remove

    ss = s[:a2r] + s[a2r+1:]
    aa = axes_normalized[:a2r] + tuple(ai - 1 for ai in axes_normalized[a2r+1:])
    return ss, aa


cdef cnp.ndarray _trim_array(cnp.ndarray arr, object s, object axes):
    """Forms a view into subarray of arr if any element of shape parameter s is 
    smaller than the corresponding element of the shape of the input array arr,
    otherwise returns the input array"""
    arr_shape = (<object> arr).shape
    no_trim = True
    for si, ai in zip(s, axes):
        try:
            shp_i = arr_shape[ai]
        except IndexError:
            raise ValueError("Invalid axis (%d) specified" % ai)
        if si < shp_i:
            if no_trim:
                ind = [slice(None,None,None),] * len(s)
            no_trim = False
            ind[ai] = slice(None, si, None)
    if no_trim:
        return arr
    return arr[ tuple(ind) ]


def _fix_dimensions(cnp.ndarray arr, object s, object axes):
    """Pads array arr with zeros to attain shape s associated with axes"""
    arr_shape = (<object> arr).shape
    no_trim = True
    for si, ai in zip(s, axes):
        try:
            shp_i = arr_shape[ai]
        except IndexError:
            raise ValueError("Invalid axis (%d) specified" % ai)
        if si > shp_i:
            if no_trim:
                pad_widths = [(0,0),] * len(arr_shape)
            no_trim = False
            pad_widths[ai] = (0, si - shp_i)
    if no_trim:
        return arr
    return np.pad(arr, tuple(pad_widths), 'constant')


def rfftn_numpy(x, s=None, axes=None):
    a = np.asarray(x)
    no_trim = (s is None) and (axes is None)
    s, axes = _cook_nd_args(a, s, axes)
    la = axes[-1]
    # trim array, so that rfft_numpy avoids doing
    # unnecessary computations
    if not no_trim:
        a = _trim_array(a, s, axes)
    a = rfft_numpy(a, n = s[-1], axis=la)
    if len(s) > 1:
        if not no_trim:
            ss = list(s)
            ss[-1] = a.shape[la]
            a = _fix_dimensions(a, tuple(ss), axes)
        if len(set(axes)) == len(axes) and len(axes) == a.ndim and len(axes) > 2:
            ss, aa = _remove_axis(s, axes, la)
            ind = [slice(None,None,1),] * len(s)
            for ii in range(a.shape[la]):
                ind[la] = ii
                tind = tuple(ind)
                a_inp = a[tind]
                a_res = _fftnd_impl(
                    a_inp, shape=ss, axes=aa,
                    overwrite_x=True, direction=1)
                if a_res is not a_inp:
                    a[tind] = a_res # copy in place
        else:
            for ii in range(len(axes)-1):
                a = fft(a, s[ii], axes[ii], overwrite_x=True)
    return a


def irfftn_numpy(x, s=None, axes=None):
    a = np.asarray(x)
    no_trim = (s is None) and (axes is None)
    s, axes = _cook_nd_args(a, s, axes, invreal=True)
    la = axes[-1]
    if len(s) > 1:
        if not no_trim:
            a = _fix_dimensions(a, s, axes)
        ovr_x = True if _datacopied(<cnp.ndarray> a, x) else False
        if len(set(axes)) == len(axes) and len(axes) == a.ndim and len(axes) > 2:
            # due to need to write into a, we must copy
            if not ovr_x:
                a = a.copy()
                ovr_x = True
            ss, aa = _remove_axis(s, axes, la)
            ind = [slice(None,None,1),] * len(s)
            for ii in range(a.shape[la]):
                ind[la] = ii
                tind = tuple(ind)
                a_inp = a[tind]
                a_res = _fftnd_impl(
                    a_inp, shape=ss, axes=aa,
                    overwrite_x=True, direction=-1)
                if a_res is not a_inp:
                    a[tind] = a_res # copy in place
        else:
            for ii in range(len(axes)-1):
                a = ifft(a, s[ii], axes[ii], overwrite_x=ovr_x)
                ovr_x = True
    a = irfft_numpy(a, n = s[-1], axis=la)
    return a

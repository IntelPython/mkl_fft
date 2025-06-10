# cython: language_level=3
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

# imports
import sys

import numpy as np

if np.lib.NumpyVersion(np.__version__) >= "2.0.0a0":
    from numpy._core._multiarray_tests import internal_overlap
else:
    from numpy.core._multiarray_tests import internal_overlap

from threading import local as threading_local

# cimports
cimport cpython.pycapsule
cimport numpy as cnp
from cpython.exc cimport PyErr_Clear, PyErr_Occurred
from cpython.mem cimport PyMem_Free, PyMem_Malloc
from libc.string cimport memcpy

# thread-local storage
_tls = threading_local()

cdef const char *capsule_name = "dfti_cache"

cdef void _capsule_destructor(object caps) noexcept:
    cdef DftiCache *_cache = NULL
    cdef int status = 0
    if (caps is None):
        print(
            "CapsuleDestructorInternalError: Nothing to destroy",
            file=sys.stderr,
        )
        return
    _cache = <DftiCache *>cpython.pycapsule.PyCapsule_GetPointer(
        caps, capsule_name
    )
    status = _free_dfti_cache(_cache)
    PyMem_Free(_cache)
    if (status != 0):
        print("CapsuleDestructorInternalError: Freeing DFTI Cache "
              f"returned with error code = '{status}'", file=sys.stderr)


def _tls_dfti_cache_capsule():
    cdef DftiCache *_cache_struct

    init = getattr(_tls, "initialized", None)
    if (init is None):
        _cache_struct = <DftiCache *> PyMem_Malloc(sizeof(DftiCache))
        # important to initialized
        _cache_struct.initialized = 0
        _cache_struct.hand = NULL
        _tls.initialized = True
        _tls.capsule = cpython.pycapsule.PyCapsule_New(
            <void *>_cache_struct, capsule_name, &_capsule_destructor
        )
    capsule = getattr(_tls, "capsule", None)
    if (not cpython.pycapsule.PyCapsule_IsValid(capsule, capsule_name)):
        raise ValueError("Internal Error: invalid capsule stored in TLS")
    return capsule


cdef extern from "Python.h":
    ctypedef int size_t

    long PyLong_AsLong(object ob)
    int PyObject_HasAttrString(object, char*)


# These are commented out in the numpy support we cimported above.
# Here I have declared them as taking void* instead of PyArrayDescr
# and object. In this file, only NULL is passed to these parameters.
cdef extern from *:
    cnp.ndarray PyArray_CheckFromAny(object, void*, int, int, int, void*)
    object PyArray_BASE(cnp.ndarray)

cdef extern from "src/mklfft.h":
    cdef struct DftiCache:
        void * hand
        int initialized
    int _free_dfti_cache(DftiCache *)
    int cdouble_mkl_fft1d_in(cnp.ndarray, int, int, double, DftiCache*)
    int cfloat_mkl_fft1d_in(cnp.ndarray, int, int, double, DftiCache*)
    int float_cfloat_mkl_fft1d_out(
        cnp.ndarray, int, int, cnp.ndarray, int, double, DftiCache*
    )
    int cfloat_cfloat_mkl_fft1d_out(
        cnp.ndarray, int, int, cnp.ndarray, double, DftiCache*
    )
    int double_cdouble_mkl_fft1d_out(
        cnp.ndarray, int, int, cnp.ndarray, int, double, DftiCache*
    )
    int cdouble_cdouble_mkl_fft1d_out(
        cnp.ndarray, int, int, cnp.ndarray, double, DftiCache*
    )

    int cdouble_mkl_ifft1d_in(cnp.ndarray, int, int, double, DftiCache*)
    int cfloat_mkl_ifft1d_in(cnp.ndarray, int, int, double, DftiCache*)
    int float_cfloat_mkl_ifft1d_out(
        cnp.ndarray, int, int, cnp.ndarray, int, double, DftiCache*
    )
    int cfloat_cfloat_mkl_ifft1d_out(
        cnp.ndarray, int, int, cnp.ndarray, double, DftiCache*
    )
    int double_cdouble_mkl_ifft1d_out(
        cnp.ndarray, int, int, cnp.ndarray, int, double, DftiCache*
    )
    int cdouble_cdouble_mkl_ifft1d_out(
        cnp.ndarray, int, int, cnp.ndarray, double, DftiCache*
    )

    int cdouble_double_mkl_irfft_out(
        cnp.ndarray, int, int, cnp.ndarray, double, DftiCache*
    )
    int cfloat_float_mkl_irfft_out(
        cnp.ndarray, int, int, cnp.ndarray, double, DftiCache*
    )

    int cdouble_cdouble_mkl_fftnd_in(cnp.ndarray, double)
    int cdouble_cdouble_mkl_ifftnd_in(cnp.ndarray, double)
    int cfloat_cfloat_mkl_fftnd_in(cnp.ndarray, double)
    int cfloat_cfloat_mkl_ifftnd_in(cnp.ndarray, double)

    int cdouble_cdouble_mkl_fftnd_out(cnp.ndarray, cnp.ndarray, double)
    int cdouble_cdouble_mkl_ifftnd_out(cnp.ndarray, cnp.ndarray, double)
    int cfloat_cfloat_mkl_fftnd_out(cnp.ndarray, cnp.ndarray, double)
    int cfloat_cfloat_mkl_ifftnd_out(cnp.ndarray, cnp.ndarray, double)

    int float_cfloat_mkl_fftnd_out(cnp.ndarray, cnp.ndarray, double)
    int double_cdouble_mkl_fftnd_out(cnp.ndarray, cnp.ndarray, double)
    int float_cfloat_mkl_ifftnd_out(cnp.ndarray, cnp.ndarray, double)
    int double_cdouble_mkl_ifftnd_out(cnp.ndarray, cnp.ndarray, double)
    char * mkl_dfti_error(int)


# Initialize numpy
cdef int numpy_import_status = cnp.import_array()
if numpy_import_status < 0:
    raise ImportError("Failed to import NumPy as dependency of mkl_fft")


cdef int _datacopied(cnp.ndarray arr, object orig):
    """
    Strict check for `arr` not sharing any data with `original`,
    under the assumption that arr = asarray(original)
    """
    if not cnp.PyArray_Check(orig) and PyObject_HasAttrString(
        orig, "__array__"
    ):
        return 0
    if isinstance(orig, np.ndarray) and (arr is (<cnp.ndarray> orig)):
        return 0
    arr_obj = <object> arr
    return 1 if (arr_obj.base is None) else 0


cdef cnp.ndarray _pad_array(
    cnp.ndarray x_arr, cnp.npy_intp n, int axis, int realQ
):
    "Internal utility to zero-pad input array along given axis"
    cdef cnp.ndarray b_arr "b_arrayObject"
    cdef int x_type, b_type, b_ndim, x_arr_is_fortran
    cdef cnp.npy_intp *b_shape

    x_type = cnp.PyArray_TYPE(x_arr)
    if realQ:
        b_type = x_type
    else:
        if x_type is cnp.NPY_FLOAT or x_type is cnp.NPY_CFLOAT:
            b_type = cnp.NPY_CFLOAT
        else:
            b_type = cnp.NPY_CDOUBLE
    b_ndim = cnp.PyArray_NDIM(x_arr)

    b_shape = <cnp.npy_intp*> PyMem_Malloc(b_ndim * sizeof(cnp.npy_intp))
    memcpy(b_shape, cnp.PyArray_DIMS(x_arr), b_ndim * sizeof(cnp.npy_intp))
    b_shape[axis] = n

    # allocating temporary buffer
    x_arr_is_fortran = cnp.PyArray_CHKFLAGS(x_arr, cnp.NPY_ARRAY_F_CONTIGUOUS)
    b_arr = <cnp.ndarray> cnp.PyArray_EMPTY(
        b_ndim, b_shape, <cnp.NPY_TYPES> b_type, x_arr_is_fortran
    )  # 0 for C-contiguous
    PyMem_Free(b_shape)

    ind = [slice(0, None, None), ] * b_ndim
    ind[axis] = slice(0, cnp.PyArray_DIM(x_arr, axis), None)
    bo = <object> b_arr
    xo = <object> x_arr
    bo[tuple(ind)] = xo
    ind[axis] = slice(cnp.PyArray_DIM(x_arr, axis), None, None)
    bo[tuple(ind)] = 0.0

    return b_arr


cdef cnp.ndarray _process_arguments(
    object x,
    object n,
    object axis,
    object overwrite_x,
    object direction,
    long *axis_,
    long *n_,
    int *in_place,
    int *xnd,
    int *dir_,
    int realQ,
):
    """
    Internal utility to validate and process input arguments of 1D FFT functions
    """
    cdef int err
    cdef long n_max = 0
    cdef cnp.ndarray x_arr "xx_arrayObject"

    if direction not in [-1, +1]:
        raise ValueError("Direction of FFT should +1 or -1")
    else:
        dir_[0] = -1 if direction is -1 else +1

    in_place[0] = 1 if overwrite_x else 0

    # convert x to ndarray, ensure that strides are multiples of itemsize
    x_arr = PyArray_CheckFromAny(
        x, NULL, 0, 0,
        cnp.NPY_ARRAY_ELEMENTSTRIDES |
        cnp.NPY_ARRAY_ENSUREARRAY |
        cnp.NPY_ARRAY_NOTSWAPPED,
        NULL
    )

    if (<void *> x_arr) is NULL:
        raise ValueError("An input argument x is not an array-like object")

    if _datacopied(x_arr, x):
        in_place[0] = 1  # a copy was made, so we can work in place.

    xnd[0] = cnp.PyArray_NDIM(x_arr)  # tensor-rank of the array

    err = 0
    axis_[0] = PyLong_AsLong(axis)
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
            n_[0] = PyLong_AsLong(n)
        except:
            err = 1

    if not err:
        n_max = <long> cnp.PyArray_DIM(x_arr, axis_[0])
        if n_[0] < 1:
            err = 1
        elif n_[0] > n_max:
            in_place[0] = 1  # we must copy to pad and will work in-place
            x_arr = _pad_array(x_arr, n_[0], axis_[0], realQ)

    if err:
        raise ValueError(
            "Dimension n should be a positive integer not "
            "larger than the shape of the array along the chosen axis"
        )

    return x_arr


cdef cnp.ndarray _allocate_result(
    cnp.ndarray x_arr, long n_, long axis_, int f_type
):
    """
    An internal utility to allocate an empty array for output
    of not-in-place FFT.
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
    x_arr_is_fortran = cnp.PyArray_CHKFLAGS(x_arr, cnp.NPY_ARRAY_F_CONTIGUOUS)
    f_arr = <cnp.ndarray> cnp.PyArray_EMPTY(
        f_ndim, f_shape, <cnp.NPY_TYPES> f_type, x_arr_is_fortran
    )  # 0 for C-contiguous
    PyMem_Free(f_shape)

    return f_arr


cdef int _is_integral(object num):
    cdef long n
    cdef int _integral
    if num is None:
        return 0
    try:
        n = PyLong_AsLong(num)
        _integral = 1 if n > 0 else 0
    except:
        _integral = 0

    return _integral


def _get_element_strides(array):
    """Convert byte strides to element strides."""

    byte_strides = array.strides
    array_itemsize = array.itemsize
    return tuple(s // array_itemsize for s in byte_strides)


def _validate_out_array(out, x, out_dtype, axis=None, n=None):
    """Validate out keyword argument."""

    if type(out) is not np.ndarray:
        raise TypeError("return array must be of ArrayType")

    x_shape = list(x.shape)
    if axis is not None:
        x_shape[axis] = n
    if out.shape != tuple(x_shape):
        raise ValueError(
            "output array has wrong shape, expected (%s) got (%s)."
            % (tuple(x_shape), out.shape)
        )

    if out.dtype != out_dtype:
        raise TypeError(
            "Cannot cast 'fft' output from dtype(%s) to dtype(%s)."
            % (out_dtype, out.dtype)
        )


# this routine implements complex forward/backward FFT
# float/double inputs are not cast to complex, but are effectively
# treated as complexes with zero imaginary parts.
# All other types are cast to complex double.
def _c2c_fft1d_impl(
    x,
    n=None,
    axis=-1,
    overwrite_x=False,
    direction=+1,
    double fsc=1.0,
    out=None,
):
    """
    Uses MKL to perform 1D FFT on the input array x along the given axis.
    """
    cdef cnp.ndarray x_arr "x_arrayObject"
    cdef cnp.ndarray f_arr "f_arrayObject"
    cdef int xnd, n_max = 0, in_place, dir_
    cdef long n_, axis_
    cdef int x_type, f_type, status = 0
    cdef int ALL_HARMONICS = 1
    cdef char * c_error_msg = NULL
    cdef bytes py_error_msg
    cdef DftiCache *_cache

    x_arr = _process_arguments(x, n, axis, overwrite_x, direction,
                               &axis_, &n_, &in_place, &xnd, &dir_, 0)

    x_type = cnp.PyArray_TYPE(x_arr)

    if out is not None:
        in_place = 0
    elif x_type is cnp.NPY_CFLOAT or x_type is cnp.NPY_CDOUBLE:
        # we can operate in place if requested.
        if in_place:
            if not cnp.PyArray_ISONESEGMENT(x_arr):
                in_place = 0 if internal_overlap(x_arr) else 1
    elif x_type is cnp.NPY_FLOAT or x_type is cnp.NPY_DOUBLE:
        # to work in place we need to cast the input to complex,
        # which may be more expensive than creating the output using MKL
        in_place = 0
    else:
        # we must cast the input and allocate the output,
        # so we cast to complex double and operate in place
        try:
            x_arr = <cnp.ndarray> cnp.PyArray_FROM_OTF(
                x_arr, cnp.NPY_CDOUBLE,
                cnp.NPY_ARRAY_BEHAVED | cnp.NPY_ARRAY_ENSURECOPY
            )
        except:
            raise ValueError(
                "First argument must be a complex "
                "or real sequence of single or double precision"
            )
        x_type = cnp.PyArray_TYPE(x_arr)
        in_place = 1

    if in_place:
        _cache_capsule = _tls_dfti_cache_capsule()
        _cache = <DftiCache *>cpython.pycapsule.PyCapsule_GetPointer(
            _cache_capsule, capsule_name
        )
        if x_type is cnp.NPY_CDOUBLE:
            if dir_ < 0:
                status = cdouble_mkl_ifft1d_in(
                    x_arr, n_, <int> axis_, fsc, _cache
                )
            else:
                status = cdouble_mkl_fft1d_in(
                    x_arr, n_, <int> axis_, fsc, _cache
                )
        elif x_type is cnp.NPY_CFLOAT:
            if dir_ < 0:
                status = cfloat_mkl_ifft1d_in(
                    x_arr, n_, <int> axis_, fsc, _cache
                )
            else:
                status = cfloat_mkl_fft1d_in(
                    x_arr, n_, <int> axis_, fsc, _cache
                )
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
        if x_type is cnp.NPY_FLOAT or x_type is cnp.NPY_CFLOAT:
            f_type = cnp.NPY_CFLOAT
        else:
            f_type = cnp.NPY_CDOUBLE

        if out is None:
            f_arr = _allocate_result(x_arr, n_, axis_, f_type)
        else:
            out_dtype = np.dtype(cnp.PyArray_DescrFromType(f_type))
            _validate_out_array(out, x, out_dtype, axis=axis_, n=n_)
            # out array that is used in OneMKL c2c FFT must have the exact same
            # stride as input array. If not, we need to allocate a new array.
            # TODO: check to see if this condition can be relaxed
            if _get_element_strides(x) == _get_element_strides(out):
                f_arr = <cnp.ndarray> out
            else:
                f_arr = _allocate_result(x_arr, n_, axis_, f_type)

        # call out-of-place FFT
        _cache_capsule = _tls_dfti_cache_capsule()
        _cache = <DftiCache *>cpython.pycapsule.PyCapsule_GetPointer(
            _cache_capsule, capsule_name
        )
        if f_type is cnp.NPY_CDOUBLE:
            if x_type is cnp.NPY_DOUBLE:
                if dir_ < 0:
                    status = double_cdouble_mkl_ifft1d_out(
                        x_arr,
                        n_,
                        <int> axis_,
                        f_arr,
                        ALL_HARMONICS,
                        fsc,
                        _cache,
                    )
                else:
                    status = double_cdouble_mkl_fft1d_out(
                        x_arr,
                        n_,
                        <int> axis_,
                        f_arr,
                        ALL_HARMONICS,
                        fsc,
                        _cache,
                    )
            elif x_type is cnp.NPY_CDOUBLE:
                if dir_ < 0:
                    status = cdouble_cdouble_mkl_ifft1d_out(
                        x_arr, n_, <int> axis_, f_arr, fsc, _cache
                    )
                else:
                    status = cdouble_cdouble_mkl_fft1d_out(
                        x_arr, n_, <int> axis_, f_arr, fsc, _cache
                    )
        else:
            if x_type is cnp.NPY_FLOAT:
                if dir_ < 0:
                    status = float_cfloat_mkl_ifft1d_out(
                        x_arr,
                        n_,
                        <int> axis_,
                        f_arr,
                        ALL_HARMONICS,
                        fsc,
                        _cache,
                    )
                else:
                    status = float_cfloat_mkl_fft1d_out(
                        x_arr,
                        n_,
                        <int> axis_,
                        f_arr,
                        ALL_HARMONICS,
                        fsc,
                        _cache,
                    )
            elif x_type is cnp.NPY_CFLOAT:
                if dir_ < 0:
                    status = cfloat_cfloat_mkl_ifft1d_out(
                        x_arr, n_, <int> axis_, f_arr, fsc, _cache
                    )
                else:
                    status = cfloat_cfloat_mkl_fft1d_out(
                        x_arr, n_, <int> axis_, f_arr, fsc, _cache
                    )

        if (status):
            c_error_msg = mkl_dfti_error(status)
            py_error_msg = c_error_msg
            raise ValueError("Internal error occurred: {}".format(py_error_msg))

        if out is not None and f_arr is not out:
            out[...] = f_arr
            return out
        else:
            return f_arr


def _r2c_fft1d_impl(
    x, n=None, axis=-1, overwrite_x=False, double fsc=1.0, out=None
):
    """
    Uses MKL to perform 1D FFT on the real input array x along the given axis,
    producing complex output, but giving only half of the harmonics.

    cf. numpy.fft.rfft
    """
    cdef cnp.ndarray x_arr "x_arrayObject"
    cdef cnp.ndarray f_arr "f_arrayObject"
    cdef int xnd, in_place, dir_
    cdef long n_, axis_
    cdef int x_type, f_type, status, requirement
    cdef int HALF_HARMONICS = 0  # give only positive index harmonics
    cdef int direction = 1  # dummy, only used for the sake of arg-processing
    cdef char * c_error_msg = NULL
    cdef bytes py_error_msg
    cdef DftiCache *_cache

    x_arr = _process_arguments(x, n, axis, overwrite_x, direction,
                               &axis_, &n_, &in_place, &xnd, &dir_, 1)

    x_type = cnp.PyArray_TYPE(x_arr)

    if (
        x_type is cnp.NPY_CFLOAT
        or x_type is cnp.NPY_CDOUBLE
        or x_type is cnp.NPY_CLONGDOUBLE
    ):
        raise TypeError("1st argument must be a real sequence.")
    elif x_type is cnp.NPY_FLOAT or x_type is cnp.NPY_DOUBLE:
        pass
    else:
        # we must cast the input to doubles and allocate the output,
        try:
            requirement = cnp.NPY_ARRAY_BEHAVED | cnp.NPY_ARRAY_ENSURECOPY
            if x_type is cnp.NPY_LONGDOUBLE:
                requirement = requirement | cnp.NPY_ARRAY_FORCECAST
            x_arr = <cnp.ndarray> cnp.PyArray_FROM_OTF(
                x_arr, cnp.NPY_DOUBLE, requirement)
            x_type = cnp.PyArray_TYPE(x_arr)
        except:
            raise TypeError("1st argument must be a real sequence 2")

    # in_place is ignored here.
    # it can be done only if 2*(n_ // 2 + 1)  <= x_arr.shape[axis_] which is not
    # the common usage
    f_type = cnp.NPY_CFLOAT if x_type is cnp.NPY_FLOAT else cnp.NPY_CDOUBLE
    f_shape = n_ // 2 + 1
    if out is None:
        f_arr = _allocate_result(x_arr, f_shape, axis_, f_type)
    else:
        out_dtype = np.dtype(cnp.PyArray_DescrFromType(f_type))
        _validate_out_array(out, x, out_dtype, axis=axis_, n=f_shape)
        # out array that is used in OneMKL r2c FFT must have comparable strides
        # with input array. If not, we need to allocate a new array.
        # For r2c, out and input arrays have different size and strides cannot
        # be compared directly.
        # TODO: currently instead of this condition, we check both input
        # and output to be c_contig or f_contig, relax this condition
        c_contig = x.flags.c_contiguous and out.flags.c_contiguous
        f_contig = x.flags.f_contiguous and out.flags.f_contiguous
        if c_contig or f_contig:
            f_arr = <cnp.ndarray> out
        else:
            f_arr = _allocate_result(x_arr, f_shape, axis_, f_type)

    # call out-of-place FFT
    if x_type is cnp.NPY_FLOAT:
        _cache_capsule = _tls_dfti_cache_capsule()
        _cache = <DftiCache *>cpython.pycapsule.PyCapsule_GetPointer(
            _cache_capsule, capsule_name
        )
        status = float_cfloat_mkl_fft1d_out(
            x_arr, n_, <int> axis_, f_arr, HALF_HARMONICS, fsc, _cache
        )
    else:
        _cache_capsule = _tls_dfti_cache_capsule()
        _cache = <DftiCache *>cpython.pycapsule.PyCapsule_GetPointer(
            _cache_capsule, capsule_name
        )
        status = double_cdouble_mkl_fft1d_out(
            x_arr, n_, <int> axis_, f_arr, HALF_HARMONICS, fsc, _cache
        )

    if (status):
        c_error_msg = mkl_dfti_error(status)
        py_error_msg = c_error_msg
        raise ValueError(
            "Internal error occurred: {}".format(str(py_error_msg))
        )

    if out is not None and f_arr is not out:
        out[...] = f_arr
        return out
    else:
        return f_arr


def _c2r_fft1d_impl(
    x, n=None, axis=-1, overwrite_x=False, double fsc=1.0, out=None
):
    """
    Uses MKL to perform 1D FFT on the real/complex input array x along the
    given axis, producing real output.

    cf. numpy.fft.irfft
    """
    cdef cnp.ndarray x_arr "x_arrayObject"
    cdef cnp.ndarray f_arr "f_arrayObject"
    cdef int xnd, in_place, dir_, int_n
    cdef long n_, axis_
    cdef int x_type, f_type, status
    cdef int direction = 1  # dummy, only used for the sake of arg-processing
    cdef char * c_error_msg = NULL
    cdef bytes py_error_msg
    cdef DftiCache *_cache

    int_n = _is_integral(n)
    # nn gives the number elements along axis of the input that we use
    nn = (n // 2 + 1) if int_n and n > 0 else n
    x_arr = _process_arguments(x, nn, axis, overwrite_x, direction,
                               &axis_, &n_, &in_place, &xnd, &dir_, 0)
    n_ = 2*(n_ - 1)
    if int_n and (n % 2 == 1):
        n_ += 1

    x_type = cnp.PyArray_TYPE(x_arr)

    if x_type is cnp.NPY_CFLOAT or x_type is cnp.NPY_CDOUBLE:
        # we can operate in place if requested.
        if in_place:
            if not cnp.PyArray_ISONESEGMENT(x_arr):
                in_place = 0 if internal_overlap(x_arr) else 1
    else:
        # we must cast the input and allocate the output,
        # so we cast to complex double and operate in place
        if x_type is cnp.NPY_FLOAT:
            x_arr = <cnp.ndarray> cnp.PyArray_FROM_OTF(
                x_arr, cnp.NPY_CFLOAT, cnp.NPY_ARRAY_BEHAVED
            )
        else:
            x_arr = <cnp.ndarray> cnp.PyArray_FROM_OTF(
                x_arr, cnp.NPY_CDOUBLE, cnp.NPY_ARRAY_BEHAVED
            )
        x_type = cnp.PyArray_TYPE(x_arr)
        in_place = 1

    in_place = 0
    if in_place:
        # TODO: Provide in-place functionality
        pass
    else:
        f_type = cnp.NPY_FLOAT if x_type is cnp.NPY_CFLOAT else cnp.NPY_DOUBLE
        if out is None:
            f_arr = _allocate_result(x_arr, n_, axis_, f_type)
        else:
            out_dtype = np.dtype(cnp.PyArray_DescrFromType(f_type))
            _validate_out_array(out, x, out_dtype, axis=axis_, n=n_)
            # out array that is used in OneMKL c2r FFT must have comparable
            # strides with input array. If not, we need to allocate a new
            # array. For c2r, out and input arrays have different size and
            # strides cannot be compared directly.
            # TODO: currently instead of this condition, we check both input
            # and output to be c_contig or f_contig, relax this condition
            c_contig = x.flags.c_contiguous and out.flags.c_contiguous
            f_contig = x.flags.f_contiguous and out.flags.f_contiguous
            if c_contig or f_contig:
                f_arr = <cnp.ndarray> out
            else:
                f_arr = _allocate_result(x_arr, n_, axis_, f_type)

        # call out-of-place FFT
        if x_type is cnp.NPY_CFLOAT:
            _cache_capsule = _tls_dfti_cache_capsule()
            _cache = <DftiCache *>cpython.pycapsule.PyCapsule_GetPointer(
                _cache_capsule, capsule_name
            )
            status = cfloat_float_mkl_irfft_out(
                x_arr, n_, <int> axis_, f_arr, fsc, _cache
            )
        else:
            _cache_capsule = _tls_dfti_cache_capsule()
            _cache = <DftiCache *>cpython.pycapsule.PyCapsule_GetPointer(
                _cache_capsule, capsule_name
            )
            status = cdouble_double_mkl_irfft_out(
                x_arr, n_, <int> axis_, f_arr, fsc, _cache
            )

        if (status):
            c_error_msg = mkl_dfti_error(status)
            py_error_msg = c_error_msg
            raise ValueError(
                "Internal error occurred: {}".format(str(py_error_msg))
            )

        if out is not None and f_arr is not out:
            out[...] = f_arr
            return out
        else:
            return f_arr


def _direct_fftnd(
    x, overwrite_x=False, direction=+1, double fsc=1.0, out=None
):
    """Perform n-dimensional FFT over all axes"""
    cdef int err
    cdef cnp.ndarray x_arr "xxnd_arrayObject"
    cdef cnp.ndarray f_arr "ffnd_arrayObject"
    cdef int dir_, in_place, x_type, f_type

    if direction not in [-1, +1]:
        raise ValueError("Direction of FFT should +1 or -1")
    else:
        dir_ = -1 if direction is -1 else +1

    in_place = 1 if overwrite_x else 0

    # convert x to ndarray, ensure that strides are multiples of itemsize
    x_arr = PyArray_CheckFromAny(
        x, NULL, 0, 0,
        cnp.NPY_ARRAY_ELEMENTSTRIDES |
        cnp.NPY_ARRAY_ENSUREARRAY |
        cnp.NPY_ARRAY_NOTSWAPPED,
        NULL
    )

    if <void *> x_arr is NULL:
        raise ValueError("An input argument x is not an array-like object")

    if _datacopied(x_arr, x):
        in_place = 1  # a copy was made, so we can work in place.

    x_type = cnp.PyArray_TYPE(x_arr)
    if (
        x_type == cnp.NPY_CDOUBLE
        or x_type == cnp.NPY_CFLOAT
        or x_type == cnp.NPY_DOUBLE
        or x_type == cnp.NPY_FLOAT
    ):
        pass
    else:
        x_arr = <cnp.ndarray> cnp.PyArray_FROM_OTF(
            x_arr, cnp.NPY_CDOUBLE,
            cnp.NPY_ARRAY_BEHAVED | cnp.NPY_ARRAY_ENSURECOPY
        )
        x_type = cnp.PyArray_TYPE(x_arr)
        assert x_type == cnp.NPY_CDOUBLE
        in_place = 1

    if out is not None:
        in_place = 0

    if in_place:
        if x_type == cnp.NPY_CDOUBLE or x_type == cnp.NPY_CFLOAT:
            in_place = 1
        else:
            in_place = 0

    if in_place:
        if x_type == cnp.NPY_CDOUBLE:
            if dir_ == 1:
                err = cdouble_cdouble_mkl_fftnd_in(x_arr, fsc)
            else:
                err = cdouble_cdouble_mkl_ifftnd_in(x_arr, fsc)
        elif x_type == cnp.NPY_CFLOAT:
            if dir_ == 1:
                err = cfloat_cfloat_mkl_fftnd_in(x_arr, fsc)
            else:
                err = cfloat_cfloat_mkl_ifftnd_in(x_arr, fsc)
        else:
            raise ValueError("An input argument x is not complex type array")

        return x_arr
    else:
        if x_type == cnp.NPY_CDOUBLE or x_type == cnp.NPY_DOUBLE:
            f_type = cnp.NPY_CDOUBLE
        else:
            f_type = cnp.NPY_CFLOAT
        if out is None:
            f_arr = _allocate_result(x_arr, -1, 0, f_type)
        else:
            out_dtype = np.dtype(cnp.PyArray_DescrFromType(f_type))
            _validate_out_array(out, x, out_dtype)
            # out array that is used in OneMKL c2c FFT must have the exact same
            # stride as input array. If not, we need to allocate a new array.
            # TODO: check to see if this condition can be relaxed
            if _get_element_strides(x) == _get_element_strides(out):
                f_arr = <cnp.ndarray> out
            else:
                f_arr = _allocate_result(x_arr, -1, 0, f_type)

        if x_type == cnp.NPY_CDOUBLE:
            if dir_ == 1:
                err = cdouble_cdouble_mkl_fftnd_out(x_arr, f_arr, fsc)
            else:
                err = cdouble_cdouble_mkl_ifftnd_out(x_arr, f_arr, fsc)
        elif x_type == cnp.NPY_CFLOAT:
            if dir_ == 1:
                err = cfloat_cfloat_mkl_fftnd_out(x_arr, f_arr, fsc)
            else:
                err = cfloat_cfloat_mkl_ifftnd_out(x_arr, f_arr, fsc)
        elif x_type == cnp.NPY_DOUBLE:
            if dir_ == 1:
                err = double_cdouble_mkl_fftnd_out(x_arr, f_arr, fsc)
            else:
                err = double_cdouble_mkl_ifftnd_out(x_arr, f_arr, fsc)
        elif x_type == cnp.NPY_FLOAT:
            if dir_ == 1:
                err = float_cfloat_mkl_fftnd_out(x_arr, f_arr, fsc)
            else:
                err = float_cfloat_mkl_ifftnd_out(x_arr, f_arr, fsc)
        else:
            raise ValueError("An input argument x is not complex type array")

        if out is not None and f_arr is not out:
            out[...] = f_arr
            return out
        else:
            return f_arr


# ========================= deprecated functions ==============================
cdef object _rc_to_rr(cnp.ndarray rc_arr, int n, int axis, int xnd, int x_type):
    cdef object res
    inp = <object>rc_arr

    slice_ = [slice(None, None, None)] * xnd
    sl_0 = list(slice_)
    sl_0[axis] = 0

    sl_1 = list(slice_)
    sl_1[axis] = 1
    if (inp.flags["C"] and inp.strides[axis] == inp.itemsize):
        res = inp
        res = res.view(
            dtype=np.single if x_type == cnp.NPY_FLOAT else np.double
        )
        res[tuple(sl_1)] = res[tuple(sl_0)]

        slice_[axis] = slice(1, n + 1, None)

        return res[tuple(slice_)]
    else:
        res_shape = list(inp.shape)
        res_shape[axis] = n
        res = np.empty(
            tuple(res_shape),
            dtype = np.single if x_type == cnp.NPY_FLOAT else np.double,
        )

        res[tuple(sl_0)] = inp[tuple(sl_0)].real
        sl_dst_real = list(slice_)
        sl_dst_real[axis] = slice(1, None, 2)
        sl_src_real = list(slice_)
        sl_src_real[axis] = slice(1, None, None)
        res[tuple(sl_dst_real)] = inp[tuple(sl_src_real)].real
        sl_dst_imag = list(slice_)
        sl_dst_imag[axis] = slice(2, None, 2)
        sl_src_imag = list(slice_)
        sl_src_imag[axis] = slice(
            1, inp.shape[axis] if (n & 1) else inp.shape[axis] - 1, None
        )
        res[tuple(sl_dst_imag)] = inp[tuple(sl_src_imag)].imag

        return res[tuple(slice_)]


cdef object _rr_to_rc(cnp.ndarray rr_arr, int n, int axis, int xnd, int x_type):

    inp = <object> rr_arr

    rc_shape = list(inp.shape)
    rc_shape[axis] = (n // 2 + 1)
    rc_shape = tuple(rc_shape)

    rc_dtype = np.cdouble if x_type == cnp.NPY_DOUBLE else np.csingle
    rc = np.empty(rc_shape, dtype=rc_dtype, order="C")

    slice_ = [slice(None, None, None)] * xnd
    sl_src_real = list(slice_)
    sl_src_imag = list(slice_)
    sl_src_real[axis] = slice(1, n, 2)
    sl_src_imag[axis] = slice(2, n, 2)

    sl_dest_real = list(slice_)
    sl_dest_real[axis] = slice(1, None, None)
    sl_dest_imag = list(slice_)
    sl_dest_imag[axis] = slice(1, (n+1)//2, None)

    sl_0 = list(slice_)
    sl_0[axis] = 0

    rc_real = rc.real
    rc_imag = rc.imag

    rc_real[tuple(sl_dest_real)] = inp[tuple(sl_src_real)]
    rc_imag[tuple(sl_dest_imag)] = inp[tuple(sl_src_imag)]
    rc_real[tuple(sl_0)] = inp[tuple(sl_0)]
    rc_imag[tuple(sl_0)] = 0
    if (n & 1 == 0):
        sl_last = list(slice_)
        sl_last[axis] = -1
        rc_imag[tuple(sl_last)] = 0

    return rc


def _rr_fft1d_impl(x, n=None, axis=-1, overwrite_x=False, double fsc=1.0):
    """
    Uses MKL to perform real packed 1D FFT on the input array x
    along the given axis.

    This done by using rfft and post-processing the result.
    Thus overwrite_x is effectively discarded.

    Functionally equivalent to scipy.fftpack.rfft
    """
    cdef cnp.ndarray x_arr "x_arrayObject"
    cdef cnp.ndarray f_arr "f_arrayObject"
    cdef int xnd, in_place, dir_
    cdef long n_, axis_
    cdef int HALF_HARMONICS = 0  # give only positive index harmonics
    cdef int x_type, status, f_type
    cdef char * c_error_msg = NULL
    cdef bytes py_error_msg
    cdef DftiCache *_cache

    x_arr = _process_arguments(x, n, axis, overwrite_x, <object>(+1),
                               &axis_, &n_, &in_place, &xnd, &dir_, 1)

    x_type = cnp.PyArray_TYPE(x_arr)

    if x_type is cnp.NPY_FLOAT or x_type is cnp.NPY_DOUBLE:
        in_place = 0
    elif x_type is cnp.NPY_CFLOAT or x_type is cnp.NPY_CDOUBLE:
        raise TypeError("1st argument must be a real sequence")
    else:
        try:
            x_arr = <cnp.ndarray> cnp.PyArray_FROM_OTF(
                x_arr, cnp.NPY_DOUBLE,
                cnp.NPY_ARRAY_BEHAVED | cnp.NPY_ARRAY_ENSURECOPY
            )
        except:
            raise TypeError("1st argument must be a real sequence")
        x_type = cnp.PyArray_TYPE(x_arr)
        in_place = 0

    f_type = cnp.NPY_CFLOAT if x_type is cnp.NPY_FLOAT else cnp.NPY_CDOUBLE
    f_arr = _allocate_result(x_arr, n_ // 2 + 1, axis_, f_type)

    _cache_capsule = _tls_dfti_cache_capsule()
    _cache = <DftiCache *>cpython.pycapsule.PyCapsule_GetPointer(
        _cache_capsule, capsule_name
    )
    if x_type is cnp.NPY_DOUBLE:
        status = double_cdouble_mkl_fft1d_out(
            x_arr, n_, <int> axis_, f_arr, HALF_HARMONICS, fsc, _cache
        )
    else:
        status = float_cfloat_mkl_fft1d_out(
            x_arr, n_, <int> axis_, f_arr, HALF_HARMONICS, fsc, _cache
        )

    if (status):
        c_error_msg = mkl_dfti_error(status)
        py_error_msg = c_error_msg
        raise ValueError("Internal error occurred: {}".format(py_error_msg))

    # post-process and return
    return _rc_to_rr(f_arr, n_, axis_, xnd, x_type)


def _rr_ifft1d_impl(x, n=None, axis=-1, overwrite_x=False, double fsc=1.0):
    """
    Uses MKL to perform real packed 1D FFT on the input array x along
    the given axis.

    This done by using rfft and post-processing the result.
    Thus overwrite_x is effectively discarded.

    Functionally equivalent to scipy.fftpack.irfft
    """
    cdef cnp.ndarray x_arr "x_arrayObject"
    cdef cnp.ndarray f_arr "f_arrayObject"
    cdef int xnd, in_place, dir_
    cdef long n_, axis_
    cdef int x_type, rc_type, status
    cdef char * c_error_msg = NULL
    cdef bytes py_error_msg
    cdef DftiCache *_cache

    x_arr = _process_arguments(x, n, axis, overwrite_x, <object>(-1),
                               &axis_, &n_, &in_place, &xnd, &dir_, 1)

    x_type = cnp.PyArray_TYPE(x_arr)

    if x_type is cnp.NPY_FLOAT or x_type is cnp.NPY_DOUBLE:
        pass
    else:
        # we must cast the input and allocate the output,
        # so we cast to complex double and operate in place
        try:
            x_arr = <cnp.ndarray> cnp.PyArray_FROM_OTF(
                x_arr, cnp.NPY_DOUBLE,
                cnp.NPY_ARRAY_BEHAVED | cnp.NPY_ARRAY_ENSURECOPY
            )
        except:
            raise ValueError(
                "First argument should be a real "
                "or a complex sequence of single or double precision"
            )
        x_type = cnp.PyArray_TYPE(x_arr)
        in_place = 1

    # need to convert this into complex array
    rc_obj = _rr_to_rc(x_arr, n_, axis_, xnd, x_type)
    rc_arr = <cnp.ndarray> rc_obj

    rc_type = cnp.NPY_CFLOAT if x_type is cnp.NPY_FLOAT else cnp.NPY_CDOUBLE
    in_place = False
    if in_place:
        f_arr = x_arr
    else:
        f_arr = _allocate_result(x_arr, n_, axis_, x_type)

    # call out-of-place FFT
    if rc_type is cnp.NPY_CFLOAT:
        _cache_capsule = _tls_dfti_cache_capsule()
        _cache = <DftiCache *>cpython.pycapsule.PyCapsule_GetPointer(
            _cache_capsule, capsule_name
        )
        status = cfloat_float_mkl_irfft_out(
            rc_arr, n_, <int> axis_, f_arr, fsc, _cache
        )
    elif rc_type is cnp.NPY_CDOUBLE:
        _cache_capsule = _tls_dfti_cache_capsule()
        _cache = <DftiCache *>cpython.pycapsule.PyCapsule_GetPointer(
            _cache_capsule, capsule_name
        )
        status = cdouble_double_mkl_irfft_out(
            rc_arr, n_, <int> axis_, f_arr, fsc, _cache
        )
    else:
        raise ValueError(
            "Internal mkl_fft error occurred: Unrecognized rc_type"
        )

    if (status):
        c_error_msg = mkl_dfti_error(status)
        py_error_msg = c_error_msg
        raise ValueError(
            "Internal error occurred: {}".format(str(py_error_msg))
        )

    return f_arr


def rfftpack(x, n=None, axis=-1, overwrite_x=False, fwd_scale=1.0):
    """Packed real-valued harmonics of FFT of a real sequence x"""
    return _rr_fft1d_impl(
        x, n=n, axis=axis, overwrite_x=overwrite_x, fsc=fwd_scale
    )


def irfftpack(x, n=None, axis=-1, overwrite_x=False, fwd_scale=1.0):
    """IFFT of a real sequence, takes packed real-valued harmonics of FFT"""
    return _rr_ifft1d_impl(
        x, n=n, axis=axis, overwrite_x=overwrite_x, fsc=fwd_scale
    )

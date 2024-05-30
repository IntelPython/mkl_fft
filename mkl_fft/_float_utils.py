#!/usr/bin/env python
# Copyright (c) 2017-2024, Intel Corporation
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

__all__ = ['__upcast_float16_array', '__downcast_float128_array', '__supported_array_or_not_implemented']

def __upcast_float16_array(x):
    """
    Used in _scipy_fft to upcast float16 to float32, 
    instead of float64, as mkl_fft would do"""
    if hasattr(x, "dtype"):
        xdt = x.dtype
        if xdt == np.half:
            # no half-precision routines, so convert to single precision
            return np.asarray(x, dtype=np.float32)
        if xdt == np.longdouble and not xdt == np.float64:
            raise ValueError("type %s is not supported" % xdt)
    if not isinstance(x, np.ndarray):
        __x = np.asarray(x)
        xdt = __x.dtype
        if xdt == np.half:
            # no half-precision routines, so convert to single precision
            return np.asarray(__x, dtype=np.float32)
        if xdt == np.longdouble and not xdt == np.float64:
            raise ValueError("type %s is not supported" % xdt)
        return __x
    return x


def __downcast_float128_array(x):
    """
    Used in _numpy_fft to unsafely downcast float128/complex256 to 
    complex128, instead of raising an error"""
    if hasattr(x, "dtype"):
        xdt = x.dtype
        if xdt == np.longdouble and not xdt == np.float64:
            return np.asarray(x, dtype=np.float64)
        elif xdt == np.clongdouble and not xdt == np.complex128:
            return np.asarray(x, dtype=np.complex128)
    if not isinstance(x, np.ndarray):
        __x = np.asarray(x)
        xdt = __x.dtype
        if xdt == np.longdouble and not xdt == np.float64:
            return np.asarray(x, dtype=np.float64)
        elif xdt == np.clongdouble and not xdt == np.complex128:
            return np.asarray(x, dtype=np.complex128)
        return __x
    return x


def __supported_array_or_not_implemented(x):
    """
    Used in _scipy_fft_backend to convert array to float32,
    float64, complex64, or complex128 type or return NotImplemented
    """
    __x = np.asarray(x)
    black_list = [np.half]
    if hasattr(np, 'float128'):
        black_list.append(np.float128)
    if hasattr(np, 'complex256'):
        black_list.append(np.complex256)
    if __x.dtype in black_list:
        return NotImplemented
    return __x

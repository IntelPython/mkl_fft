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

from __future__ import division, absolute_import, print_function

import numpy as np
from numpy.testing import (
        TestCase, run_module_suite, assert_, assert_raises, assert_equal,
        assert_warns, assert_allclose)
from numpy import random as rnd
import sys
import warnings

import mkl_fft
import numpy.fft.fftpack as np_fft

reps_64 = (2**11)*np.finfo(np.float64).eps
reps_32 = (2**11)*np.finfo(np.float32).eps
atol_64 = (2**8)*np.finfo(np.float64).eps
atol_32 = (2**8)*np.finfo(np.float32).eps

def _get_rtol_atol(x):
    dt = x.dtype
    if dt == np.double or dt == np.complex128:
        return reps_64, atol_64
    elif dt == np.single or dt == np.complex64:
        return reps_32, atol_32
    else:
        assert (dt == np.double or dt == np.complex128 or dt == np.single or dt == np.complex64), "Unexpected dtype {}".format(dt)
        return reps_64, atol_64


class Test_mklfft_matrix(TestCase):
    def setUp(self):
        rnd.seed(123456)
        self.md = rnd.randn(256, 256)
        self.mf = self.md.astype(np.float32)
        self.mz = rnd.randn(256, 256*2).view(np.complex128)
        self.mc = self.mz.astype(np.complex64)

    def test_matrix1(self):
        """fftn equals repeated fft"""
        for ar in [self.md, self.mz, self.mf, self.mc]:
            r_tol, a_tol = _get_rtol_atol(ar)
            d = ar.copy()
            t1 = mkl_fft.fftn(d)
            t2 = mkl_fft.fft(mkl_fft.fft(d, axis=0), axis=1)
            t3 = mkl_fft.fft(mkl_fft.fft(d, axis=1), axis=0)
            assert_allclose(t1, t2, rtol=r_tol, atol=a_tol, err_msg = "failed test for dtype {}, max abs diff: {}".format(d.dtype, np.max(np.abs(t1-t2))))
            assert_allclose(t1, t3, rtol=r_tol, atol=a_tol, err_msg = "failed test for dtype {}, max abs diff: {}".format(d.dtype, np.max(np.abs(t1-t3))))

    def test_matrix2(self):
        """ifftn(fftn(x)) is x"""
        for ar in [self.md, self.mz, self.mf, self.mc]:
            d = ar.copy()
            r_tol, a_tol = _get_rtol_atol(d)
            t = mkl_fft.ifftn(mkl_fft.fftn(d))
            assert_allclose(d, t, rtol=r_tol, atol=a_tol, err_msg = "failed test for dtype {}, max abs diff: {}".format(d.dtype, np.max(np.abs(d-t))))

    def test_matrix3(self):
        """fftn(ifftn(x)) is x"""
        for ar in [self.md, self.mz, self.mf, self.mc]:
            d = ar.copy()
            r_tol, a_tol = _get_rtol_atol(d)
            t = mkl_fft.fftn(mkl_fft.ifftn(d))
            assert_allclose(d, t, rtol=r_tol, atol=a_tol, err_msg = "failed test for dtype {}, max abs diff: {}".format(d.dtype, np.max(np.abs(d-t))))


    def test_matrix4(self):
        """fftn of strided array is same as fftn of a contiguous copy"""
        for ar in [self.md, self.mz, self.mf, self.mc]:
            r_tol, a_tol = _get_rtol_atol(ar)
            d_strided = ar[::2,::2]
            d_contig = d_strided.copy()
            t_strided = mkl_fft.fftn(d_strided)
            t_contig = mkl_fft.fftn(d_contig)
            assert_allclose(t_strided, t_contig, rtol=r_tol, atol=a_tol)


class Test_Regressions(TestCase):

    def setUp(self):
        rnd.seed(123456)
        self.ad = rnd.randn(32, 17, 23)
        self.af = self.ad.astype(np.float32)
        self.az = rnd.randn(32, 17, 23*2).view(np.complex128)
        self.ac = self.az.astype(np.complex64)

    def test_cf_contig(self):
        """fft of F-contiguous array is the same as of C-contiguous with same data"""
        for ar in [self.ad, self.af, self.az, self.ac]:
            r_tol, a_tol = _get_rtol_atol(ar)
            d_ccont = ar.copy()
            d_fcont = np.asfortranarray(d_ccont)
            f1 = mkl_fft.fft(d_ccont)
            f2 = mkl_fft.fft(d_fcont)
            assert_allclose(f1, f2, rtol=r_tol, atol=a_tol)

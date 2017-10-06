/*
 Copyright (c) 2017, Intel Corporation

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

     * Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
     * Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.
     * Neither the name of Intel Corporation nor the names of its contributors
       may be used to endorse or promote products derived from this software
       without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* Complex input, in-place */
extern int cdouble_mkl_fft1d_in(PyArrayObject*, npy_intp, int);
extern int cfloat_mkl_fft1d_in(PyArrayObject*, npy_intp, int);
extern int cdouble_mkl_ifft1d_in(PyArrayObject*, npy_intp, int);
extern int cfloat_mkl_ifft1d_in(PyArrayObject*, npy_intp, int);

/* Complex input/output, out-of-place */
extern int cfloat_cfloat_mkl_fft1d_out(PyArrayObject*, npy_intp, int, PyArrayObject*);
extern int cdouble_cdouble_mkl_fft1d_out(PyArrayObject*, npy_intp, int, PyArrayObject*);
extern int cfloat_cfloat_mkl_ifft1d_out(PyArrayObject*, npy_intp, int, PyArrayObject*);
extern int cdouble_cdouble_mkl_ifft1d_out(PyArrayObject*, npy_intp, int, PyArrayObject*);

/* Real input, complex output, out-of-place */
extern int float_cfloat_mkl_fft1d_out(PyArrayObject*, npy_intp, int, PyArrayObject*, int);
extern int double_cdouble_mkl_fft1d_out(PyArrayObject*, npy_intp, int, PyArrayObject*, int);
extern int float_cfloat_mkl_ifft1d_out(PyArrayObject*, npy_intp, int, PyArrayObject*, int);
extern int double_cdouble_mkl_ifft1d_out(PyArrayObject*, npy_intp, int, PyArrayObject*, int);

/* Real input, real output, in-place */
extern int float_mkl_rfft_in(PyArrayObject*, npy_intp, int);
extern int float_mkl_irfft_in(PyArrayObject*, npy_intp, int);

extern int double_mkl_rfft_in(PyArrayObject*, npy_intp, int);
extern int double_mkl_irfft_in(PyArrayObject*, npy_intp, int);

/* Real input, real output, out-of-place */
extern int float_float_mkl_rfft_out(PyArrayObject*, npy_intp, int, PyArrayObject*);
extern int float_float_mkl_irfft_out(PyArrayObject*, npy_intp, int, PyArrayObject*);

extern int double_double_mkl_rfft_out(PyArrayObject*, npy_intp, int, PyArrayObject*);
extern int double_double_mkl_irfft_out(PyArrayObject*, npy_intp, int, PyArrayObject*);

/* Complex input. real output, out-of-place */
extern int cdouble_double_mkl_irfft_out(PyArrayObject*, npy_intp, int, PyArrayObject*);
extern int cfloat_float_mkl_irfft_out(PyArrayObject*, npy_intp, int, PyArrayObject*);

/* Complex, ND, in-place */
extern int cdouble_cdouble_mkl_fftnd_in(PyArrayObject*);
extern int cdouble_cdouble_mkl_ifftnd_in(PyArrayObject*);
extern int cfloat_cfloat_mkl_fftnd_in(PyArrayObject*);
extern int cfloat_cfloat_mkl_ifftnd_in(PyArrayObject*);

extern int cdouble_cdouble_mkl_fftnd_out(PyArrayObject*, PyArrayObject*);
extern int cdouble_cdouble_mkl_ifftnd_out(PyArrayObject*, PyArrayObject*);
extern int cfloat_cfloat_mkl_fftnd_out(PyArrayObject*, PyArrayObject*);
extern int cfloat_cfloat_mkl_ifftnd_out(PyArrayObject*, PyArrayObject*);

extern int float_cfloat_mkl_fftnd_out(PyArrayObject*, PyArrayObject*);
extern int float_cfloat_mkl_ifftnd_out(PyArrayObject*, PyArrayObject*);
extern int double_cdouble_mkl_fftnd_out(PyArrayObject*, PyArrayObject*);
extern int double_cdouble_mkl_ifftnd_out(PyArrayObject*, PyArrayObject*);

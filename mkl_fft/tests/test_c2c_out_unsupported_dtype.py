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

"""
Regression test for a silent-garbage-output bug in ``_c2c_fft1d_impl``
(mkl_fft/_pydfti.pyx).

``mkl_fft.fft``/``mkl_fft.ifft`` accept input of any dtype and, for dtypes
other than float32/float64/complex64/complex128, cast the input to
complex128 before computing the transform (cf. test_vector12 and
test_vector7/8 in test_fft1d.py, which exercise this cast when ``out`` is
not given).

Before the fix, that cast only happened when ``out`` was *not* passed: the
original code checked ``if out is not None: in_place = 0`` *before* looking
at ``x_type`` at all, so the branch that performs
``PyArray_FROM_OTF(..., NPY_CDOUBLE, ...)`` was skipped whenever the caller
supplied ``out``. ``x_type`` was then left as the unsupported dtype's type
code, which the dispatch further down (``if x_type is NPY_DOUBLE: ... elif
x_type is NPY_CDOUBLE: ...``) does not recognize, so none of its branches
ran and the ``status`` variable kept its initial value of 0. Because a
"success" status was reported without MKL ever having been called, the
function returned ``out`` unmodified -- whatever uninitialized values
``np.empty`` happened to produce -- instead of raising an error or
computing the transform.
"""

import numpy as np
import pytest

import mkl_fft


@pytest.mark.parametrize("dt", ["i4", "i8", "f2"])
@pytest.mark.parametrize("func, npfunc", [("fft", "fft"), ("ifft", "ifft")])
def test_c2c_out_with_unsupported_input_dtype(dt, func, npfunc):
    """fft/ifft with `out=` must compute the real transform for dtypes
    that require an internal cast to complex128 (e.g. integer, float16),
    not silently return the uninitialized `out` buffer."""
    x = np.arange(1, 17, dtype=dt)
    expected = getattr(np.fft, npfunc)(x.astype(np.float64))

    # fill `out` with a sentinel value that could never, by chance, match
    # the expected FFT output, so a correctness regression is detected
    # reliably rather than only "most of the time".
    out = np.full(x.shape, -1 - 1j, dtype=np.complex128)
    result = getattr(mkl_fft, func)(x, out=out)

    assert result is out
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)

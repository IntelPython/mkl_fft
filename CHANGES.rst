=================
mkl_fft changelog
=================

1.3.13
======

migrate from :code:`setup.py` to :code:`pyproject.toml`

includes support in virtual environment out of the box

the original :code:`mkl_fft.rfft` and :code:`mkl_fft.irfft` are renamed to :code:`mkl_fft.rfftpack` and :code:`mkl_fft.irfftpack`,
since they replicate the behavior from the deprectaed :code:`scipy.fftpack` module.

:code:`mkl_fft.rfft_numpy`, :code:`mkl_fft.irfft_numpy`, :code:`mkl_fft.rfft2_numpy`, :code:`mkl_fft.irfft2_numpy`,
:code:`mkl_fft.rfftn_numpy`, and :code:`mkl_fft.irfftn_numpy` are renamed to :code:`mkl_fft.rfft`, :code:`mkl_fft.irfft`,
:code:`mkl_fft.rfft2`, :code:`mkl_fft.irfft2`, `mkl_fft.rfftn`, and :code:`mkl_fft.irfftn`, respectively.
(consistent with :code:`numpy.fft` and :code:`scipy.fft` modules)

file :code:`_scipy_fft_backend.py` is renamed to :code:`_scipy_fft.py` since it replicates :code:`scipy.fft` module
(similar to file :code:`_numpy_fft.py` which replicates :code:`numpy.fft` module)

1.3.11
======

Bugfix release, resolving gh-109 and updating installation instructions


1.3.10
======

Bugfix release, resolving leftover uses of NumPy attributes removed in NumPy 2.0 that break
test suite run on Windows.

1.3.9
=====

Updated code and build system to support NumPy 2.0

1.3.8
=====

Added vendored :code:`conv_template.py` from NumPy's distutils submodule to enable building of :code:`mkl_fft` with
NumPy >=1.25 and Python 3.12

1.3.7
=====

Updated build system away from removed in NumPy 1.25 numpy.distutils module.

Transitioned to Cython 3.0.


1.3.0
=====

Updated numpy interface to support new in NumPy 1.20 supported values of norm keyword, such as "forward" and "backward".
To enable this, :code:`mkl_fft` functions now support `forward_scale` parameter that defaults to 1.

Fixed issue #48.

1.2.1
=====

Includes bug fix #54

1.2.0
=====

Due to removal of deprecated real-to-real FFT with :code:`DFTI_CONJUGATE_EVEN_STORAGE=DFTI_COMPLEX_REAL` and
:code:`DFTI_PACKED_FORMAT=DFTI_PACK` from Intel(R) Math Kernel Library, reimplemented :code:`mkl_fft.rfft` and
:code:`mkl_fft.irfft` to use real-to-complex functionality with subsequent copying to rearange the transform as expected
of :code:`mkl_fft.rfft`, with the associated performance penalty. The use of the real-to-complex
transform improves multi-core utilization which may offset the performance loss incurred due to copying.


1.1.0
=====

Added :code:`scipy.fft` backend, see #42. Fixed #46.


.. code-block:: python

    Python 3.7.5 (default, Nov 23 2019, 04:02:01)
    Type 'copyright', 'credits' or 'license' for more information
    IPython 7.11.1 -- An enhanced Interactive Python. Type '?' for help.

    In [1]: import numpy as np, mkl_fft, mkl_fft._scipy_fft as mkl_be, scipy, scipy.fft, mkl

    In [2]: mkl.verbose(1)
    Out[2]: True

    In [3]: x = np.random.randn(8*7).reshape((7, 8))
    ...: with scipy.fft.set_backend(mkl_be, only=True):
    ...:     ff = scipy.fft.fft2(x, workers=4)
    ...: ff2 = scipy.fft.fft2(x)
    MKL_VERBOSE Intel(R) MKL 2020.0 Product build 20191102 for Intel(R) 64 architecture Intel(R) Advanced Vector Extensions 2 (Intel(R) AVX2) enabled processors, Lnx 2.40GHz intel_thread
    MKL_VERBOSE FFT(drfo7:8:8x8:1:1,bScale:0.0178571,tLim:1,desc:0x5629ad31b800) 24.85ms CNR:OFF Dyn:1 FastMM:1 TID:0  NThr:16,FFT:4

    In [4]: np.allclose(ff, ff2)
    Out[4]: True


1.0.15
======

Changed tests to not compare against numpy fft, as this broke due to renaming of :code:`np.fft.pocketfft` to
:code:`np.fft._pocketfft`. Instead compare against naive realization of 1D FFT as a sum.

Setup script is now aware of :code:`MKLROOT` environment variable. If unset, NumPy's mkl_info will be queried.


1.0.14
======

Fixed unreferenced bug in :code:`irfftn_numpy`, and adjusted NumPy interfaces to change to pocketfft in NumPy 1.17


1.0.13
======

Issue #39 fixed (memory leak with complex FFT on real arrays)


1.0.12
======
Issue #37 fixed.

Inhibited vectorization of short loops computing pointer to memory referenced by a multi-iterator by Intel (R) C Compiler,
improving performance of ND :code:`fft` and :code:`ifft` on real input arrays.


1.0.11
======
Improvement for performance of ND `fft` on real input arrays by inlining multi-iterators.
This particularly benefits performance of mkl_fft built with Intel (R) C Compiler.


1.0.10
======
Fix for issue #29.


1.0.7
=====
Improved exception message raised if MKL is signalling an error. The message now includes MKL's own description of the exception.
This partially improves #24.

Improved argument validation for ND transforms aligning with scipy 1.2.0

1.0.6
=====

Fixed issues #21, and addressed NumPy 1.15 deprecation warnings from using lists instead of tuples to specify multiple slices.

1.0.5
=====

Fixed issues #7, #17, #18.
Consolidated version specification into a single file :code:`mkl_fft/_version.py`.

1.0.4
=====

Added CHANGES.rst. Fixed issue #11 by using lock around calls to 1D FFT routines.

1.0.3
=====

This is a bug fix release.

It fixes issues #9, and #13.

As part of fixing issue #13, out-of-place 1D FFT calls such as :code:`fft`, :code:`ifft`, :code:`rfft_numpy`
and :code:`irfftn_numpy` will allocate Fortran layout array for the output is the input is a Fotran array.


1.0.2
=====

Minor update of :code:`mkl_fft`, reflecting renaming of :code:`numpy.core.multiarray_tests` module to
:code:`numpy.core._multiarray_tests` as well as fixing #4.


1.0.1
=====

Bug fix release.

1.0.0
=====

Initial release of :code:`mkl_fft`.

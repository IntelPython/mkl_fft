=================
mkl_fft changelog
=================

1.0.13
======

Issue #39 fixed (memory leak with complex FFT on real arrays)


1.0.12
======
Issue #37 fixed.

Inhibited vectorization of short loops computing pointer to memory referenced by a multi-iterator by Intel (R) C Compiler, improving
performance of ND `fft` and `ifft` on real input arrays.


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
Consolidated version specification into a single file `mkl_fft/_version.py`.

1.0.4
=====

Added CHANGES.rst. Fixed issue #11 by using lock around calls to 1D FFT routines.

1.0.3
=====

This is a bug fix release.

It fixes issues #9, and #13.

As part of fixing issue #13, out-of-place 1D FFT calls such as `fft`, `ifft`, `rfft_numpy` and `irfftn_numpy` will allocate Fortran layout array for the output is the input is a Fotran array.


1.0.2
=====

Minor update of `mkl_fft`, reflecting renaming of `numpy.core.multiarray_tests` module to `numpy.core._multiarray_tests` as well as fixing #4.


1.0.1
=====

Bug fix release.

1.0.0
=====

Initial release of `mkl_fft`.

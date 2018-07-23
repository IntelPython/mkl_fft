=================
mkl_fft changelog
=================

1.0.4
=====

Added CHANGES.rst. Fixed issue #11 by using lock around calls to 1D FFT routines.

1.0.3
====

This is a bug fix release.

It fixes issues #9, and #13.

As part of fixing issue #13, out-of-place 1D FFT calls such as `fft`, `ifft`, `rfft_numpy` and `irfftn_numpy` will allocate Fortran layout array for the output is the input is a Fotran array.


1.0.2
====

Minor update of `mkl_fft`, reflecting renaming of `numpy.core.multiarray_tests` module to `numpy.core._multiarray_tests` as well as fixing #4.


1.0.1
====

Bug fix release.

1.0.0
====

Initial release of `mkl_fft`.

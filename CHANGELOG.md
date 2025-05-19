# changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [dev] (MM/DD/YY)

### Added
* Added Hermitian FFT functions to SciPy interface `mkl_fft.interfaces.scipy_fft`: `hfft`, `ihfft`, `hfftn`, `ihfftn`, `hfft2`, and `ihfft2` [gh-161](https://github.com/IntelPython/mkl_fft/pull/161)
* Added support for `out` kwarg to all FFT functions in `mkl_fft` and `mkl_fft.interfaces.numpy_fft` [gh-157](https://github.com/IntelPython/mkl_fft/pull/157)
* Added `fftfreq`, `fftshift`, `ifftshift`, and `rfftfreq` to both NumPy and SciPy interfaces [gh-179](https://github.com/IntelPython/mkl_fft/pull/179)

### Changed
* NumPy interface `mkl_fft.interfaces.numpy_fft` is aligned with numpy-2.x.x [gh-139](https://github.com/IntelPython/mkl_fft/pull/139), [gh-157](https://github.com/IntelPython/mkl_fft/pull/157)
* To set `mkl_fft` as the backend for SciPy is only possible through `mkl_fft.interfaces.scipy_fft` [gh-179](https://github.com/IntelPython/mkl_fft/pull/179)
* SciPy interface `mkl_fft.interfaces.scipy_fft` uses the same function from SciPy for handling `s` and `axes` for N-D FFTs [gh-181](https://github.com/IntelPython/mkl_fft/pull/181)

### Fixed
* Fixed an issue for calling `mkl_fft.interfaces.numpy.fftn` with an empty axes [gh-139](https://github.com/IntelPython/mkl_fft/pull/139)
* Fixed an issue for calling `mkl_fft.interfaces.numpy.fftn` with a zero-size array [gh-139](https://github.com/IntelPython/mkl_fft/pull/139)
* Fixed inconsistency of input and output arrays dtype for `irfft` function [gh-180](https://github.com/IntelPython/mkl_fft/pull/180)
* Fixed issues with `set_workers` function in SciPy interface `mkl_fft.interfaces.scipy_fft` [gh-183](https://github.com/IntelPython/mkl_fft/pull/183)

## [1.3.14] (04/10/2025)

resolves gh-152 by adding an explicit `mkl-service` dependency to `mkl-fft` when building the wheel
to ensure the `mkl` Python module is always available at runtime

resolves gh-115, gh-116, gh-119, gh-135

## [1.3.13] (03/25/2025)

Supported python versions are 3.9, 3.10, 3.11, 3.12

migrate from `setup.py` to `pyproject.toml`

includes support in virtual environment out of the box

the original `mkl_fft.rfft` and `mkl_fft.irfft` are renamed to `mkl_fft.rfftpack` and `mkl_fft.irfftpack`,
since they replicate the behavior from the deprecated `scipy.fftpack` module.

`mkl_fft.rfft_numpy`, `mkl_fft.irfft_numpy`, `mkl_fft.rfft2_numpy`, `mkl_fft.irfft2_numpy`,
`mkl_fft.rfftn_numpy`, and `mkl_fft.irfftn_numpy` are renamed to `mkl_fft.rfft`, `mkl_fft.irfft`,
`mkl_fft.rfft2`, `mkl_fft.irfft2`, `mkl_fft.rfftn`, and `mkl_fft.irfftn`, respectively.
(consistent with `numpy.fft` and `scipy.fft` modules)

file `_scipy_fft_backend.py` is renamed to `_scipy_fft.py` since it replicates `scipy.fft` module
(similar to file `_numpy_fft.py` which replicates `numpy.fft` module)

## [1.3.11]

Bugfix release, resolving gh-109 and updating installation instructions

## [1.3.10]

Bugfix release, resolving leftover uses of NumPy attributes removed in NumPy 2.0 that break
test suite run on Windows.

## [1.3.9]

Updated code and build system to support NumPy 2.0

## [1.3.8]

Added vendored `conv_template.py` from NumPy's distutils submodule to enable building of `mkl_fft` with
NumPy >=1.25 and Python 3.12

## [1.3.7]

Updated build system away from removed in NumPy 1.25 numpy.distutils module.

Transitioned to Cython 3.0.

## [1.3.0]

Updated numpy interface to support new in NumPy 1.20 supported values of norm keyword, such as "forward" and "backward".
To enable this, `mkl_fft` functions now support `forward_scale` parameter that defaults to 1.

Fixed issue #48.

## [1.2.1]

Includes bug fix #54

## [1.2.0]

Due to removal of deprecated real-to-real FFT with `DFTI_CONJUGATE_EVEN_STORAGE=DFTI_COMPLEX_REAL` and
`DFTI_PACKED_FORMAT=DFTI_PACK` from Intel(R) Math Kernel Library, reimplemented `mkl_fft.rfft` and
`mkl_fft.irfft` to use real-to-complex functionality with subsequent copying to rearange the transform as expected
of `mkl_fft.rfft`, with the associated performance penalty. The use of the real-to-complex
transform improves multi-core utilization which may offset the performance loss incurred due to copying.

## [1.1.0]

Added `scipy.fft` backend, see #42. Fixed #46.

## [1.0.15]

Changed tests to not compare against numpy fft, as this broke due to renaming of `np.fft.pocketfft` to
`np.fft._pocketfft`. Instead compare against naive realization of 1D FFT as a sum.

Setup script is now aware of `MKLROOT` environment variable. If unset, NumPy's mkl_info will be queried.

## [1.0.14]

Fixed unreferenced bug in `irfftn_numpy`, and adjusted NumPy interfaces to change to pocketfft in NumPy 1.17

## [1.0.13]

Issue #39 fixed (memory leak with complex FFT on real arrays)

## [1.0.12]

Issue #37 fixed.

Inhibited vectorization of short loops computing pointer to memory referenced by a multi-iterator by Intel (R) C Compiler,
improving performance of ND `fft` and `ifft` on real input arrays.

## [1.0.11]

Improvement for performance of ND `fft` on real input arrays by inlining multi-iterators.
This particularly benefits performance of mkl_fft built with Intel (R) C Compiler.

## [1.0.10]

Fix for issue #29.

## [1.0.7]

Improved exception message raised if MKL is signalling an error. The message now includes MKL's own description of the exception.
This partially improves #24.

Improved argument validation for ND transforms aligning with scipy 1.2.0

## [1.0.6]

Fixed issues #21, and addressed NumPy 1.15 deprecation warnings from using lists instead of tuples to specify multiple slices.

## [1.0.5]

Fixed issues #7, #17, #18.
Consolidated version specification into a single file `mkl_fft/_version.py`.

## [1.0.4]

Added CHANGES.rst. Fixed issue #11 by using lock around calls to 1D FFT routines.

## [1.0.3]

This is a bug fix release.

It fixes issues #9, and #13.

As part of fixing issue #13, out-of-place 1D FFT calls such as `fft`, `ifft`, `rfft_numpy`
and `irfftn_numpy` will allocate Fortran layout array for the output is the input is a Fotran array.

## [1.0.2]

Minor update of `mkl_fft`, reflecting renaming of `numpy.core.multiarray_tests` module to
`numpy.core._multiarray_tests` as well as fixing #4.

## [1.0.1]

Bug fix release.

## [1.0.0]

Initial release of `mkl_fft`.

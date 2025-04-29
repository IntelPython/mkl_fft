## `mkl_fft` --  a NumPy-based Python interface to Intel® oneAPI Math Kernel Library (OneMKL) FFT functionality
[![Conda package](https://github.com/IntelPython/mkl_fft/actions/workflows/conda-package.yml/badge.svg)](https://github.com/IntelPython/mkl_fft/actions/workflows/conda-package.yml)
[![Editable build using pip and pre-release NumPy](https://github.com/IntelPython/mkl_fft/actions/workflows/build_pip.yaml/badge.svg)](https://github.com/IntelPython/mkl_fft/actions/workflows/build_pip.yaml)
[![Conda package with conda-forge channel only](https://github.com/IntelPython/mkl_fft/actions/workflows/conda-package-cf.yml/badge.svg)](https://github.com/IntelPython/mkl_fft/actions/workflows/conda-package-cf.yml)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/IntelPython/mkl_fft/badge)](https://securityscorecards.dev/viewer/?uri=github.com/IntelPython/mkl_fft)

`mkl_fft` started as a part of Intel® Distribution for Python* optimizations to NumPy, and is now being released
as a stand-alone package. It can be installed into conda environment from Intel's channel using:

```
   conda install -c https://software.repos.intel.com/python/conda mkl_fft
```

or from conda-forge channel:

```
   conda install -c conda-forge mkl_fft
```

---

To install `mkl_fft` PyPI package please use following command:

```
   python -m pip install --index-url https://software.repos.intel.com/python/pypi --extra-index-url https://pypi.org/simple mkl_fft
```

If command above installs NumPy package from the PyPI, please use following command to install Intel optimized NumPy wheel package from Intel PyPI Cloud:

```
   python -m pip install --index-url https://software.repos.intel.com/python/pypi --extra-index-url https://pypi.org/simple mkl_fft numpy==<numpy_version>
```

Where `<numpy_version>` should be the latest version from https://software.repos.intel.com/python/conda/

---

Since MKL FFT supports performing discrete Fourier transforms over non-contiguously laid out arrays, OneMKL can be directly
used on any well-behaved floating point array with no internal overlaps for both in-place and not in-place transforms of
arrays in single and double floating point precision.

This eliminates the need to copy input array contiguously into an intermediate buffer.

`mkl_fft` directly supports N-dimensional Fourier transforms.

More details can be found in SciPy 2017 conference proceedings:
     https://github.com/scipy-conference/scipy_proceedings/tree/2017/papers/oleksandr_pavlyk

---

`mkl_fft` implements the following functions:

### complex-to-complex (c2c) transforms:

`fft(x, n=None, axis=-1, overwrite_x=False, fwd_scale=1.0, out=out)` - 1D FFT, similar to `scipy.fft.fft`

`fft2(x, s=None, axes=(-2, -1), overwrite_x=False, fwd_scale=1.0, out=out)` - 2D FFT, similar to `scipy.fft.fft2`

`fftn(x, s=None, axes=None, overwrite_x=False, fwd_scale=1.0, out=out)` - ND FFT, similar to `scipy.fft.fftn`

and similar inverse FFT (`ifft*`) functions.

### real-to-complex (r2c) and complex-to-real (c2r) transforms:

`rfft(x, n=None, axis=-1, fwd_scale=1.0, out=out)` - r2c 1D FFT, similar to `numpy.fft.rfft`

`rfft2(x, s=None, axes=(-2, -1), fwd_scale=1.0, out=out)` - r2c 2D FFT, similar to `numpy.fft.rfft2`

`rfftn(x, s=None, axes=None, fwd_scale=1.0, out=out)` - r2c ND FFT, similar to `numpy.fft.rfftn`

and similar inverse c2r FFT (`irfft*`) functions.

The package also provides `mkl_fft.interfaces.numpy_fft` and `mkl_fft.interfaces.scipy_fft` interfaces which provide drop-in replacements for equivalent functions in NumPy and SciPy, respectively.

`mkl_fft.interfaces.scipy_fft` can also be used as a backend for `scipy.fft.set_backend()`

```python
>>> import numpy as np, mkl_fft, mkl_fft.interfaces.scipy_fft as mkl_be, scipy, scipy.fft, mkl

>>> mkl.verbose(1)
# True

>>> x = np.random.randn(8*7).reshape((7, 8))
>>> with scipy.fft.set_backend(mkl_be, only=True):
>>>     ff = scipy.fft.fft2(x, workers=4)
>>> ff2 = scipy.fft.fft2(x)
# MKL_VERBOSE oneMKL 2025 Update 1 Product build 20250306 for Intel(R) 64 architecture Intel(R) Advanced Vector Extensions 2 (Intel(R) AVX2) enabled processors, Lnx 2.70GHz intel_thread
# MKL_VERBOSE FFT(drfo7:8:8x8:1:1,input_strides:{0,8,1},output_strides:{0,8,1},bScale:0.0178571,tLim:1,unaligned_input,unaligned_output,desc:0x561750094440) 15.56us CNR:OFF Dyn:1 FastMM:1 TID:0  NThr:4

>>> np.allclose(ff, ff2)
```
---

To build `mkl_fft` from sources on Linux with Intel® OneMKL:
  - create a virtual environment: `python3 -m venv fft_env`
  - activate the environment: `source fft_env/bin/activate`
  - install a recent version of OneMKL, if necessary
  - execute `source /path_to_oneapi/mkl/latest/env/vars.sh`
  - `git clone https://github.com/IntelPython/mkl_fft.git mkl_fft`
  - `cd mkl_fft`
  - `python -m pip install .`
  - `cd ..`
  - `python -c "import mkl_fft"`

To build `mkl_fft` from sources on Linux with conda follow these steps:
  - `conda create -n fft_env python=3.12 mkl-devel`
  - `conda activate fft_env`
  - `export MKLROOT=$CONDA_PREFIX`
  - `git clone https://github.com/IntelPython/mkl_fft.git mkl_fft`
  - `cd mkl_fft`
  - `python -m pip install .`
  - `cd ..`
  - `python -c "import mkl_fft"`

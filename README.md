[![Conda package](https://github.com/IntelPython/mkl_fft/actions/workflows/conda-package.yml/badge.svg)](https://github.com/IntelPython/mkl_fft/actions/workflows/conda-package.yml)
[![Editable build using pip and pre-release NumPy](https://github.com/IntelPython/mkl_fft/actions/workflows/build_pip.yaml/badge.svg)](https://github.com/IntelPython/mkl_fft/actions/workflows/build_pip.yaml)
[![Conda package with conda-forge channel only](https://github.com/IntelPython/mkl_fft/actions/workflows/conda-package-cf.yml/badge.svg)](https://github.com/IntelPython/mkl_fft/actions/workflows/conda-package-cf.yml)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/IntelPython/mkl_fft/badge)](https://securityscorecards.dev/viewer/?uri=github.com/IntelPython/mkl_fft)

## `mkl_fft` --  a NumPy-based Python interface to Intel® oneAPI Math Kernel Library (oneMKL) Fourier Transform Functions

# Introduction
`mkl_fft` is part of [Intel® Distribution for Python*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html) optimizations to NumPy.
It offers a thin layered python interface to the [Intel® oneAPI Math Kernel Library (oneMKL) Fourier Transform Functions](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-2/fourier-transform-functions.html) that allows efficient access to computing a discrete Fourier transform through the fast Fourier transform (FFT) algorithm. As a result, its performance is close to the performance of native C/Intel® oneMKL. The optimizations are provided for real and complex data types in both single and double precisions for in-place and out-of-place modes of operation. For analyzing the performance use [FFT benchmarks](https://github.com/intelpython/fft_benchmark).

Thanks to Intel® oneMKL’s flexibility in its supports for arbitrarily strided input and output arrays both one-dimensional and multi-dimensional FFTs along distinct axes can be performed directly, without the need to copy the input into a contiguous array first. Furthermore, input strides can be arbitrary, including negative or zero, as long as strides remain an integer multiple of array’s item size, otherwise a copy will be made.

More details can be found in ["Accelerating Scientific Python with Intel Optimizations"](https://proceedings.scipy.org/articles/shinma-7f4c6e7-00f) from Proceedings of the 16th Python in Science Conference (SciPy 2017).

---
# Installation
`mkl_fft` can be installed into conda environment from Intel's channel using:

```
   conda install -c https://software.repos.intel.com/python/conda mkl_fft
```

or from conda-forge channel:

```
   conda install -c conda-forge mkl_fft
```

To install `mkl_fft` PyPI package please use the following command:

```
   python -m pip install --index-url https://software.repos.intel.com/python/pypi --extra-index-url https://pypi.org/simple mkl_fft
```

If command above installs NumPy package from the PyPI, please use following command to install Intel optimized NumPy wheel package from Intel PyPI Cloud:

```
   python -m pip install --index-url https://software.repos.intel.com/python/pypi --extra-index-url https://pypi.org/simple mkl_fft numpy==<numpy_version>
```

where `<numpy_version>` should be the latest version from https://software.repos.intel.com/python/conda/.

---
# How to use?
## `mkl_fft.interfaces` module
The recommended way to use `mkl_fft` package is through `mkl_fft.interfaces` module. These interfaces act as drop-in replacements for equivalent functions in NumPy and SciPy. Learn more about these interfaces [here](https://github.com/IntelPython/mkl_fft/blob/master/mkl_fft/interfaces/README.md).

## `mkl_fft` package
While using the interfaces module is the recommended way to leverage `mk_fft`, one can also use `mkl_fft` directly with the following FFT functions:

### complex-to-complex (c2c) transforms:

`fft(x, n=None, axis=-1, overwrite_x=False, fwd_scale=1.0, out=None)` - 1D FFT, similar to `scipy.fft.fft`

`fft2(x, s=None, axes=(-2, -1), overwrite_x=False, fwd_scale=1.0, out=None)` - 2D FFT, similar to `scipy.fft.fft2`

`fftn(x, s=None, axes=None, overwrite_x=False, fwd_scale=1.0, out=None)` - ND FFT, similar to `scipy.fft.fftn`

and similar inverse FFT (`ifft*`) functions.

### real-to-complex (r2c) and complex-to-real (c2r) transforms:

`rfft(x, n=None, axis=-1, fwd_scale=1.0, out=None)` - r2c 1D FFT, similar to `numpy.fft.rfft`

`rfft2(x, s=None, axes=(-2, -1), fwd_scale=1.0, out=None)` - r2c 2D FFT, similar to `numpy.fft.rfft2`

`rfftn(x, s=None, axes=None, fwd_scale=1.0, out=None)` - r2c ND FFT, similar to `numpy.fft.rfftn`

and similar inverse c2r FFT (`irfft*`) functions.

The following example shows how to use `mkl_fft` for calculating a 1D FFT.

```python
import numpy, mkl_fft
a = numpy.random.randn(10) + 1j*numpy.random.randn(10)

mkl_res = mkl_fft.fft(a)
np_res = numpy.fft.fft(a)
numpy.allclose(mkl_res, np_res)
# True
```

---
# Building from source

To build `mkl_fft` from sources on Linux with Intel® oneMKL:
  - create a virtual environment: `python3 -m venv fft_env`
  - activate the environment: `source fft_env/bin/activate`
  - install a recent version of oneMKL, if necessary
  - execute `source /path_to_oneapi/mkl/latest/env/vars.sh`
  - `git clone https://github.com/IntelPython/mkl_fft.git mkl_fft`
  - `cd mkl_fft`
  - `python -m pip install .`
  - `pip install scipy` (optional: for using `mkl_fft.interface.scipy_fft` module)
  - `cd ..`
  - `python -c "import mkl_fft"`

To build `mkl_fft` from sources on Linux with conda follow these steps:
  - `conda create -n fft_env python=3.12 mkl-devel`
  - `conda activate fft_env`
  - `export MKLROOT=$CONDA_PREFIX`
  - `git clone https://github.com/IntelPython/mkl_fft.git mkl_fft`
  - `cd mkl_fft`
  - `python -m pip install .`
  - `conda install scipy` (optional: for using `mkl_fft.interface.scipy_fft` module)
  - `cd ..`
  - `python -c "import mkl_fft"`

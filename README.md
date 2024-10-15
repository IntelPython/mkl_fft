## ``mkl_fft`` --  a NumPy-based Python interface to Intel (R) MKL FFT functionality
[![Conda package](https://github.com/IntelPython/mkl_fft/actions/workflows/conda-package.yml/badge.svg)](https://github.com/IntelPython/mkl_fft/actions/workflows/conda-package.yml)
[![Editable build using pip and pre-release NumPy](https://github.com/IntelPython/mkl_fft/actions/workflows/build_pip.yaml/badge.svg)](https://github.com/IntelPython/mkl_fft/actions/workflows/build_pip.yaml)
[![Conda package with conda-forge channel only](https://github.com/IntelPython/mkl_fft/actions/workflows/conda-package-cf.yml/badge.svg)](https://github.com/IntelPython/mkl_fft/actions/workflows/conda-package-cf.yml)

`mkl_fft` started as a part of Intel (R) Distribution for Python* optimizations to NumPy, and is now being released
as a stand-alone package. It can be installed into conda environment using

```
   conda install -c https://software.repos.intel.com/python/conda mkl_fft
```

or from conda-forge channel:

```
   conda install -c conda-forge mkl_fft
```

---

To install mkl_fft Pypi package please use following command:

```
   python -m pip install --index-url https://software.repos.intel.com/python/pypi --extra-index-url https://pypi.org/simple mkl_fft
```

If command above installs NumPy package from the Pypi, please use following command to install Intel optimized NumPy wheel package from Intel Pypi Cloud:

```
   python -m pip install --index-url https://software.repos.intel.com/python/pypi --extra-index-url https://pypi.org/simple mkl_fft numpy==<numpy_version>
```

Where `<numpy_version>` should be the latest version from https://software.repos.intel.com/python/conda/

---

Since MKL FFT supports performing discrete Fourier transforms over non-contiguously laid out arrays, MKL can be directly
used on any well-behaved floating point array with no internal overlaps for both in-place and not in-place transforms of
arrays in single and double floating point precision.

This eliminates the need to copy input array contiguously into an intermediate buffer.

`mkl_fft` directly supports N-dimensional Fourier transforms.

More details can be found in SciPy 2017 conference proceedings:
     https://github.com/scipy-conference/scipy_proceedings/tree/2017/papers/oleksandr_pavlyk

---

It implements the following functions:

### Complex transforms, similar to those in `scipy.fftpack`:

`fft(x, n=None, axis=-1, overwrite_x=False)`

`ifft(x, n=None, axis=-1, overwrite_x=False)`

`fft2(x, shape=None, axes=(-2,-1), overwrite_x=False)`

`ifft2(x, shape=None, axes=(-2,-1), overwrite_x=False)`

`fftn(x, n=None, axes=None, overwrite_x=False)`

`ifftn(x, n=None, axes=None, overwrite_x=False)`

### Real transforms

`rfft(x, n=None, axis=-1, overwrite_x=False)` - real 1D Fourier transform, like `scipy.fftpack.rfft`

`rfft_numpy(x, n=None, axis=-1)` - real 1D Fourier transform, like `numpy.fft.rfft`

`rfft2_numpy(x, s=None, axes=(-2,-1))` - real 2D Fourier transform, like `numpy.fft.rfft2`

`rfftn_numpy(x, s=None, axes=None)` - real 2D Fourier transform, like `numpy.fft.rfftn`

... and similar `irfft*` functions.


The package also provides `mkl_fft._numpy_fft` and `mkl_fft._scipy_fft` interfaces which provide drop-in replacements for equivalent functions in NumPy and SciPy respectively.

---

To build ``mkl_fft`` from sources on Linux:
  - install a recent version of MKL, if necessary;
  - execute ``source /path/to/mklroot/bin/mklvars.sh intel64`` ;
  - execute ``pip install .``

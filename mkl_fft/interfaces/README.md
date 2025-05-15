# Interfaces
The `mkl_fft` package provides interfaces that serve as drop-in replacements for equivalent functions in NumPy and SciPy.

---

## NumPy interface - `mkl_fft.interfaces.numpy_fft`

This interface is a drop-in replacement for the [`numpy.fft`](https://numpy.org/devdocs/reference/routines.fft.html) module and includes **all** the functions available there:

* complex-to-complex FFTs: `fft`, `ifft`, `fft2`, `ifft2`, `fftn`, `ifftn`.

* real-to-complex and complex-to-real FFTs: `rfft`, `irfft`, `rfft2`, `irfft2`, `rfftn`, `irfftn`.

* Hermitian FFTs: `hfft`, `ihfft`.

* Helper routines: `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift`. These routines serve as a fallback to the NumPy implementation and are included for completeness.

The following example shows how to use this interface for calculating a 1D FFT.

```python
import numpy
import mkl_fft.interfaces.numpy_fft as numpy_fft

a = numpy.random.randn(10) + 1j*numpy.random.randn(10)

mkl_res = numpy_fft.fft(a)
np_res = numpy.fft.fft(a)
numpy.allclose(mkl_res, np_res)
# True
```

---

## SciPy interface - `mkl_fft.interfaces.scipy_fft`
This interface is a drop-in replacement for the [`scipy.fft`](https://scipy.github.io/devdocs/reference/fft.html) module and includes **subset** of the functions available there:

* complex-to-complex FFTs: `fft`, `ifft`, `fft2`, `ifft2`, `fftn`, `ifftn`.

* real-to-complex and complex-to-real FFTs: `rfft`, `irfft`, `rfft2`, `irfft2`, `rfftn`, `irfftn`.

* Hermitian FFTs: `hfft`, `ihfft`, `hfft2`, `ihfft2`, `hfftn`, `ihfftn`.

* Helper functions: `fftshift`, `ifftshift`, `fftfreq`, `rfftfreq`, `set_workers`, `get_workers`. All of these functions, except for `set_workers` and `get_workers`, serve as a fallback to the SciPy implementation and are included for completeness.

The following example shows how to use this interface for calculating a 1D FFT.

```python
import numpy, scipy
import mkl_fft.interfaces.scipy_fft as scipy_fft

a = numpy.random.randn(10) + 1j * numpy.random.randn(10)

mkl_res = scipy_fft.fft(a)
sp_res = scipy.fft.fft(a)
numpy.allclose(mkl_res, sp_res)
# True
```

---

### Registering `mkl_fft` as the FFT backend for SciPy

`mkl_fft.interfaces.scipy_fft` can be registered as a backend for SciPy. To learn more about how to control the backend [see the SciPy documentation](https://docs.scipy.org/doc/scipy/reference/fft.html#backend-control). The following example shows how to set `mkl_fft` as the FFT backend for SciPy using a context manager.

```python
import numpy, scipy, mkl
import mkl_fft.interfaces.scipy_fft as mkl_backend
x = numpy.random.randn(56).reshape(7, 8)

# turning on verbosity to show `mkl_fft` is used as the SciPy backend
mkl.verbose(1)
# True

with scipy.fft.set_backend(mkl_backend, only=True):
    mkl_res = scipy.fft.fft2(x, workers=4)  # Calls `mkl_fft` backend
# MKL_VERBOSE oneMKL 2024.0 Update 2 Patch 2 Product build 20240823 for Intel(R) 64 architecture Intel(R) Advanced Vector Extensions 512 (Intel(R) AVX-512) with support for INT8, BF16, FP16 (limited) instructions, and Intel(R) Advanced Matrix Extensions (Intel(R) AMX) with INT8 and BF16, Lnx 2.00GHz intel_thread
# MKL_VERBOSE FFT(drfo7:8:8x8:1:1,input_strides:{0,8,1},output_strides:{0,8,1},bScale:0.0178571,tLim:1,unaligned_output,desc:0x557affb60d40) 36.11us CNR:OFF Dyn:1 FastMM:1 TID:0  NThr:4

sp_res = scipy.fft.fft2(x, workers=4)  # Calls default SciPy backend
numpy.allclose(mkl_res, sp_res)
# True
```

The previous example was only for illustration purposes. In practice, there is no added benefit to defining a backend and calculating FFT, since this functionality is already accessible through the scipy interface, as shown earlier.
To demonstrate the advantages of using `mkl_fft` as a backend, the following example compares the timing of `scipy.signal.fftconvolve` using the default SciPy backend versus the `mkl_fft` backend on an Intel® Xeon® CPU.

```python
import numpy, scipy
import mkl_fft.interfaces.scipy_fft as mkl_backend
import timeit
shape = (4096, 2048)
a = numpy.random.randn(*shape) + 1j*numpy.random.randn(*shape)
b = numpy.random.randn(*shape) + 1j*numpy.random.randn(*shape)

t1 = timeit.timeit(lambda: scipy.signal.fftconvolve(a, b), number=10)
print(f"Time with scipy.fft default backend: {t1:.1f} seconds")
# Time with scipy.fft default backend: 51.9 seconds

with scipy.fft.set_backend(mkl_backend, only=True):
    t2 = timeit.timeit(lambda: scipy.signal.fftconvolve(a, b), number=10)

print(f"Time with OneMKL FFT backend installed: {t2:.1f} seconds")
# Time with MKL FFT backend installed: 9.1 seconds
```

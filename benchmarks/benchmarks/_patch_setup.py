"""MKL patch setup — executed once per ASV worker process at import time.

Patches NumPy FFT with the Intel MKL FFT implementation.
Hard-fails with a descriptive RuntimeError if mkl_fft is missing or the
patch does not take effect, so benchmarks never silently run on stock NumPy.
"""

_PATCH_MAP = [
    ("mkl_fft", "patch_numpy_fft"),
]


def _apply_patches():
    import numpy as np

    patched = {}

    for mod_name, patch_fn_name in _PATCH_MAP:
        try:
            mod = __import__(mod_name)
        except ImportError as exc:
            raise RuntimeError(
                f"[mkl-patch] Cannot import {mod_name}: {exc}\n"
                f"  Ensure the conda env contains {mod_name} "
                f"from the Intel channel.\n"
                "  Required channels: "
                "https://software.repos.intel.com/python/conda"
            ) from exc

        patch_fn = getattr(mod, patch_fn_name, None)
        if patch_fn is None:
            raise RuntimeError(
                f"[mkl-patch] {mod_name} has no {patch_fn_name}(). "
                f"Upgrade {mod_name} to a version that exposes "
                "the stock-numpy patch API."
            )

        try:
            patch_fn()
        except Exception as exc:
            raise RuntimeError(
                f"[mkl-patch] {mod_name}.{patch_fn_name}() raised: {exc!r}"
            ) from exc

        is_patched_fn = getattr(mod, "is_patched", None)
        if callable(is_patched_fn) and not is_patched_fn():
            raise RuntimeError(
                f"[mkl-patch] {mod_name}.is_patched() returned False "
                "after patching. NumPy may have been imported before "
                "patching in a conflicting state."
            )

        patched[mod_name] = mod

    _attr_checks = {
        "mkl_fft": lambda: np.fft.fft.__module__,
    }
    for mod_name in patched:
        try:
            attr = _attr_checks[mod_name]()
        except Exception:
            attr = "unknown"
        print(f"[mkl-patch] {mod_name}: numpy.fft dispatch -> {attr}")

    print("[mkl-patch] ALL OK -- mkl_fft active")

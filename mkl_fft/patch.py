# Copyright (c) 2026, Intel Corporation
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
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Persistent patch management for NumPy FFT submodule."""

import site
import sys
import warnings
from pathlib import Path


class PatchOperationError(RuntimeError):
    """Raised when a persistent patch operation cannot be completed."""


def get_pth_path():
    """Get the path to mkl_fft_patch.pth in the appropriate site-packages."""
    site_packages = site.getsitepackages()
    if site_packages:
        target_site = site_packages[0]
    else:
        target_site = site.getusersitepackages()
    return Path(target_site) / "mkl_fft_patch.pth"


PTH_CONTENT = """import mkl_fft._patch_startup"""


def install_patch(verbose=False):
    """Install persistent NumPy FFT patch using .pth file."""
    pth_path = get_pth_path()

    if pth_path.exists():
        if verbose:
            warnings.warn(
                f"Persistent patch already installed at {pth_path}",
                UserWarning,
                stacklevel=2
            )
        return

    try:
        pth_path.parent.mkdir(parents=True, exist_ok=True)
        pth_path.write_text(PTH_CONTENT)
        if verbose:
            print(f"Persistent patch installed at {pth_path}")
            print()
            print("NumPy FFT will now use MKL-accelerated implementations in all")
            print("Python sessions. To disable, run:")
            print("  python -m mkl_fft patch uninstall")
    except OSError as e:
        raise PatchOperationError(
            f"Error installing patch at {pth_path}: {e}\n\n"
            "You may need to run with appropriate permissions or install to "
            "a user site-packages directory."
        ) from e


def uninstall_patch(verbose=False):
    """Uninstall persistent NumPy FFT patch."""
    pth_path = get_pth_path()

    if not pth_path.exists():
        if verbose:
            print("No persistent patch found.")
        return

    try:
        pth_path.unlink()
        if verbose:
            print(f"Persistent patch removed from {pth_path}")
            print()
            print("NumPy FFT will now use the default implementations.")
    except OSError as e:
        raise PatchOperationError(
            f"Error removing patch at {pth_path}: {e}"
        ) from e


def check_status(verbose=False):
    """Check if persistent patch is installed."""
    pth_path = get_pth_path()

    if pth_path.exists():
        if verbose:
            print(f"Persistent patch is installed at {pth_path}")
            print()
            print("NumPy FFT is configured to use MKL-accelerated implementations.")
        return True
    else:
        if verbose:
            print("No persistent patch installed")
            print()
            print("To enable MKL-accelerated NumPy FFT globally, run:")
            print("  python -m mkl_fft patch install")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m mkl_fft.patch <install|uninstall|status>")
        sys.exit(1)

    command = sys.argv[1]
    try:
        if command == "install":
            install_patch(verbose=True)
        elif command == "uninstall":
            uninstall_patch(verbose=True)
        elif command == "status":
            sys.exit(0 if check_status(verbose=True) else 1)
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
    except PatchOperationError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

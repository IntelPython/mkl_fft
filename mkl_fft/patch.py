# Copyright (c) 2017, Intel Corporation
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

import argparse
import site
import sys
from pathlib import Path


def get_pth_path():
    """Get the path to mkl_fft_patch.pth in the appropriate site-packages."""
    site_packages = site.getsitepackages()
    if site_packages:
        target_site = site_packages[0]
    else:
        target_site = site.getusersitepackages()
    return Path(target_site) / "mkl_fft_patch.pth"


PTH_CONTENT = """import mkl_fft._patch_startup"""


def install_patch():
    """Install persistent NumPy FFT patch using .pth file."""
    pth_path = get_pth_path()

    if pth_path.exists():
        print(f"Persistent patch already installed at {pth_path}")
        return

    try:
        pth_path.parent.mkdir(parents=True, exist_ok=True)
        pth_path.write_text(PTH_CONTENT)
        print(f"Persistent patch installed at {pth_path}")
        print()
        print("NumPy FFT will now use MKL-accelerated implementations in all")
        print("Python sessions. To disable, run:")
        print("  python -m mkl_fft patch uninstall")
    except OSError as e:
        print(f"Error installing patch: {e}")
        print()
        print("You may need to run with appropriate permissions or install to")
        print("a user site-packages directory.")
        sys.exit(1)


def uninstall_patch():
    """Uninstall persistent NumPy FFT patch."""
    pth_path = get_pth_path()

    if not pth_path.exists():
        print("No persistent patch found.")
        return

    try:
        pth_path.unlink()
        print(f"Persistent patch removed from {pth_path}")
        print()
        print("NumPy FFT will now use the default implementations.")
    except OSError as e:
        print(f"Error removing patch: {e}")
        sys.exit(1)


def check_status():
    """Check if persistent patch is installed."""
    pth_path = get_pth_path()

    if pth_path.exists():
        print(f"Persistent patch is installed at {pth_path}")
        print()
        print("NumPy FFT is configured to use MKL-accelerated implementations.")
        return True
    else:
        print("No persistent patch installed")
        print()
        print("To enable MKL-accelerated NumPy FFT globally, run:")
        print("  python -m mkl_fft patch install")
        return False


def main(args=None):
    """Main entry point for patch command."""
    parser = argparse.ArgumentParser(
        prog="python -m mkl_fft patch",
        description="Manage persistent NumPy FFT patching with MKL acceleration",
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands"
    )

    subparsers.add_parser("install", help="Install persistent NumPy FFT patch")
    subparsers.add_parser(
        "uninstall", help="Uninstall persistent NumPy FFT patch"
    )
    subparsers.add_parser(
        "status", help="Check if persistent patch is installed"
    )

    parsed_args = parser.parse_args(args)

    if not parsed_args.command:
        parser.print_help()
        sys.exit(1)

    if parsed_args.command == "install":
        install_patch()
    elif parsed_args.command == "uninstall":
        uninstall_patch()
    elif parsed_args.command == "status":
        sys.exit(0 if check_status() else 1)


if __name__ == "__main__":
    main()

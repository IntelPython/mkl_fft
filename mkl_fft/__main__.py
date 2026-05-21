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
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Command-line interface for mkl_fft."""

import argparse
import sys


def main():
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m mkl_fft",
        description="MKL-accelerated FFT for NumPy",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--patch",
        choices=["install", "uninstall", "status"],
        help="Manage persistent NumPy FFT patching",
    )
    parser.add_argument(
        "--with-patch",
        dest="with_patch",
        nargs=argparse.REMAINDER,
        help="Run command with temporary NumPy FFT patch",
    )

    args = parser.parse_args()

    if args.patch:
        from mkl_fft.patch import check_status, install_patch, uninstall_patch

        if args.patch == "install":
            install_patch(verbose=args.verbose)
        elif args.patch == "uninstall":
            uninstall_patch(verbose=args.verbose)
        elif args.patch == "status":
            sys.exit(0 if check_status(verbose=args.verbose) else 1)

    elif args.with_patch is not None:
        from mkl_fft.with_patch import run_with_patch

        run_with_patch(args.with_patch)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

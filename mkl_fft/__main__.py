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
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # patch subcommand with its own subparsers
    patch_parser = subparsers.add_parser(
        "patch", help="Manage persistent NumPy FFT patching"
    )
    patch_subparsers = patch_parser.add_subparsers(
        dest="patch_command", help="Patch operations"
    )
    patch_subparsers.add_parser("install", help="Install persistent NumPy FFT patch")
    patch_subparsers.add_parser(
        "uninstall", help="Uninstall persistent NumPy FFT patch"
    )
    patch_subparsers.add_parser("status", help="Check if persistent patch is installed")

    # with_patch subcommand
    with_patch_parser = subparsers.add_parser(
        "with_patch", help="Run command with temporary NumPy FFT patch"
    )
    with_patch_parser.add_argument(
        "command", nargs=argparse.REMAINDER, help="Command to execute with patch"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "patch":
        from mkl_fft.patch import check_status, install_patch, uninstall_patch

        if not args.patch_command:
            patch_parser.print_help()
            sys.exit(1)

        if args.patch_command == "install":
            install_patch(verbose=args.verbose)
        elif args.patch_command == "uninstall":
            uninstall_patch(verbose=args.verbose)
        elif args.patch_command == "status":
            sys.exit(0 if check_status(verbose=args.verbose) else 1)

    elif args.command == "with_patch":
        from mkl_fft.with_patch import run_with_patch

        run_with_patch(args.command)


if __name__ == "__main__":
    main()

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
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Command-line interface for mkl_fft."""

import sys


def main_impl():
    if len(sys.argv) < 2:
        print("Usage: python -m mkl_fft <command> [args]")
        print()
        print("Commands:")
        print("  patch install      Install persistent NumPy FFT patch")
        print("  patch uninstall    Uninstall persistent NumPy FFT patch")
        print("  patch status       Check if persistent patch is installed")
        print("  with_patch <cmd>   Run command with temporary NumPy FFT patch")
        print()
        print("Examples:")
        print("  python -m mkl_fft patch install")
        print("  python -m mkl_fft with_patch python script.py")
        sys.exit(1)

    command = sys.argv[1]

    if command == "patch":
        from mkl_fft.patch import main as patch_main

        patch_main(sys.argv[2:])
    elif command == "with_patch":
        from mkl_fft.with_patch import main as with_patch_main

        with_patch_main(sys.argv[2:])
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


def main():
    """Entry point for the CLI."""
    main_impl()


if __name__ == "__main__":
    main()

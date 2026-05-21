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

"""Run Python commands with temporary NumPy FFT patch."""

import os
import subprocess
import sys
import tempfile


def run_with_patch(args):
    """Run a command with mkl_fft NumPy patch enabled."""
    if not args:
        print("Usage: python -m mkl_fft with_patch <command> [args...]")
        print()
        print("Examples:")
        print("  python -m mkl_fft with_patch python script.py")
        print("  python -m mkl_fft with_patch python -m pytest tests/")
        print(
            "  python -m mkl_fft with_patch python -c 'import numpy; print(numpy.fft.fft.__module__)'"
        )
        sys.exit(1)

    sitecustomize_content = """# mkl_fft temporary patch
try:
    import mkl_fft
    mkl_fft.patch_numpy_fft()
except Exception:
    pass
"""

    temp_dir = tempfile.mkdtemp(prefix="mkl_fft_patch_")
    sitecustomize_path = os.path.join(temp_dir, "sitecustomize.py")

    try:
        with open(sitecustomize_path, "w") as f:
            f.write(sitecustomize_content)

        env = os.environ.copy()

        existing_pythonpath = env.get("PYTHONPATH", "")
        if existing_pythonpath:
            env["PYTHONPATH"] = f"{temp_dir}{os.pathsep}{existing_pythonpath}"
        else:
            env["PYTHONPATH"] = temp_dir

        result = subprocess.run(args, env=env)
        sys.exit(result.returncode)
    finally:
        try:
            os.unlink(sitecustomize_path)
            os.rmdir(temp_dir)
        except OSError:
            pass


def main(args=None):
    """Deprecated entry point. Use run_with_patch() instead."""
    run_with_patch(args if args else sys.argv[1:])


if __name__ == "__main__":
    main()

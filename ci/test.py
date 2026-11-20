"""
Tests mkl_fft's installed package.

Runs with one built conda package or wheel installed in a clean environment, as
ci/hooks.py sets up, from an empty working directory, so the tests import the
installed mkl_fft rather than the source tree. The test requirements come from
ci/config.yml. Writes a JUnit report to $GH_CI_RESULTS.
"""

import importlib
import importlib.util
import os
import pathlib
import sys
import sysconfig

import pytest

PACKAGE = "mkl_fft"


def running_python() -> str:
    """This interpreter's version as artifact names spell it, such as 3.14t."""
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    threaded = sysconfig.get_config_var("Py_GIL_DISABLED")
    return version + ("t" if threaded else "")


def scipy_interface_error() -> str | None:
    """Why mkl_fft.interfaces.scipy_fft fails to import with SciPy and mkl-service installed."""
    if not all(importlib.util.find_spec(name) for name in ("scipy", "mkl")):
        return None
    try:
        importlib.import_module("mkl_fft.interfaces.scipy_fft")
    except ImportError as error:
        return f"{type(error).__name__}: {error}"
    return None


def main() -> int:
    """Runs the installed package's test suite."""
    expected = os.environ["GH_CI_PYTHON"]
    if running_python() != expected:
        print(f"running Python {running_python()}, expected {expected}")
        return 1
    # Its tests skip without the interface, which would hide a broken import
    error = scipy_interface_error()
    if error:
        print(f"mkl_fft.interfaces.scipy_fft does not import: {error}")
        return 1
    # The suite name says which platform and Python ran.
    cell = f"{os.environ.get('GH_CI_PLATFORM', 'linux')}.py{expected}"
    report = pathlib.Path(os.environ["GH_CI_RESULTS"]) / f"{PACKAGE}.xml"
    return pytest.main(
        [
            "-v",
            "-p",
            "no:cacheprovider",
            f"--junitxml={report}",
            "-o",
            f"junit_suite_name={PACKAGE} {cell}",
            "--pyargs",
            PACKAGE,
        ]
    )


if __name__ == "__main__":
    sys.exit(main())

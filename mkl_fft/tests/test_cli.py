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

import site

import mkl_fft
import pytest

from mkl_fft.patch import (
    PatchOperationError,
    check_status,
    install_patch,
    uninstall_patch,
)


@pytest.fixture
def mock_pth_path(tmp_path, monkeypatch):
    """Mock the .pth file path to use a temporary directory."""
    pth_file = tmp_path / "mkl_fft_patch.pth"

    def mock_get_pth_path():
        return pth_file

    monkeypatch.setattr("mkl_fft.patch.get_pth_path", mock_get_pth_path)
    return pth_file


def test_install_patch(mock_pth_path, capsys):
    """Test installing persistent patch."""
    install_patch(verbose=True)

    assert mock_pth_path.exists()
    content = mock_pth_path.read_text()
    assert "import mkl_fft._patch_startup" in content

    captured = capsys.readouterr()
    assert "Persistent patch installed" in captured.out


def test_install_patch_already_installed(mock_pth_path):
    """Test installing patch when already installed."""
    install_patch()
    with pytest.warns(UserWarning, match="already installed"):
        install_patch(verbose=True)


def test_uninstall_patch(mock_pth_path, capsys):
    """Test uninstalling persistent patch."""
    install_patch()
    assert mock_pth_path.exists()

    uninstall_patch(verbose=True)
    assert not mock_pth_path.exists()

    captured = capsys.readouterr()
    assert "Persistent patch removed" in captured.out


def test_uninstall_patch_not_installed(mock_pth_path, capsys):
    """Test uninstalling patch when not installed."""
    uninstall_patch(verbose=True)

    captured = capsys.readouterr()
    assert "No persistent patch found" in captured.out


def test_patch_status_check_function(mock_pth_path):
    """Test check_status function return values."""
    assert not check_status()

    install_patch()
    assert check_status()

    uninstall_patch()
    assert not check_status()


def test_install_patch_enables_runtime_patch_via_pth(mock_pth_path):
    """Test that .pth activation results in patched NumPy FFT runtime state."""
    install_patch()

    preexisting_patch_state = mkl_fft.is_patched()
    try:
        site.addsitedir(str(mock_pth_path.parent))
        assert mkl_fft.is_patched()
    finally:
        if mkl_fft.is_patched() and not preexisting_patch_state:
            mkl_fft.restore_numpy_fft()


def test_install_patch_raises_patch_operation_error_on_oserror(
    mock_pth_path, monkeypatch
):
    """Test install_patch raises a typed error for filesystem failures."""

    def mock_write_text(*args, **kwargs):
        raise OSError("mock write failure")

    monkeypatch.setattr("pathlib.Path.write_text", mock_write_text)

    with pytest.raises(PatchOperationError, match="Error installing patch"):
        install_patch()


def test_uninstall_patch_raises_patch_operation_error_on_oserror(
    mock_pth_path, monkeypatch
):
    """Test uninstall_patch raises a typed error for filesystem failures."""
    install_patch()

    def mock_unlink(*args, **kwargs):
        raise OSError("mock unlink failure")

    monkeypatch.setattr("pathlib.Path.unlink", mock_unlink)

    with pytest.raises(PatchOperationError, match="Error removing patch"):
        uninstall_patch()


def test_cli_patch_install_exits_with_error_on_patch_operation_error(
    monkeypatch, capsys
):
    """Test CLI maps patch operation failures to exit code 1."""
    from mkl_fft import __main__ as cli_main

    def mock_install_patch(*args, **kwargs):
        raise PatchOperationError("mock cli install failure")

    monkeypatch.setattr("mkl_fft.patch.install_patch", mock_install_patch)
    monkeypatch.setattr("sys.argv", ["python", "--patch", "install"])

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "mock cli install failure" in captured.err

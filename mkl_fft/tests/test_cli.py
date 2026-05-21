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

import pytest

from mkl_fft.patch import check_status, install_patch, uninstall_patch


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
    install_patch()

    assert mock_pth_path.exists()
    content = mock_pth_path.read_text()
    assert "import mkl_fft._patch_startup" in content

    captured = capsys.readouterr()
    assert "Persistent patch installed" in captured.out


def test_install_patch_already_installed(mock_pth_path, capsys):
    """Test installing patch when already installed."""
    install_patch()
    install_patch()

    captured = capsys.readouterr()
    assert "already installed" in captured.out


def test_uninstall_patch(mock_pth_path, capsys):
    """Test uninstalling persistent patch."""
    install_patch()
    assert mock_pth_path.exists()

    uninstall_patch()
    assert not mock_pth_path.exists()

    captured = capsys.readouterr()
    assert "Persistent patch removed" in captured.out


def test_uninstall_patch_not_installed(mock_pth_path, capsys):
    """Test uninstalling patch when not installed."""
    uninstall_patch()

    captured = capsys.readouterr()
    assert "No persistent patch found" in captured.out


def test_patch_status_check_function(mock_pth_path):
    """Test check_status function return values."""
    assert not check_status()

    install_patch()
    assert check_status()

    uninstall_patch()
    assert not check_status()

#!/bin/bash -ex

# gnu_thread is safe on conda-forge: llvm-openmp provides the GNU OpenMP symbols
# (libgomp) in addition to its own so MKL finds the correct OpenMP runtime at runtime.
# See: https://conda-forge.org/docs/maintainer/knowledge_base/#openmp
$PYTHON -m pip install --no-build-isolation --no-deps -Csetup-args="-Dmkl_threading=gnu_thread" .

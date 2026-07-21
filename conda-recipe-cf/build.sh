#!/bin/bash -ex

$PYTHON -m pip install --no-build-isolation --no-deps -Csetup-args="-Dmkl_threading=gnu_thread" .

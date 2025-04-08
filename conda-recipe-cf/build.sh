#!/bin/bash -x

export MKLROOT=$PREFIX
export CFLAGS="-I$PREFIX/include $CFLAGS"
$PYTHON -m pip install --no-build-isolation --no-deps .

#!/bin/bash -x

if [ "$(uname)" == Darwin ]; then
    export MACOSX_DEPLOYMENT_TARGET=10.9
fi

export MKLROOT=$PREFIX
export CFLAGS="-I$PREFIX/include $CFLAGS"
$PYTHON -m pip install --no-build-isolation --no-deps .

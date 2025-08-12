#!/bin/bash -x

export MKLROOT=$PREFIX
export CFLAGS="-I$PREFIX/include $CFLAGS"
export LDFLAGS="-Wl,-rpath,\$ORIGIN/../.. -Wl,-rpath,\$ORIGIN/../../.. -L${PREFIX}/lib ${LDFLAGS}"

read -r GLIBC_MAJOR GLIBC_MINOR <<<"$(conda list '^sysroot_linux-64$' \
    | tail -n 1 | awk '{print $2}' | grep -oP '\d+' | head -n 2 | tr '\n' ' ')"

# Build wheel package
if [ -n "${WHEELS_OUTPUT_FOLDER}" ]; then
     $PYTHON -m pip wheel --no-build-isolation --no-deps .
     ${PYTHON} -m wheel tags --remove --platform-tag "manylinux_${GLIBC_MAJOR}_${GLIBC_MINOR}_x86_64" mkl_fft*.whl
     cp mkl_fft*.whl "${WHEELS_OUTPUT_FOLDER}"
else
    # Build conda package
    $PYTHON -m pip install --no-build-isolation --no-deps .
fi

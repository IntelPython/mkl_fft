rem intel_thread (default) is safe on conda-forge: llvm-openmp ships the Intel OpenMP
rem runtime library directly so MKL finds the correct OpenMP runtime at runtime.
%PYTHON% -m pip install --no-build-isolation --no-deps .
if errorlevel 1 exit 1

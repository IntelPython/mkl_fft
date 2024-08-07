set MKLROOT=%PREFIX%
%PYTHON% -m pip install --no-build-isolation --no-deps .
if errorlevel 1 exit 1

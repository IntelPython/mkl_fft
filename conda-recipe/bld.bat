@rem Remember to activate Intel Compiler, or remoe these two lines to ise Microsoft Visual Studio compiler

set CC=icl
set LD=xilink

%PYTHON% setup.py build --force --compiler=intelemw install --old-and-unmanageable
if errorlevel 1 exit 1

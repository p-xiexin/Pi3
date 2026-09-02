@echo off
setlocal

if defined PI3_PYTHON (
    "%PI3_PYTHON%" -m datasets.tools %*
) else (
    python -m datasets.tools %*
)

exit /b %ERRORLEVEL%

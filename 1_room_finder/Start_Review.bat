@echo off
rem Open the review app. Usage: Start_Review.bat [scan folder]
rem Without an argument a folder picker opens in scans\
rem Python: RCF_REVIEW_PYTHON if set, otherwise .venv (created by Setup.ps1 in the package root).
rem Language: English by default; set RCF_LANG=zh for Chinese (or switch in the app).
setlocal
rem resolve a relative scan folder against the caller's folder before changing to this one
set "SCAN="
if not "%~1"=="" set "SCAN=%~f1"
cd /d "%~dp0"
set "PY="
if defined RCF_REVIEW_PYTHON if exist "%RCF_REVIEW_PYTHON%" set "PY=%RCF_REVIEW_PYTHON%"
if not defined PY if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY goto nopython
"%PY%" -c "import tkinter, PIL" >nul 2>&1
if errorlevel 1 goto badpython
if not defined SCAN goto picker
"%PY%" -m finder.app --scan "%SCAN%"
goto done
:picker
"%PY%" -m finder.app
goto done
:nopython
echo Python environment not found. Run Setup.ps1 in the package root first.
pause
exit /b 1
:badpython
echo The review environment does not run on this computer (was the folder copied from another computer?).
echo Run Setup.ps1 in the package root to rebuild it.
pause
exit /b 1
:done
if errorlevel 1 pause
endlocal

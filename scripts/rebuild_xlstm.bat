@echo off
REM Rebuild xlstm with CUDA support on Windows
REM This script sets up the Visual Studio environment and rebuilds xlstm

echo ========================================
echo Rebuilding xlstm with CUDA support
echo ========================================
echo.

REM Set up Visual Studio environment
echo Setting up Visual Studio Build Tools environment...
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to set up Visual Studio environment
    echo Please ensure Visual Studio 2022 Build Tools are installed
    pause
    exit /b 1
)

REM Set CUDA environment variables
echo Setting CUDA environment variables...
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
if not exist "%CUDA_HOME%" (
    echo WARNING: CUDA_HOME not found at %CUDA_HOME%
    echo Please adjust CUDA_HOME if your CUDA is installed elsewhere
)

set PATH=%CUDA_HOME%\bin;%CUDA_HOME%\libnvvp;%PATH%

REM Uninstall existing xlstm
echo.
echo Uninstalling existing xlstm...
pip uninstall xlstm -y

REM Install xlstm from source
echo.
echo Installing xlstm from source (this may take several minutes)...
pip install xlstm --no-cache-dir --verbose

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo xlstm installed successfully!
    echo ========================================
    echo.
    echo Testing xlstm import...
    python -c "from xlstm import xLSTMBlockStack; print('SUCCESS: xlstm imported correctly')"
) else (
    echo.
    echo ========================================
    echo ERROR: xlstm installation failed
    echo ========================================
    echo.
    echo Please check the error messages above.
    echo Common issues:
    echo 1. CUDA toolkit not found
    echo 2. Visual Studio Build Tools not properly configured
    echo 3. Incompatible PyTorch/CUDA versions
)

pause



@echo off
REM =============================================================
REM build_cuda_ext.bat - build 3DGS CUDA extensions on Windows
REM
REM Why this script exists:
REM   Building CUDA extensions on Windows needs THREE things:
REM     1. cl.exe  -> must be injected by vcvars64.bat
REM     2. nvcc    -> installed in a separate env "cuda129"
REM     3. TORCH_CUDA_ARCH_LIST=12.0 -> otherwise no kernel for
REM        RTX 5070 (Blackwell, sm_120)
REM   Calling pip install from a plain shell fails with
REM   "cl.exe not found" or "CUDA_HOME not set".
REM
REM IMPORTANT: keep this file pure ASCII (cmd codepage issues).
REM =============================================================

setlocal

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 echo [WARN] vcvars64.bat returned error

set "CUDA_HOME=D:\conda\Anaconda\envs\cuda129\Library"
set "TORCH_CUDA_ARCH_LIST=12.0"
set "DISTUTILS_USE_SDK=1"
set "MSSdk=1"
set "PYTHON=D:\conda\Anaconda\envs\gs\python.exe"

echo.
echo ==== env check ====
echo CUDA_HOME=%CUDA_HOME%
if not exist "%CUDA_HOME%\bin\nvcc.exe" echo [ERROR] nvcc.exe missing under CUDA_HOME
if not exist "%CUDA_HOME%\include\cuda_runtime.h" echo [ERROR] cuda_runtime.h missing
echo TORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST%
where cl.exe
where nvcc.exe

cd /d D:\deepseek\cv

echo.
echo ==== building %1 ====
"%PYTHON%" -m pip install --progress-bar off --no-build-isolation --no-cache-dir %1
echo BUILD_EXIT=%ERRORLEVEL%
endlocal

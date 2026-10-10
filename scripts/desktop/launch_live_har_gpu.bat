@echo off
setlocal
cd /d "%~dp0..\.."
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
echo Loading KARD HAR. Q / Escape closes the preview.
call conda run --no-capture-output -n xai python -B scripts\desktop\realtime_kard.py --config configs\movinet_kard_cuda_fp16.json %*
if errorlevel 1 (
  echo Launch failed. Check the xai environment and installed runtime dependencies above.
  pause
  exit /b 1
)
endlocal

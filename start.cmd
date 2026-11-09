@echo off
setlocal
pushd "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Missing .venv\Scripts\python.exe. Set up the local Windows ARM environment first.
    popd
    exit /b 1
)
if not defined COMFYUI_ANIMA_QNN set "COMFYUI_ANIMA_QNN=1"
if "%COMFYUI_ANIMA_QNN%"=="0" set "COMFYUI_DISABLE_QNN_UNET=1"
".venv\Scripts\python.exe" main.py --cpu --bf16-unet --use-pytorch-cross-attention --listen 127.0.0.1 --port 8192 --auto-launch %*
set "comfy_exit=%ERRORLEVEL%"
popd
exit /b %comfy_exit%

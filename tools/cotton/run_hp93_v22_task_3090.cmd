@echo off
set REPO=C:\Users\WYZ\Desktop\cotton\rtdetrv4_hp93\RT-DETRv4
set OUT=%REPO%\outputs\improved_v2_2_finetune
if not exist "%OUT%" mkdir "%OUT%"
cd /d "%REPO%"
echo task_start=%DATE% %TIME% > "%OUT%\task_wrapper.log"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%REPO%\tools\cotton\train_hp93_v22_3090.ps1" >> "%OUT%\task_wrapper.log" 2>&1
echo task_exit=%ERRORLEVEL% >> "%OUT%\task_wrapper.log"

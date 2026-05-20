@echo off
set REPO=C:\Users\WYZ\Desktop\cotton\rtdetrv4_hp93\RT-DETRv4
set OUT=%REPO%\outputs\improved_v2_2_finetune
if not exist "%OUT%" mkdir "%OUT%"
D:\Anaconda3\python.exe -u C:\Users\WYZ\Desktop\cotton\rtdetrv4_baseline\monitor\cotton_sync_monitor.py --log-dir "%OUT%" --config "%REPO%\configs\cotton\rtv4_hgnetv2_m_cotton_hp93_v22.yml" --host 0.0.0.0 --port 18080 > "%OUT%\monitor.log" 2>&1

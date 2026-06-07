$ErrorActionPreference = "Stop"
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"
$Python = "D:\Anaconda3\python.exe"
$Config = "configs\cotton\rtv4_hgnetv2_m_cotton_v11_mixema_b3_fromscratch_768.yml"
$Output = "outputs\rtv4_hgnetv2_m_cotton_v11_mixema_b3_fromscratch_768"
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$Log = Join-Path $Output "train_stdout.log"
& $Python -u train.py -c $Config -d cuda --seed 0 --use-amp 2>&1 | Tee-Object -FilePath $Log
exit $LASTEXITCODE

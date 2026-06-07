param(
    [string]$Repo = "C:\Users\WYZ\Desktop\cotton\rtdetrv4_hp93\RT-DETRv4",
    [string]$Python = "D:\Anaconda3\python.exe",
    [switch]$SkipStage1
)

$ErrorActionPreference = "Stop"
Set-Location $Repo
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"

$Stage1Config = "configs\cotton\rtv4_hgnetv2_m_cotton_balanced_v6_768.yml"
$Stage1Out = "outputs\rtv4_hgnetv2_m_cotton_balanced_v6_768"
$Stage2Config = "configs\cotton\rtv4_hgnetv2_m_cotton_v10_progressive_refine_768.yml"
$Stage2Out = "outputs\rtv4_hgnetv2_m_cotton_v10_progressive_refine_768"

function Find-Stage1Checkpoint {
    $candidates = @(
        (Join-Path $Stage1Out "best_ap50.pth"),
        (Join-Path $Stage1Out "best_stg2.pth"),
        (Join-Path $Stage1Out "best_stg1.pth"),
        (Join-Path $Stage1Out "last.pth")
    )
    foreach ($p in $candidates) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

New-Item -ItemType Directory -Force -Path $Stage1Out | Out-Null
New-Item -ItemType Directory -Force -Path $Stage2Out | Out-Null

$Stage1Ckpt = Find-Stage1Checkpoint
if (-not $SkipStage1 -and -not $Stage1Ckpt) {
    $Stage1Log = Join-Path $Stage1Out "train_stdout_v10_stage1.log"
    $Args1 = @(
        "-u", "train.py",
        "-c", $Stage1Config,
        "-d", "cuda",
        "--seed", "0",
        "--use-amp"
    )
    Write-Host "[v10] Stage 1: train stable v6 baseline from scratch"
    Write-Host "$Python $($Args1 -join ' ')"
    $ErrorActionPreference = "Continue"
    & $Python @Args1 2>&1 | Tee-Object -FilePath $Stage1Log
    $ExitCode = $LASTEXITCODE
    "python_exit=$ExitCode" | Tee-Object -FilePath $Stage1Log -Append
    $ErrorActionPreference = "Stop"
    if ($ExitCode -ne 0) { exit $ExitCode }
    $Stage1Ckpt = Find-Stage1Checkpoint
}

if (-not $Stage1Ckpt) {
    throw "No Stage-1 checkpoint found under $Stage1Out. Run v6 first or remove -SkipStage1."
}

$Stage2Log = Join-Path $Stage2Out "train_stdout.log"
$Args2 = @(
    "-u", "train.py",
    "-c", $Stage2Config,
    "-t", $Stage1Ckpt,
    "-d", "cuda",
    "--seed", "0",
    "--use-amp"
)
Write-Host "[v10] Stage 2: progressive refinement from $Stage1Ckpt"
Write-Host "$Python $($Args2 -join ' ')"
$ErrorActionPreference = "Continue"
& $Python @Args2 2>&1 | Tee-Object -FilePath $Stage2Log
$ExitCode = $LASTEXITCODE
"python_exit=$ExitCode" | Tee-Object -FilePath $Stage2Log -Append
$ErrorActionPreference = "Stop"
exit $ExitCode

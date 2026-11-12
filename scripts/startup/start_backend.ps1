# Terminal 1: Start FastAPI + Authoritative PyBullet Simulation Manager
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $Root
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PATH = "$Root\.conda-env;$Root\.conda-env\Library\bin;$Root\.conda-env\Scripts;$env:PATH"
Write-Host "Starting Dexterous-Robot-Learning FastAPI Backend on http://127.0.0.1:8000 ..."
.\.conda-env\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000


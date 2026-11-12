# Terminal 2: Start Vite + React + Three.js Workstation Frontend
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location "$Root\frontend"
Write-Host "Starting Dexterous-Robot-Learning Vite Frontend on http://localhost:5173 ..."
npm run dev


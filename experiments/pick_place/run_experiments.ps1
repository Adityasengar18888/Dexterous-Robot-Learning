<#
.SYNOPSIS
Full-scale experiment execution for Pick and Place (E1 vs E2).
#>

Write-Host "Starting E1: SAC without HER (1,000,000 steps)..."
.\.conda-env\python.exe -m src.rl.train --config configs/pick_place_no_her.yaml
# Ensure you update the config total_timesteps to 1000000 before running

Write-Host "Starting E2: SAC + HER (1,000,000 steps)..."
.\.conda-env\python.exe -m src.rl.train --config configs/pick_place_her.yaml
# Ensure you update the config total_timesteps to 1000000 before running

Write-Host "Evaluating E1..."
.\.conda-env\python.exe -m src.rl.evaluate --model experiments/pick_place/E1_no_her/models/final_model.zip --config configs/pick_place_no_her.yaml --episodes 100

Write-Host "Evaluating E2..."
.\.conda-env\python.exe -m src.rl.evaluate --model experiments/pick_place/E2_her/models/final_model.zip --config configs/pick_place_her.yaml --episodes 100

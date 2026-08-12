$ErrorActionPreference = "Stop"

$OutputDir = "runs/imagination_only_fast_trace_gpu_seed7"

Write-Host "=== CUDA sanity ==="
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

Write-Host "=== Imagination-only fast trace ==="
python scripts/run_imagination_only_fast_trace.py `
  --output-dir $OutputDir `
  --seed 7 `
  --transitions 2048 `
  --block-target 512 `
  --margin 0.05 `
  --stages 1,2,3,4 `
  --scenario-seeds 94001,94003 `
  --device cuda:0

Write-Host "=== Summary ==="
Get-Content "$OutputDir/summary.json"

Write-Host "=== Intervention trace ==="
Get-Content "$OutputDir/intervention_trace_aassr_imagination_margin_0.05.jsonl"

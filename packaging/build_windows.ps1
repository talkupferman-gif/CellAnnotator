$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
python -m PyInstaller --noconfirm --windowed --name CellAnnotator --collect-all PySide6 cellannotator/__main__.py
Write-Host "Built dist\CellAnnotator"

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
python -m PyInstaller --noconfirm --windowed --name CellAnnotator --icon cellannotator/assets/icon.ico --add-data "cellannotator/assets;cellannotator/assets" --collect-all PySide6 cellannotator/__main__.py
Write-Host "Built dist\CellAnnotator"

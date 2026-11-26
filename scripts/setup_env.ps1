# Reproducible Windows environment setup for DSCI-41.
#
# Why a signed python.org interpreter and the stdlib `venv` module:
# Windows Smart App Control blocks unsigned executables. uv-managed CPython builds and uv's venv launcher are
# unsigned and were blocked on the development machine (RISK_REGISTER R-14). The python.org build installed by
# the Python Install Manager (`py install`) is Authenticode-signed, and so is the launcher `python -m venv` copies.
#
# Usage (from the repository root):  powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1

$ErrorActionPreference = 'Stop'
$VenvDir = if ($env:DSCI41_VENV) { $env:DSCI41_VENV } else { Join-Path $HOME 'dsci41\venv' }

Write-Host "1/4 Ensuring signed Python 3.11 is installed (Python Install Manager)"
py -V:3.11 -c "import sys" 2>$null
if ($LASTEXITCODE -ne 0) { py install -y 3.11-64 }
$Py311 = (py -V:3.11 -c "import sys; print(sys.executable)").Trim()

Write-Host "2/4 Creating virtual environment at $VenvDir (outside OneDrive)"
if (-not (Test-Path (Join-Path $VenvDir 'Scripts\python.exe'))) { & $Py311 -m venv $VenvDir }

Write-Host "3/4 Installing locked dependencies with uv"
$env:UV_PROJECT_ENVIRONMENT = $VenvDir
$env:UV_PYTHON = Join-Path $VenvDir 'Scripts\python.exe'
uv sync --frozen
if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }

Write-Host "4/4 Verifying environment"
& (Join-Path $VenvDir 'Scripts\python.exe') scripts\check_env.py
if ($LASTEXITCODE -ne 0) { throw "environment check failed" }
Write-Host "Done. Activate with: & '$VenvDir\Scripts\Activate.ps1'"

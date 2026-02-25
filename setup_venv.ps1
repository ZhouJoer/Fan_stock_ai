# Setup Python venv and install dependencies (UTF-8)
# Python 3.8+ supported; Python 3.10 is recommended for ta/akshare compatibility.

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "my_stock - Setup Virtual Environment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Use first available: python, py -3.10, python3.10 (no version required)
$PythonCmd = $null
foreach ($cmd in @("python", "py -3.10", "python3.10")) {
    try {
        $ver = & cmd /c "$cmd --version 2>&1"
        if ($LASTEXITCODE -eq 0 -and $ver) {
            $PythonCmd = $cmd
            break
        }
    } catch {}
}
if (-not $PythonCmd) {
    Write-Host "[ERROR] Python not found. Install Python 3.x (recommended: 3.10) from https://www.python.org/downloads/" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

function Invoke-Python {
    $p = $PythonCmd -split ' ', 2
    if ($p.Length -ge 2) { & $p[0] $p[1] @args } else { & $p[0] @args }
}

Write-Host "[1/4] Checking Python..." -ForegroundColor Yellow
$pythonVersion = Invoke-Python --version 2>&1
Write-Host "Using: $PythonCmd -> $pythonVersion" -ForegroundColor Green
$versionOk = Invoke-Python -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" 2>$null; if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Python 3.8+ required." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "[TIP] Python 3.10 is recommended for best compatibility with ta/akshare." -ForegroundColor Cyan

# Create venv
Write-Host "[2/4] Creating venv..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "venv exists, skip." -ForegroundColor Green
} else {
    Write-Host "Creating venv (may take a few minutes)..." -ForegroundColor Yellow
    try {
        Invoke-Python -m venv venv
        if ($LASTEXITCODE -ne 0) {
            throw "Failed"
        }
        Write-Host "venv created." -ForegroundColor Green
        if (Test-Path "venv\Scripts\python.exe") {
            & "venv\Scripts\python.exe" -m ensurepip --upgrade | Out-Null
        }
    } catch {
        Write-Host "[ERROR] Failed to create venv." -ForegroundColor Red
        Write-Host "If Z: is a network drive, try running from a local path or: python -m venv venv" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Activate and upgrade pip
Write-Host "[3/4] Activating venv and upgrading pip..." -ForegroundColor Yellow
if (-not (Test-Path "venv\Scripts\Activate.ps1")) {
    Write-Host "[ERROR] venv\Scripts\Activate.ps1 not found." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

& "venv\Scripts\Activate.ps1"
try {
    python -m pip install --upgrade pip | Out-Null
    Write-Host "pip upgraded." -ForegroundColor Green
} catch {
    Write-Host "[WARN] pip upgrade failed, continuing..." -ForegroundColor Yellow
}

# Install dependencies
Write-Host "[4/4] Installing requirements..." -ForegroundColor Yellow
if (-not (Test-Path "requirements.txt")) {
    Write-Host "[ERROR] requirements.txt not found." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

try {
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        throw "Failed"
    }
    Write-Host "Requirements installed." -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Failed to install requirements." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Done!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Activate:  .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "  2. Start API: python -m uvicorn api_server:app --host 127.0.0.1 --port 8000 --reload" -ForegroundColor Cyan
Write-Host "     or:       npm run api" -ForegroundColor Cyan
Write-Host "  3. Deactivate: deactivate" -ForegroundColor Cyan
Write-Host ""
if ($PWD.Path -like "Z:*") {
    Write-Host "[TIP] Z: may be a network drive; venv can be slow. Consider Docker or local path." -ForegroundColor Yellow
}
Read-Host "Press Enter to exit"

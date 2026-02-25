@echo off
REM Windows virtual environment setup script
REM Usage: Creates Python venv and installs dependencies

echo ========================================
echo my_stock - Setup Virtual Environment
echo ========================================
echo.

REM Check if Python is installed
echo [1/4] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.x from https://www.python.org/downloads/
    echo         Recommended: Python 3.10
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python version: %PYTHON_VERSION%
python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.8+ required. Current: %PYTHON_VERSION%
    pause
    exit /b 1
)
echo [TIP] Python 3.10 is recommended for best compatibility with ta/akshare.

echo [2/4] Creating venv...
if exist venv (
    echo venv exists, skip.
) else (
    echo Creating venv (may take a few minutes)...
    python -m venv venv --without-pip
    if errorlevel 1 (
        echo [ERROR] venv creation failed, trying without --without-pip...
        python -m venv venv
        if errorlevel 1 (
            echo [ERROR] Failed to create venv.
            echo If Z: is a network drive, try running from a local path or: python -m venv venv
            pause
            exit /b 1
        )
    )
    echo venv created.
    REM Install pip if --without-pip was used
    if exist venv\Scripts\python.exe (
        echo Installing pip...
        venv\Scripts\python.exe -m ensurepip --upgrade >nul 2>&1
    )
)

echo [3/4] Activating venv and upgrading pip...
if not exist venv\Scripts\activate.bat (
    echo [ERROR] venv\Scripts\activate.bat not found.
    echo Check that venv directory is complete.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [WARN] venv activation failed, using venv python directly...
    set PATH=%~dp0venv\Scripts;%PATH%
)
python -m pip install --upgrade pip >nul 2>&1
if errorlevel 1 (
    echo [WARN] pip upgrade failed, continuing with install...
)

echo [4/4] Installing requirements...
if not exist requirements.txt (
    echo [ERROR] requirements.txt not found.
    pause
    exit /b 1
)

pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install requirements.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Done!
echo ========================================
echo.
echo Next steps:
echo   1. Activate venv:
if exist venv\Scripts\activate.bat (
    echo    venv\Scripts\activate.bat
) else (
    echo    [NOTE] If activation fails, use:
    echo    set PATH=%~dp0venv\Scripts;%%PATH%%
)
echo.
echo   2. Start API:
echo    python -m uvicorn api_server:app --host 127.0.0.1 --port 8000 --reload
echo    or: npm run api
echo.
echo   3. Deactivate: deactivate
echo.
echo [TIP] If Z: is a network drive, venv can be slow. Consider Docker or local path.
echo.
pause

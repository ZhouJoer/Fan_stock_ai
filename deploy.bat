@echo off
REM Windows Docker deployment script
REM Usage: One-click deploy my_stock (Docker Compose)

echo ========================================
echo my_stock - Docker Deployment
echo ========================================
echo.

REM Check Docker
echo [1/5] Checking Docker...
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker not found. Install Docker Desktop from:
    echo https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

docker-compose --version >nul 2>&1
if errorlevel 1 (
    docker compose version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] docker-compose not found.
        echo Ensure Docker Desktop is installed and running.
        pause
        exit /b 1
    ) else (
        set COMPOSE_CMD=docker compose
    )
) else (
    set COMPOSE_CMD=docker-compose
)

echo Docker installed.
echo.

REM Check .env
echo [2/5] Checking env config...
if not exist .env (
    if exist .env.example (
        echo .env not found, creating from .env.example...
        copy .env.example .env >nul
        echo [WARN] Edit .env and set DEEPSEEK_API_KEY
        echo Press any key to continue (ensure API key is set)...
        pause >nul
    ) else (
        echo [ERROR] .env or .env.example not found
        pause
        exit /b 1
    )
) else (
    echo .env already exists
)
echo.

REM Create data dirs
echo [3/5] Creating data directories...
if not exist data mkdir data
if not exist db mkdir db
if not exist pools mkdir pools
if not exist data\etf_sim_accounts mkdir data\etf_sim_accounts
echo Data directories created.
echo.

REM Build images
echo [4/5] Building Docker images...
%COMPOSE_CMD% build
if errorlevel 1 (
    echo [ERROR] Docker build failed
    pause
    exit /b 1
)
echo.

REM Start services
echo [5/5] Starting services...
%COMPOSE_CMD% up -d
if errorlevel 1 (
    echo [ERROR] Failed to start services
    pause
    exit /b 1
)

echo.
echo ========================================
echo Done!
echo ========================================
echo.
echo Access:
echo   Frontend:  http://localhost:5173
echo   Backend:   http://localhost:8000
echo   Health:    http://localhost:8000/api/health
echo.
echo Useful commands:
echo   Logs:   %COMPOSE_CMD% logs -f
echo   Stop:   %COMPOSE_CMD% down
echo   Restart: %COMPOSE_CMD% restart
echo   Status: %COMPOSE_CMD% ps
echo.
pause

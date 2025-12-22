@echo off
REM Start Celery workers for Windows
REM
REM Usage: start_workers.bat [mode]
REM
REM Modes:
REM   dev      - Single worker for development
REM   monitor  - Start Flower monitoring dashboard

setlocal

set MODE=%1
if "%MODE%"=="" set MODE=dev

echo ================================
echo Hybrid Agentic System - Celery Workers
echo ================================
echo.

REM Check if Redis is running
echo Checking Redis connection...
redis-cli ping >nul 2>&1
if errorlevel 1 (
    echo ERROR: Redis is not running!
    echo Start Redis with: docker run -d -p 6379:6379 redis:alpine
    echo Or: docker-compose up -d redis
    exit /b 1
)
echo OK: Redis is running
echo.

if /i "%MODE%"=="dev" (
    echo Starting DEVELOPMENT mode single worker...
    echo.
    celery -A src.tasks.celery_app worker --loglevel=info --queue=code_execution,code_execution_heavy --concurrency=2 --max-tasks-per-child=100 --hostname=worker-dev@%%h
) else if /i "%MODE%"=="monitor" (
    echo Starting Flower monitoring dashboard...
    echo.
    echo Access at: http://localhost:5555
    echo Username: admin
    echo Password: admin123
    echo.
    celery -A src.tasks.celery_app flower --port=5555 --basic_auth=admin:admin123
) else (
    echo Invalid mode: %MODE%
    echo.
    echo Usage: %0 [mode]
    echo.
    echo Modes:
    echo   dev      - Single worker for development
    echo   monitor  - Start Flower monitoring dashboard
    exit /b 1
)

endlocal

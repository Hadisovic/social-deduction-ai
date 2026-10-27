@echo off
echo Social Deduction AI - Spectator ^& Replay Studio
echo =============================================
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe run_training_spectator.py %*
) else (
    python run_training_spectator.py %*
)
pause

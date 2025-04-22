@echo off
:: Check if running as administrator.
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Requesting administrative privileges...
    powershell -Command "Start-Process '%~dpn0.bat' -Verb runAs"
    exit
)

:: Change directory to the folder where the BAT file is located.
cd /d "%~dp0"

python loader.py
pause

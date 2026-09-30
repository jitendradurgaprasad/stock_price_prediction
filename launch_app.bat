@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_CMD="
where py >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=py -3"
if not defined PYTHON_CMD (
    where python >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
    echo Python 3.10 or newer is required. Install it from https://www.python.org/downloads/ and try again.
    pause
    exit /b 1
)

%PYTHON_CMD% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
    echo Python 3.10 or newer is required. Update Python and try again.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating the project virtual environment...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 (
        echo Could not create the virtual environment.
        pause
        exit /b 1
    )
    call ".venv\Scripts\activate.bat"
    python -m pip install --upgrade pip
    if errorlevel 1 (
        echo Could not upgrade pip.
        pause
        exit /b 1
    )
) else (
    call ".venv\Scripts\activate.bat"
)

echo Installing or updating the required project packages...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Package installation failed. Check your internet connection and try again.
    pause
    exit /b 1
)

echo Starting the Stock Price Trend Prediction dashboard...
python -m streamlit run streamlit_app.py
set "APP_EXIT=%ERRORLEVEL%"
if not "%APP_EXIT%"=="0" (
    echo The dashboard stopped with an error. Review the messages above.
    pause
)
exit /b %APP_EXIT%

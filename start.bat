@echo off
echo Starting Turkey Pivots Data Visualization Service...
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found. Using system Python.
)

echo.
echo Starting the application...
echo Open your browser to: http://127.0.0.1:8050
echo Press Ctrl+C to stop the server
echo.

python app.py

pause

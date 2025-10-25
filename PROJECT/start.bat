@echo off
echo Starting ClimaCast Climate Impact Predictor...

REM Check if database exists
if not exist "climacast.db" (
    echo Database not found. Please run setup first.
    pause
    exit /b 1
)

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

REM Start Streamlit app
echo Starting Streamlit app...
start "ClimaCast Streamlit" cmd /k "streamlit run climacast_app.py --server.port=8501 --server.address=localhost"

echo.
echo ClimaCast is now running!
echo Dashboard: http://localhost:8501
echo.
echo Press any key to stop...
pause > nul

REM Kill Streamlit process
taskkill /f /im python.exe 2>nul
echo ClimaCast stopped.

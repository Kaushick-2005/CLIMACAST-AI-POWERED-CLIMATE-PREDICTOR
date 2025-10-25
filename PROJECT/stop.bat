@echo off
echo Stopping ClimaCast services...

REM Kill Python processes (Streamlit)
taskkill /f /im python.exe 2>nul
taskkill /f /im streamlit.exe 2>nul

echo ClimaCast services stopped.
pause

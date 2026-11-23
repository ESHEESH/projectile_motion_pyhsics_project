@echo off
chcp 65001 >nul
title Bow Master
cd /d "%~dp0"
echo ==============================================
echo              Bow Master
echo ==============================================
echo Controls:
echo   1 / 2 / 3  = Easy / Normal / Hard
echo   Left hand  = Move + aim (bow hand)
echo   Right hand = Pull back for power
echo   Open palm  = FIRE
echo   F          = Fullscreen
echo   R          = Restart
echo   Q          = Quit
echo ==============================================
"%LOCALAPPDATA%\Programs\Python\Python314\python.exe" main.py
pause

@echo off
set NODE_NO_WARNINGS=1
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "PORT=8441"
if not "%~1"=="" set "PORT=%~1"

rem 1) Yritä ensin paikallista projektikansiota
set "CERT=%CD%\192.168.1.105.pem"
set "KEY=%CD%\192.168.1.105-key.pem"

rem 2) Jos ei löydy, yritä mkcertin CAROOT-kansiota (AppData\Local\mkcert)
if not exist "%CERT%" (
  set "CERT=C:\Users\jouko\AppData\Local\mkcert\192.168.1.105.pem"
)
if not exist "%KEY%" (
  set "KEY=C:\Users\jouko\AppData\Local\mkcert\192.168.1.105-key.pem"
)

echo.
echo ======================================
echo   HTTPS DEV SERVER
echo ======================================
echo Folder: %CD%
echo Port  : %PORT%
echo Cert  : %CERT%
echo Key   : %KEY%
echo.

if not exist "%CERT%" (
  echo ERROR: Certificate not found:
  echo %CERT%
  echo.
  echo Tip: run in this folder:
  echo   mkcert 192.168.1.105
  echo.
  pause
  exit /b 1
)

if not exist "%KEY%" (
  echo ERROR: Key not found:
  echo %KEY%
  echo.
  pause
  exit /b 1
)

echo Open:
echo   https://192.168.1.105:%PORT%
echo.
echo Press Ctrl+C to stop.
echo.

http-server . -a 0.0.0.0 -S -C "%CERT%" -K "%KEY%" -p %PORT%

echo.
echo Server stopped.
pause
@echo off
echo Testing Node.js version check syntax...
echo.

REM Test with system Node.js if available
set NODE_CMD=node
node --version >nul 2>&1
if errorlevel 1 (
    echo Node.js not found - skipping test
    goto :end
)

echo Testing PowerShell-based version capture...
for /f "tokens=*" %%i in ('powershell -Command "& '%NODE_CMD%' --version"') do set TEST_NODE_VERSION=%%i
echo Captured version: %TEST_NODE_VERSION%

echo Testing PowerShell-based version check...
powershell -Command "try { $version = & '%NODE_CMD%' --version; $major = [int]($version -replace 'v', '' -split '\.')[0]; exit ($major -ge 16 ? 0 : 1) } catch { exit 1 }" >nul 2>&1
if errorlevel 1 (
    echo FAIL: Node.js version check failed
) else (
    echo PASS: Node.js version check succeeded
)

:end
pause

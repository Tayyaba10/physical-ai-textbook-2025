@echo off
REM setup-and-start.bat - Batch script to set up and start the documentation site

echo AI-Native Robotics Textbook Development Setup

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo Node.js is not installed. Please install Node.js before proceeding.
    echo Visit https://nodejs.org/ to download and install.
    pause
    exit /b 1
)

echo Node.js is installed ✓

REM Check Node.js version
for /f "tokens=1,2,3 delims=." %%a in ('node --version') do (
    set NODE_VERSION=%%b
)

if %NODE_VERSION% LSS 18 (
    echo Node.js version 18 or higher is required.
    node --version
    pause
    exit /b 1
)

echo Node.js version: %NODE_VERSION% ✓

REM Check if npm is installed
npm --version >nul 2>&1
if errorlevel 1 (
    echo npm is not installed. Please install npm along with Node.js.
    pause
    exit /b 1
)

echo npm is installed ✓

REM Install Docusaurus globally if not already installed
where docusaurus >nul 2>&1
if errorlevel 1 (
    echo Installing Docusaurus CLI...
    npm install -g @docusaurus/cli
) else (
    echo Docusaurus CLI is already installed ✓
)

REM Install local dependencies
echo Installing project dependencies...
npm install

if errorlevel 1 (
    echo Error installing dependencies
    pause
    exit /b 1
) else (
    echo Dependencies installed successfully ✓
)

REM Build the site
echo Building the documentation site...
npm run build

if errorlevel 1 (
    echo Error building site
    pause
    exit /b 1
) else (
    echo Site built successfully ✓
)

echo.
echo Setup complete! To start the development server:
echo 1. Run: npm start
echo 2. Open your browser to http://localhost:3000
echo.
echo The AI-Native Robotics Textbook documentation is ready for development!

pause
@echo off
echo 🔍 Tesseract OCR Installer for Windows
echo =======================================

echo.
echo Checking if Tesseract is already installed...
tesseract --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Tesseract is already installed
    goto :install_python
)

echo.
echo ⚠️  Tesseract not found in PATH
echo.
echo Trying to install with Chocolatey...
choco --version >nul 2>&1
if %errorlevel% == 0 (
    echo 🍫 Installing Tesseract with Chocolatey...
    choco install tesseract -y
    if %errorlevel% == 0 (
        echo ✅ Tesseract installed successfully
        goto :install_python
    )
)

echo.
echo ❌ Chocolatey not available or installation failed
echo.
echo 💡 Manual installation required:
echo    1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
echo    2. Run installer and check "Add to PATH"
echo    3. Restart terminal and run this script again
echo.
echo Or run the Python installer:
echo    python install_tesseract_windows.py
echo.
pause
exit /b 1

:install_python
echo.
echo 📦 Installing Python packages...
pip install pytesseract Pillow
if %errorlevel% == 0 (
    echo ✅ Python packages installed
) else (
    echo ❌ Failed to install Python packages
    pause
    exit /b 1
)

echo.
echo 🧪 Testing installation...
python -c "import pytesseract; print('✅ pytesseract working'); print('Version:', pytesseract.get_tesseract_version())" 2>nul
if %errorlevel% == 0 (
    echo ✅ Tesseract OCR is working correctly
    echo.
    echo 🎉 Installation complete!
    echo    You can now use OCR features in the API
) else (
    echo ❌ Installation test failed
    echo    Please check that Tesseract is in your PATH
)

echo.
pause
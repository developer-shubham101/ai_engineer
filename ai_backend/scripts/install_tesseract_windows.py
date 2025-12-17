#!/usr/bin/env python3
"""
Tesseract OCR installer for Windows.
Downloads and installs Tesseract OCR automatically.
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path


def check_tesseract():
    """Check if Tesseract is already installed."""
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Tesseract is already installed")
            print(f"   Version: {result.stdout.split()[1]}")
            return True
    except FileNotFoundError:
        pass
    return False


def install_with_chocolatey():
    """Try to install Tesseract using Chocolatey."""
    try:
        # Check if chocolatey is available
        subprocess.run(['choco', '--version'], 
                      capture_output=True, check=True)
        
        print("🍫 Installing Tesseract using Chocolatey...")
        result = subprocess.run(['choco', 'install', 'tesseract', '-y'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Tesseract installed successfully with Chocolatey")
            return True
        else:
            print(f"❌ Chocolatey installation failed: {result.stderr}")
            return False
            
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("⚠️  Chocolatey not found")
        return False


def download_tesseract_installer():
    """Download Tesseract installer from GitHub."""
    print("📥 Downloading Tesseract installer...")
    
    # Latest Tesseract installer URL
    installer_url = "https://github.com/UB-Mannheim/tesseract/releases/download/v5.3.3.20231005/tesseract-ocr-w64-setup-5.3.3.20231005.exe"
    installer_path = Path("tesseract-installer.exe")
    
    try:
        urllib.request.urlretrieve(installer_url, installer_path)
        print(f"✅ Downloaded installer to: {installer_path}")
        return installer_path
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None


def run_installer(installer_path):
    """Run the Tesseract installer."""
    print("🚀 Running Tesseract installer...")
    print("   Please follow the installation wizard")
    print("   Make sure to check 'Add to PATH' option")
    
    try:
        # Run installer (will open GUI)
        subprocess.run([str(installer_path)], check=True)
        print("✅ Installer completed")
        return True
    except Exception as e:
        print(f"❌ Installer failed: {e}")
        return False


def install_python_package():
    """Install pytesseract Python package."""
    print("📦 Installing Python packages...")
    
    packages = ['pytesseract', 'Pillow']
    
    for package in packages:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                          check=True, capture_output=True)
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    return True


def test_installation():
    """Test if Tesseract is working."""
    print("🧪 Testing Tesseract installation...")
    
    try:
        import pytesseract
        from PIL import Image
        
        # Test Tesseract version
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract version: {version}")
        
        # Create a simple test image with text
        test_image = Image.new('RGB', (200, 50), color='white')
        
        # Try OCR (will fail on blank image but tests if Tesseract works)
        try:
            pytesseract.image_to_string(test_image)
            print("✅ Tesseract OCR is working")
            return True
        except Exception as e:
            print(f"⚠️  Tesseract found but OCR test failed: {e}")
            return False
            
    except ImportError:
        print("❌ Python packages not installed correctly")
        return False
    except Exception as e:
        print(f"❌ Tesseract test failed: {e}")
        return False


def main():
    """Main installation process."""
    print("🔍 Tesseract OCR Installer for Windows")
    print("=" * 50)
    
    # Check if already installed
    if check_tesseract():
        print("✅ Tesseract is already available")
        if install_python_package() and test_installation():
            print("\n🎉 Setup complete!")
            return True
    
    # Try Chocolatey first
    if install_with_chocolatey():
        if install_python_package() and test_installation():
            print("\n🎉 Installation complete!")
            return True
    
    # Manual download and install
    print("\n📥 Trying manual installation...")
    installer_path = download_tesseract_installer()
    
    if installer_path and run_installer(installer_path):
        print("\n⚠️  Please restart your terminal/command prompt")
        print("   Then run this script again to complete setup")
        
        # Clean up installer
        try:
            installer_path.unlink()
            print("🧹 Cleaned up installer file")
        except:
            pass
        
        return False
    
    print("\n❌ Automatic installation failed")
    print("\n💡 Manual installation steps:")
    print("1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
    print("2. Run the installer and check 'Add to PATH'")
    print("3. Restart your terminal")
    print("4. Run: pip install pytesseract pillow")
    
    return False


if __name__ == "__main__":
    success = main()
    input("\nPress Enter to exit...")
    sys.exit(0 if success else 1)
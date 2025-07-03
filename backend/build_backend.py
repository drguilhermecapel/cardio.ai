"""
Build script for CardioAI Pro backend Windows executable.
Creates a standalone executable using PyInstaller.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    import subprocess

    required_packages = ["pyinstaller", "onnx"]
    missing_packages = []

    for package in required_packages:
        try:
            import_name = "PyInstaller" if package == "pyinstaller" else package
            result = subprocess.run(
                ["poetry", "run", "python", "-c", f"import {import_name}"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                missing_packages.append(package)
        except Exception:
            missing_packages.append(package)

    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install them with: poetry add " + " ".join(missing_packages))
        return False

    return True


def create_mock_models():
    """Create mock ONNX models if they don't exist."""
    models_dir = Path("models")

    if not models_dir.exists() or not any(models_dir.glob("*.onnx")):
        print("Creating mock ONNX models...")
        try:
            subprocess.run([sys.executable, "create_mock_models.py"], check=True)
            print("✓ Mock models created successfully")
        except subprocess.CalledProcessError as e:
            print(f"Failed to create mock models: {e}")
            return False
    else:
        print("✓ ONNX models found")

    return True


def clean_build_dirs():
    """Clean previous build directories."""
    dirs_to_clean = ["build", "dist", "__pycache__"]

    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            print(f"Cleaning {dir_name}/")
            shutil.rmtree(dir_name)


def build_executable():
    """Build the executable using PyInstaller."""
    print("Building CardioAI Pro backend executable...")

    try:
        cmd = [
            "poetry",
            "run",
            "pyinstaller",
            "--clean",
            "--noconfirm",
            "cardioai-backend.spec",
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        print("✓ Build completed successfully!")

        exe_path = Path("dist/cardioai-backend")
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print(f"✓ Executable created: {exe_path} ({size_mb:.1f} MB)")

            windows_exe_path = Path("dist/cardioai-pro-backend.exe")
            exe_path.rename(windows_exe_path)
            print(f"✓ Renamed to: {windows_exe_path}")
            return True
        else:
            print("✗ Executable not found in dist/")
            return False

    except subprocess.CalledProcessError as e:
        print(f"✗ Build failed: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def create_launcher_script():
    """Create a simple launcher script for the executable."""
    launcher_content = """@echo off
title CardioAI Pro - ECG Analysis System
echo Starting CardioAI Pro Backend...
echo.
echo CardioAI Pro v1.0.0 - ECG Analysis System
echo ==========================================
echo.
echo Backend server starting on http://localhost:8000
echo Web interface will be available once frontend is launched
echo.
echo Press Ctrl+C to stop the server
echo.

cardioai-pro-backend.exe

pause
"""

    launcher_path = Path("dist/start-cardioai-pro.bat")
    with open(launcher_path, "w") as f:
        f.write(launcher_content)

    print(f"✓ Launcher script created: {launcher_path}")


def main():
    """Main build process."""
    print("CardioAI Pro Backend Build Script")
    print("=" * 40)

    if not check_dependencies():
        sys.exit(1)

    if not create_mock_models():
        sys.exit(1)

    clean_build_dirs()

    if not build_executable():
        sys.exit(1)

    create_launcher_script()

    print("\n" + "=" * 40)
    print("Build completed successfully!")
    print("\nFiles created:")
    print("- dist/cardioai-pro-backend.exe (Main executable)")
    print("- dist/start-cardioai-pro.bat (Launcher script)")
    print("\nTo test the backend:")
    print("1. Run: dist/start-cardioai-pro.bat")
    print("2. Open browser to: http://localhost:8000/docs")


if __name__ == "__main__":
    main()

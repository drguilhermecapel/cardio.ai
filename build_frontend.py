#!/usr/bin/env python3
"""
Build script for CardioAI Pro Electron Frontend
Creates a Windows desktop application package
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, cwd: str = None) -> bool:
    """Run a shell command and return success status."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=cwd, check=True, capture_output=True, text=True
        )
        print(f"✓ Command completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed with exit code {e.returncode}")
        if e.stdout:
            print(f"Stdout: {e.stdout}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        return False


def main():
    """Main build process for Electron frontend."""
    print("=" * 60)
    print("CardioAI Pro - Frontend Build Script")
    print("Building Electron Desktop Application")
    print("=" * 60)

    frontend_dir = Path(__file__).parent
    print(f"Frontend directory: {frontend_dir}")

    package_json = frontend_dir / "package.json"
    if not package_json.exists():
        print("✗ package.json not found!")
        return False

    node_modules = frontend_dir / "node_modules"
    if not node_modules.exists():
        print("Installing npm dependencies...")
        if not run_command("npm install", str(frontend_dir)):
            print("✗ Failed to install dependencies")
            return False

    print("\nBuilding React application...")
    if not run_command("npm run build", str(frontend_dir)):
        print("✗ Failed to build React application")
        return False

    dist_dir = frontend_dir / "dist"
    if not dist_dir.exists():
        print("✗ React build failed - dist directory not found")
        return False

    print(f"✓ React build completed - {len(list(dist_dir.glob('*')))} files generated")

    print("\nBuilding Electron application for Windows...")
    if not run_command("npm run electron:build:win", str(frontend_dir)):
        print("✗ Failed to build Electron application")
        return False

    electron_dist = frontend_dir / "dist-electron"
    if electron_dist.exists():
        print(f"✓ Electron build completed")
        print(f"Output directory: {electron_dist}")

        for item in electron_dist.rglob("*"):
            if item.is_file():
                print(f"  - {item.relative_to(electron_dist)}")
    else:
        print("✗ Electron build failed - dist-electron directory not found")
        return False

    print("\n" + "=" * 60)
    print("Frontend build completed successfully!")
    print("=" * 60)
    print(f"Electron app package: {electron_dist}")
    print("\nNext steps:")
    print("1. Copy the generated installer to Windows")
    print("2. Test the installer on Windows system")
    print("3. Verify frontend connects to backend")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

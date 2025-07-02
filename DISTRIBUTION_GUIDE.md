# CardioAI Pro - Distribution Guide

## 📦 Distribution Options

### Option 1: Unified Installer (Recommended)

**File**: `CardioAI-Pro-1.0.0-installer.exe` (229 MB)  
**Location**: `frontend/dist-electron/CardioAI-Pro-1.0.0-installer.exe`

#### For End Users
1. **Download**: Get the unified installer file
2. **Install**: Run installer with administrator privileges
3. **Launch**: Use desktop shortcut or start menu
4. **Access**: Application opens automatically at http://localhost:8000

#### Technical Details
- **Platform**: Windows 10/11 (64-bit)
- **Dependencies**: All bundled (Electron, Node.js, Python backend)
- **Size**: 229 MB
- **Type**: NSIS installer with desktop shortcuts
- **Backend**: Automatically starts on localhost:8000
- **Frontend**: Electron-based desktop application

### Option 2: Build from Source

#### Prerequisites
- Windows 10/11 (64-bit)
- Internet connection
- Administrator privileges (for automatic dependency installation)

#### Build Process
1. **Clone Repository**:
   ```bash
   git clone https://github.com/drguilhermecapel/cardio.ai.pro.git
   cd cardio.ai.pro
   ```

2. **Run Build Scripts**:
   ```bash
   cd windows_installer
   python build_backend.py
   python build_frontend.py
   makensis cardioai_installer.nsi
   ```

3. **Generated Files**:
   - `frontend/dist-electron/CardioAI-Pro-1.0.0-installer.exe` - Unified NSIS installer
   - Backend and frontend components included in single installer

## 🚀 User Instructions

### Quick Start (Installer)
```
1. Download CardioAI-Pro-1.0.0-installer.exe
2. Run as administrator
3. Follow installation wizard
4. Launch from desktop shortcut
5. Login with generated credentials (see console)
```

### System Requirements
- **OS**: Windows 10 or 11 (64-bit)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 500MB free space
- **Network**: Internet connection for AI analysis

### Features Available
- ✅ ECG file analysis (.csv, .txt, .xml, .dat)
- ✅ ECG image analysis (.png, .jpg, .jpeg)
- ✅ AI-powered diagnostic insights
- ✅ Medical report generation
- ✅ Patient management
- ✅ Secure data handling (LGPD/HIPAA compliant)

## 🔧 Technical Information

### Architecture
```
CardioAI-Pro-1.0.0-installer.exe
├── NSIS Installer Wrapper
├── Electron Frontend (React/TypeScript)
├── Python Backend (FastAPI)
├── AI Models (ECG Analysis)
├── Database (SQLite)
└── Dependencies (Node.js, Python runtime)
```

### Network Configuration
- **Backend Server**: http://localhost:8000
- **Frontend Interface**: Electron window + web interface
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Security Features
- Local processing (no data sent to external servers)
- Encrypted database storage
- Secure authentication
- LGPD/HIPAA compliance
- Audit logging

## 📋 Distribution Checklist

### For Developers
- [ ] Build portable executable using `npm run electron:build:win`
- [ ] Verify executable size (~209MB)
- [ ] Test double-click functionality
- [ ] Confirm all features work
- [ ] Update version numbers if needed

### For Distributors
- [ ] Test executable on clean Windows system
- [ ] Verify antivirus compatibility
- [ ] Prepare user documentation
- [ ] Set up download hosting
- [ ] Create installation video/guide

### For End Users
- [ ] Download from trusted source
- [ ] Verify file integrity (optional)
- [ ] Run executable as administrator if needed
- [ ] Allow through Windows Defender if prompted
- [ ] Access application via provided URL

## 🛠️ Troubleshooting

### Common Issues

**Issue**: "Windows protected your PC" message  
**Solution**: Click "More info" → "Run anyway"

**Issue**: Application doesn't start  
**Solution**: Run as administrator, check antivirus settings

**Issue**: Backend server fails to start  
**Solution**: Check if port 8000 is available, restart application

**Issue**: Cannot access web interface  
**Solution**: Wait 30 seconds for startup, try http://localhost:8000

### Support Resources
- **Documentation**: [GitHub Repository](https://github.com/drguilhermecapel/cardio.ai.pro)
- **Issues**: [GitHub Issues](https://github.com/drguilhermecapel/cardio.ai.pro/issues)
- **Email**: suporte@cardioai.pro

## 📊 Version Information

**Current Version**: 1.0.0  
**Build Date**: June 2025  
**Electron Version**: 28.3.3  
**Target Platform**: Windows x64  
**Code Signing**: Disabled (for easier distribution)

---

*This guide covers distribution of CardioAI Pro portable executable for Windows systems.*

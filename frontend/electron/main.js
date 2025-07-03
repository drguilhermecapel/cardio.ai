const { app, BrowserWindow, ipcMain, shell, dialog } = require('electron')
const path = require('path')
const { spawn } = require('child_process')

let mainWindow
let backendProcess

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1200,
    minHeight: 800,
    backgroundColor: '#0f172a',
    title: 'CardioAI Pro - Medical AI Analysis System',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      webSecurity: false,
    },
    icon: path.join(__dirname, '../public/favicon.ico'),
    show: false,
    frame: true,
    titleBarStyle: 'default',
    vibrancy: 'dark',
    visualEffectState: 'active',
  })

  const isDev = process.env.NODE_ENV === 'development'

  if (isDev) {
    mainWindow.loadURL('http://localhost:5173')
    mainWindow.webContents.openDevTools()
  } else {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'))
  }

  mainWindow.once('ready-to-show', () => {
    mainWindow.show()

    if (process.env.NODE_ENV === 'development') {
      mainWindow.webContents.openDevTools()
    }

    setTimeout(() => {
      checkBackendHealth()
    }, 3000)
  })

  mainWindow.on('closed', () => {
    mainWindow = null
  })

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url)
    return { action: 'deny' }
  })
}

function startBackend() {
  let backendPath

  const possiblePaths = []

  if (process.platform === 'win32') {
    possiblePaths.push(
      path.join(__dirname, '../../backend/dist/cardioai-pro-backend.exe'),
      path.join(__dirname, '../../backend/dist/cardioai-pro-backend'),
      path.join(process.resourcesPath || '', 'backend', 'cardioai-pro-backend.exe'),
      path.join(process.resourcesPath || '', 'cardioai-pro-backend.exe'),
      path.join(__dirname, 'cardioai-pro-backend.exe')
    )
  } else {
    possiblePaths.push(
      path.join(__dirname, '../../backend/dist/cardioai-pro-backend'),
      path.join(__dirname, '../../backend/dist/cardioai-pro-backend.exe'),
      path.join(process.resourcesPath || '', 'backend', 'cardioai-pro-backend'),
      path.join(process.resourcesPath || '', 'cardioai-pro-backend'),
      path.join(__dirname, 'cardioai-pro-backend')
    )
  }

  backendPath = possiblePaths.find(p => {
    try {
      return require('fs').existsSync(p)
    } catch (error) {
      console.warn('Error checking path:', p, error)
      return false
    }
  })

  if (!backendPath) {
    console.error('Backend executable not found in any of these locations:')
    possiblePaths.forEach(p => console.error('  -', p))
    console.warn('Application will run in frontend-only mode')

    dialog
      .showMessageBox({
        type: 'warning',
        title: 'CardioAI Pro - Backend Warning',
        message: 'Backend server not found',
        detail: 'The application will run in frontend-only mode. Some features may be limited.',
        buttons: ['Continue', 'Exit'],
        defaultId: 0,
        cancelId: 1,
      })
      .then(result => {
        if (result.response === 1) {
          app.quit()
        }
      })
    return
  }

  console.log('Starting backend server from:', backendPath)
  backendProcess = spawn(backendPath, [], {
    stdio: 'pipe',
    detached: false,
    windowsHide: true,
    env: {
      ...process.env,
      STANDALONE_MODE: 'true',
      ENVIRONMENT: 'production',
    },
  })

  backendProcess.stdout.on('data', data => {
    console.log(`Backend: ${data}`)
  })

  backendProcess.stderr.on('data', data => {
    console.error(`Backend Error: ${data}`)
  })

  backendProcess.on('error', err => {
    console.error('Failed to start backend process:', err)
    dialog.showErrorBox(
      'CardioAI Pro - Backend Error',
      `Failed to start backend server: ${err.message}\n\nPlease ensure the application is properly installed.`
    )
  })

  backendProcess.on('close', code => {
    console.log(`Backend process exited with code ${code}`)
    if (code !== 0 && code !== null) {
      dialog.showErrorBox(
        'CardioAI Pro - Backend Error',
        `Backend server stopped unexpectedly with code ${code}. The application may not function properly.`
      )
    }
  })
}

function stopBackend() {
  if (backendProcess) {
    console.log('Stopping backend server...')
    backendProcess.kill('SIGTERM')
    backendProcess = null
  }
}

app.whenReady().then(() => {
  startBackend()

  setTimeout(() => {
    createWindow()
  }, 2000)

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow()
    }
  })
})

app.on('window-all-closed', () => {
  stopBackend()
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('before-quit', () => {
  stopBackend()
})

async function checkBackendHealth() {
  try {
    const response = await fetch('http://localhost:8000/health')
    if (!response.ok) {
      throw new Error(`Backend health check failed: ${response.status}`)
    }
    console.log('Backend health check passed')
    return true
  } catch (error) {
    console.error('Backend health check failed:', error)
    dialog
      .showMessageBox(mainWindow, {
        type: 'warning',
        title: 'CardioAI Pro - Connection Warning',
        message: 'Backend server connection failed',
        detail: 'Unable to connect to the backend server. Some features may not work properly.',
        buttons: ['Retry', 'Continue'],
        defaultId: 0,
      })
      .then(result => {
        if (result.response === 0) {
          setTimeout(() => checkBackendHealth(), 5000)
        }
      })
    return false
  }
}

ipcMain.handle('get-backend-url', () => {
  return 'http://localhost:8000'
})

ipcMain.handle('check-backend-health', async () => {
  return await checkBackendHealth()
})

ipcMain.handle('show-about', () => {
  dialog.showMessageBox(mainWindow, {
    type: 'info',
    title: 'About CardioAI Pro',
    message: 'CardioAI Pro v1.0.0',
    detail:
      'AI-powered Electronic Medical Record System\nSpecialized in cardiovascular analysis and diagnostics\n\n© 2024 CardioAI Pro. All rights reserved.',
    buttons: ['OK'],
  })
})

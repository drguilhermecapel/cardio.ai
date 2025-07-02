// Backup System for CardioAI Pro
// Sistema de backup automático com versionamento e restauração

import { format } from 'date-fns'

// Types
interface BackupConfig {
  enabled: boolean
  schedule: string // cron expression
  retention: number // days
  compression: boolean
  encryption: boolean
  destinations: BackupDestination[]
}

interface BackupDestination {
  type: 'local' | 's3' | 'azure' | 'gcp'
  config: Record<string, any>
  enabled: boolean
}

interface BackupJob {
  id: string
  type: 'full' | 'incremental' | 'differential'
  status: 'pending' | 'running' | 'completed' | 'failed'
  startTime: string
  endTime?: string
  size: number
  files: number
  progress: number
  error?: string
}

interface BackupFile {
  id: string
  filename: string
  type: 'full' | 'incremental' | 'differential'
  timestamp: string
  size: number
  checksum: string
  encrypted: boolean
  compressed: boolean
  destination: string
}

export class BackupSystem {
  private config: BackupConfig
  private jobs: Map<string, BackupJob> = new Map()
  private worker?: Worker

  constructor(config: BackupConfig) {
    this.config = config
    this.initializeWorker()
  }

  private initializeWorker(): void {
    // Initialize Web Worker for background backup operations
    const workerCode = `
      self.onmessage = function(e) {
        const { type, data } = e.data
        
        switch (type) {
          case 'START_BACKUP':
            performBackup(data)
            break
          case 'COMPRESS_DATA':
            compressData(data)
            break
          case 'ENCRYPT_DATA':
            encryptData(data)
            break
        }
      }
      
      function performBackup(config) {
        // Backup implementation
        self.postMessage({ type: 'BACKUP_PROGRESS', progress: 0 })
        
        // Simulate backup process
        let progress = 0
        const interval = setInterval(() => {
          progress += 10
          self.postMessage({ type: 'BACKUP_PROGRESS', progress })
          
          if (progress >= 100) {
            clearInterval(interval)
            self.postMessage({ type: 'BACKUP_COMPLETE' })
          }
        }, 1000)
      }
      
      function compressData(data) {
        // Compression implementation using CompressionStream
        const stream = new CompressionStream('gzip')
        // Implementation details...
      }
      
      function encryptData(data) {
        // Encryption implementation using Web Crypto API
        // Implementation details...
      }
    `

    const blob = new Blob([workerCode], { type: 'application/javascript' })
    this.worker = new Worker(URL.createObjectURL(blob))

    this.worker.onmessage = (e) => {
      const { type, data } = e.data
      this.handleWorkerMessage(type, data)
    }
  }

  private handleWorkerMessage(type: string, data: any): void {
    switch (type) {
      case 'BACKUP_PROGRESS':
        this.updateJobProgress(data.jobId, data.progress)
        break
      case 'BACKUP_COMPLETE':
        this.completeJob(data.jobId)
        break
      case 'BACKUP_ERROR':
        this.failJob(data.jobId, data.error)
        break
    }
  }

  async createBackup(type: 'full' | 'incremental' | 'differential' = 'full'): Promise<string> {
    const jobId = this.generateJobId()
    
    const job: BackupJob = {
      id: jobId,
      type,
      status: 'pending',
      startTime: new Date().toISOString(),
      size: 0,
      files: 0,
      progress: 0
    }

    this.jobs.set(jobId, job)

    try {
      // Start backup process
      await this.startBackupProcess(job)
      return jobId
    } catch (error) {
      job.status = 'failed'
      job.error = error instanceof Error ? error.message : 'Unknown error'
      throw error
    }
  }

  private async startBackupProcess(job: BackupJob): Promise<void> {
    job.status = 'running'
    
    // Get data to backup
    const data = await this.collectBackupData(job.type)
    
    // Compress if enabled
    let processedData = data
    if (this.config.compression) {
      processedData = await this.compressData(processedData)
    }

    // Encrypt if enabled
    if (this.config.encryption) {
      processedData = await this.encryptData(processedData)
    }

    // Upload to destinations
    for (const destination of this.config.destinations) {
      if (destination.enabled) {
        await this.uploadToDestination(processedData, destination, job)
      }
    }

    job.status = 'completed'
    job.endTime = new Date().toISOString()
  }

  private async collectBackupData(type: string): Promise<any> {
    const data: any = {}

    try {
      // Backup user data
      data.users = await this.backupUsers()
      
      // Backup ECG data
      data.ecg = await this.backupECGData(type)
      
      // Backup patient data
      data.patients = await this.backupPatients()
      
      // Backup system configuration
      data.config = await this.backupConfiguration()
      
      // Backup audit logs
      data.auditLogs = await this.backupAuditLogs()

      return data
    } catch (error) {
      console.error('Error collecting backup data:', error)
      throw error
    }
  }

  private async backupUsers(): Promise<any> {
    const response = await fetch('/api/admin/backup/users', {
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('cardioai_token')}`
      }
    })

    if (!response.ok) {
      throw new Error('Failed to backup users')
    }

    return response.json()
  }

  private async backupECGData(type: string): Promise<any> {
    const since = type === 'full' ? undefined : this.getLastBackupTime()
    
    const params = new URLSearchParams()
    if (since) params.append('since', since)

    const response = await fetch(`/api/admin/backup/ecg?${params.toString()}`, {
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('cardioai_token')}`
      }
    })

    if (!response.ok) {
      throw new Error('Failed to backup ECG data')
    }

    return response.json()
  }

  private async backupPatients(): Promise<any> {
    const response = await fetch('/api/admin/backup/patients', {
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('cardioai_token')}`
      }
    })

    if (!response.ok) {
      throw new Error('Failed to backup patients')
    }

    return response.json()
  }

  private async backupConfiguration(): Promise<any> {
    const response = await fetch('/api/admin/backup/config', {
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('cardioai_token')}`
      }
    })

    if (!response.ok) {
      throw new Error('Failed to backup configuration')
    }

    return response.json()
  }

  private async backupAuditLogs(): Promise<any> {
    const response = await fetch('/api/admin/backup/audit-logs', {
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('cardioai_token')}`
      }
    })

    if (!response.ok) {
      throw new Error('Failed to backup audit logs')
    }

    return response.json()
  }

  private async compressData(data: any): Promise<ArrayBuffer> {
    const jsonString = JSON.stringify(data)
    const encoder = new TextEncoder()
    const uint8Array = encoder.encode(jsonString)

    const compressionStream = new CompressionStream('gzip')
    const writer = compressionStream.writable.getWriter()
    const reader = compressionStream.readable.getReader()

    writer.write(uint8Array)
    writer.close()

    const chunks: Uint8Array[] = []
    let done = false

    while (!done) {
      const { value, done: readerDone } = await reader.read()
      done = readerDone
      if (value) {
        chunks.push(value)
      }
    }

    const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0)
    const result = new Uint8Array(totalLength)
    let offset = 0

    for (const chunk of chunks) {
      result.set(chunk, offset)
      offset += chunk.length
    }

    return result.buffer
  }

  private async encryptData(data: ArrayBuffer): Promise<ArrayBuffer> {
    const key = await this.getEncryptionKey()
    const iv = crypto.getRandomValues(new Uint8Array(12))

    const encrypted = await crypto.subtle.encrypt(
      { name: 'AES-GCM', iv },
      key,
      data
    )

    // Prepend IV to encrypted data
    const result = new Uint8Array(iv.length + encrypted.byteLength)
    result.set(iv)
    result.set(new Uint8Array(encrypted), iv.length)

    return result.buffer
  }

  private async getEncryptionKey(): Promise<CryptoKey> {
    const keyMaterial = await crypto.subtle.importKey(
      'raw',
      new TextEncoder().encode(process.env.VITE_BACKUP_ENCRYPTION_KEY || 'default-key'),
      { name: 'PBKDF2' },
      false,
      ['deriveKey']
    )

    return crypto.subtle.deriveKey(
      {
        name: 'PBKDF2',
        salt: new TextEncoder().encode('cardioai-backup-salt'),
        iterations: 100000,
        hash: 'SHA-256'
      },
      keyMaterial,
      { name: 'AES-GCM', length: 256 },
      false,
      ['encrypt', 'decrypt']
    )
  }

  private async uploadToDestination(
    data: ArrayBuffer, 
    destination: BackupDestination, 
    job: BackupJob
  ): Promise<void> {
    const filename = this.generateBackupFilename(job)

    switch (destination.type) {
      case 'local':
        await this.uploadToLocal(data, filename, destination.config)
        break
      case 's3':
        await this.uploadToS3(data, filename, destination.config)
        break
      case 'azure':
        await this.uploadToAzure(data, filename, destination.config)
        break
      case 'gcp':
        await this.uploadToGCP(data, filename, destination.config)
        break
    }
  }

  private async uploadToLocal(data: ArrayBuffer, filename: string, config: any): Promise<void> {
    // For browser environment, we'll use IndexedDB as "local" storage
    const db = await this.openBackupDB()
    const transaction = db.transaction(['backups'], 'readwrite')
    const store = transaction.objectStore('backups')

    await store.put({
      filename,
      data,
      timestamp: new Date().toISOString(),
      size: data.byteLength
    })
  }

  private async uploadToS3(data: ArrayBuffer, filename: string, config: any): Promise<void> {
    const formData = new FormData()
    formData.append('file', new Blob([data]), filename)
    formData.append('bucket', config.bucket)
    formData.append('key', config.accessKey)

    const response = await fetch('/api/backup/upload/s3', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('cardioai_token')}`
      },
      body: formData
    })

    if (!response.ok) {
      throw new Error('Failed to upload to S3')
    }
  }

  private async uploadToAzure(data: ArrayBuffer, filename: string, config: any): Promise<void> {
    // Azure Blob Storage upload implementation
    const response = await fetch('/api/backup/upload/azure', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('cardioai_token')}`,
        'Content-Type': 'application/octet-stream'
      },
      body: data
    })

    if (!response.ok) {
      throw new Error('Failed to upload to Azure')
    }
  }

  private async uploadToGCP(data: ArrayBuffer, filename: string, config: any): Promise<void> {
    // Google Cloud Storage upload implementation
    const response = await fetch('/api/backup/upload/gcp', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('cardioai_token')}`,
        'Content-Type': 'application/octet-stream'
      },
      body: data
    })

    if (!response.ok) {
      throw new Error('Failed to upload to GCP')
    }
  }

  private async openBackupDB(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('CardioAIBackups', 1)

      request.onerror = () => reject(request.error)
      request.onsuccess = () => resolve(request.result)

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result
        if (!db.objectStoreNames.contains('backups')) {
          db.createObjectStore('backups', { keyPath: 'filename' })
        }
      }
    })
  }

  async listBackups(): Promise<BackupFile[]> {
    try {
      const response = await fetch('/api/admin/backups', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('cardioai_token')}`
        }
      })

      if (!response.ok) {
        throw new Error('Failed to list backups')
      }

      return response.json()
    } catch (error) {
      console.error('Error listing backups:', error)
      return []
    }
  }

  async restoreBackup(backupId: string): Promise<boolean> {
    try {
      const response = await fetch(`/api/admin/backups/${backupId}/restore`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('cardioai_token')}`
        }
      })

      return response.ok
    } catch (error) {
      console.error('Error restoring backup:', error)
      return false
    }
  }

  async deleteBackup(backupId: string): Promise<boolean> {
    try {
      const response = await fetch(`/api/admin/backups/${backupId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('cardioai_token')}`
        }
      })

      return response.ok
    } catch (error) {
      console.error('Error deleting backup:', error)
      return false
    }
  }

  async scheduleBackup(schedule: string): Promise<void> {
    // Parse cron expression and schedule backup
    const cronParts = schedule.split(' ')
    if (cronParts.length !== 6) {
      throw new Error('Invalid cron expression')
    }

    // Store schedule in localStorage for persistence
    localStorage.setItem('cardioai_backup_schedule', schedule)

    // Set up next backup
    this.scheduleNextBackup(schedule)
  }

  private scheduleNextBackup(schedule: string): void {
    // Calculate next execution time based on cron expression
    const nextExecution = this.calculateNextExecution(schedule)
    const delay = nextExecution.getTime() - Date.now()

    if (delay > 0) {
      setTimeout(() => {
        this.createBackup('incremental')
        this.scheduleNextBackup(schedule) // Schedule next one
      }, delay)
    }
  }

  private calculateNextExecution(schedule: string): Date {
    // Simple cron parser - in production, use a proper cron library
    const now = new Date()
    const next = new Date(now.getTime() + 24 * 60 * 60 * 1000) // Next day for simplicity
    return next
  }

  private generateJobId(): string {
    return `backup_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }

  private generateBackupFilename(job: BackupJob): string {
    const timestamp = format(new Date(), 'yyyy-MM-dd_HH-mm-ss')
    return `cardioai_${job.type}_${timestamp}.backup`
  }

  private getLastBackupTime(): string {
    return localStorage.getItem('cardioai_last_backup') || new Date(0).toISOString()
  }

  private updateJobProgress(jobId: string, progress: number): void {
    const job = this.jobs.get(jobId)
    if (job) {
      job.progress = progress
    }
  }

  private completeJob(jobId: string): void {
    const job = this.jobs.get(jobId)
    if (job) {
      job.status = 'completed'
      job.endTime = new Date().toISOString()
      job.progress = 100
      
      // Update last backup time
      localStorage.setItem('cardioai_last_backup', job.endTime)
    }
  }

  private failJob(jobId: string, error: string): void {
    const job = this.jobs.get(jobId)
    if (job) {
      job.status = 'failed'
      job.error = error
      job.endTime = new Date().toISOString()
    }
  }

  getJob(jobId: string): BackupJob | undefined {
    return this.jobs.get(jobId)
  }

  getAllJobs(): BackupJob[] {
    return Array.from(this.jobs.values())
  }

  async validateBackup(backupId: string): Promise<boolean> {
    try {
      const response = await fetch(`/api/admin/backups/${backupId}/validate`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('cardioai_token')}`
        }
      })

      const result = await response.json()
      return result.valid
    } catch (error) {
      console.error('Error validating backup:', error)
      return false
    }
  }
}

// Default backup configuration
export const defaultBackupConfig: BackupConfig = {
  enabled: true,
  schedule: '0 0 2 * * *', // Daily at 2 AM
  retention: 30, // 30 days
  compression: true,
  encryption: true,
  destinations: [
    {
      type: 'local',
      config: {},
      enabled: true
    }
  ]
}

// Singleton instance
export const backupSystem = new BackupSystem(defaultBackupConfig)

export default BackupSystem


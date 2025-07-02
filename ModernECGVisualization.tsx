// Modern ECG Visualization Component
// CardioAI Pro - Fusão entre cardiologia e inteligência artificial

import React, { useRef, useEffect, useState } from 'react'
import { Typography, Button, Badge, Card, CardContent, AIGlow } from '../ui/BasicComponents'

interface ECGData {
  id: string
  patientName: string
  timestamp: string
  duration: number // in seconds
  sampleRate: number // samples per second
  leads: {
    [key: string]: number[] // ECG data points for each lead
  }
  analysis?: {
    heartRate: number
    rhythm: string
    abnormalities: string[]
    aiConfidence: number
  }
}

interface ECGVisualizationProps {
  data: ECGData
  selectedLead?: string
  isRealTime?: boolean
  showGrid?: boolean
  showAnalysis?: boolean
  className?: string
}

export const ModernECGVisualization: React.FC<ECGVisualizationProps> = ({
  data,
  selectedLead = 'I',
  isRealTime = false,
  showGrid = true,
  showAnalysis = true,
  className = ''
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()
  const [currentTime, setCurrentTime] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [zoom, setZoom] = useState(1)

  // ECG Grid and styling constants
  const GRID_SIZE = 20
  const ECG_COLOR = '#e74c3c'
  const GRID_COLOR = '#ffeaa7'
  const BACKGROUND_COLOR = '#2d3436'

  // Generate sample ECG data if not provided
  const generateSampleECG = (lead: string, duration: number, sampleRate: number): number[] => {
    const samples = duration * sampleRate
    const ecgData: number[] = []
    
    for (let i = 0; i < samples; i++) {
      const t = i / sampleRate
      // Simulate ECG waveform with P, QRS, T waves
      let value = 0
      
      // Heart rate simulation (60-100 bpm)
      const heartRate = 75
      const beatInterval = 60 / heartRate
      const beatPhase = (t % beatInterval) / beatInterval
      
      if (beatPhase < 0.1) {
        // P wave
        value = 0.3 * Math.sin(beatPhase * 20 * Math.PI)
      } else if (beatPhase < 0.2) {
        // PR segment
        value = 0
      } else if (beatPhase < 0.35) {
        // QRS complex
        const qrsPhase = (beatPhase - 0.2) / 0.15
        if (qrsPhase < 0.3) {
          value = -0.2 * Math.sin(qrsPhase * 10 * Math.PI)
        } else if (qrsPhase < 0.7) {
          value = 1.5 * Math.sin((qrsPhase - 0.3) * 7.5 * Math.PI)
        } else {
          value = -0.4 * Math.sin((qrsPhase - 0.7) * 10 * Math.PI)
        }
      } else if (beatPhase < 0.6) {
        // ST segment and T wave
        const tPhase = (beatPhase - 0.35) / 0.25
        value = 0.4 * Math.sin(tPhase * Math.PI)
      }
      
      // Add some noise for realism
      value += (Math.random() - 0.5) * 0.05
      
      ecgData.push(value)
    }
    
    return ecgData
  }

  // Draw ECG grid
  const drawGrid = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    ctx.strokeStyle = GRID_COLOR
    ctx.lineWidth = 0.5
    ctx.globalAlpha = 0.3
    
    // Vertical lines
    for (let x = 0; x <= width; x += GRID_SIZE) {
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, height)
      ctx.stroke()
    }
    
    // Horizontal lines
    for (let y = 0; y <= height; y += GRID_SIZE) {
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
      ctx.stroke()
    }
    
    ctx.globalAlpha = 1
  }

  // Draw ECG waveform
  const drawECG = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    const leadData = data.leads[selectedLead] || generateSampleECG(selectedLead, data.duration, data.sampleRate)
    
    if (!leadData || leadData.length === 0) return
    
    ctx.strokeStyle = ECG_COLOR
    ctx.lineWidth = 2
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    
    // Add glow effect
    ctx.shadowColor = ECG_COLOR
    ctx.shadowBlur = 3
    
    const centerY = height / 2
    const amplitude = height * 0.3
    const samplesPerPixel = leadData.length / width
    
    ctx.beginPath()
    
    for (let x = 0; x < width; x++) {
      const sampleIndex = Math.floor(x * samplesPerPixel * zoom)
      if (sampleIndex >= leadData.length) break
      
      const value = leadData[sampleIndex]
      const y = centerY - (value * amplitude)
      
      if (x === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    }
    
    ctx.stroke()
    ctx.shadowBlur = 0
    
    // Draw real-time cursor if playing
    if (isRealTime && isPlaying) {
      const cursorX = (currentTime / data.duration) * width
      ctx.strokeStyle = '#00ff88'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(cursorX, 0)
      ctx.lineTo(cursorX, height)
      ctx.stroke()
    }
  }

  // Animation loop
  const animate = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    const { width, height } = canvas
    
    // Clear canvas
    ctx.fillStyle = BACKGROUND_COLOR
    ctx.fillRect(0, 0, width, height)
    
    // Draw grid
    if (showGrid) {
      drawGrid(ctx, width, height)
    }
    
    // Draw ECG
    drawECG(ctx, width, height)
    
    // Update time for real-time mode
    if (isRealTime && isPlaying) {
      setCurrentTime(prev => {
        const next = prev + 0.016 // ~60fps
        return next >= data.duration ? 0 : next
      })
    }
    
    animationRef.current = requestAnimationFrame(animate)
  }

  // Setup canvas and start animation
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const resizeCanvas = () => {
      const rect = canvas.getBoundingClientRect()
      canvas.width = rect.width * window.devicePixelRatio
      canvas.height = rect.height * window.devicePixelRatio
      
      const ctx = canvas.getContext('2d')
      if (ctx) {
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio)
      }
    }
    
    resizeCanvas()
    window.addEventListener('resize', resizeCanvas)
    
    animate()
    
    return () => {
      window.removeEventListener('resize', resizeCanvas)
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [selectedLead, zoom, isPlaying, currentTime])

  const leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

  return (
    <Card variant="medical" className={`ecg-visualization ${className}`}>
      <CardContent className="p-6">
        
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-r from-red-500 to-pink-500 rounded-lg flex items-center justify-center shadow-lg">
              <span className="text-white text-lg">💓</span>
            </div>
            <div>
              <Typography variant="h6" className="font-bold text-gray-900">
                ECG - {data.patientName}
              </Typography>
              <Typography variant="body2" className="text-gray-600">
                {data.timestamp} • Lead {selectedLead}
              </Typography>
            </div>
          </div>
          
          {/* Controls */}
          <div className="flex items-center space-x-2">
            <Button
              variant="outlined"
              color="primary"
              size="small"
              onClick={() => setIsPlaying(!isPlaying)}
            >
              {isPlaying ? '⏸️' : '▶️'}
            </Button>
            <Button
              variant="outlined"
              color="secondary"
              size="small"
              onClick={() => setZoom(zoom === 1 ? 2 : 1)}
            >
              🔍 {zoom}x
            </Button>
          </div>
        </div>

        {/* Lead Selection */}
        <div className="flex flex-wrap gap-2 mb-4">
          {leads.map(lead => (
            <Button
              key={lead}
              variant={selectedLead === lead ? "contained" : "outlined"}
              color="primary"
              size="small"
              onClick={() => setCurrentTime(0)} // Reset time when changing lead
              className="min-w-[3rem]"
            >
              {lead}
            </Button>
          ))}
        </div>

        {/* ECG Canvas */}
        <div className="relative bg-gray-900 rounded-lg overflow-hidden mb-4" style={{ height: '300px' }}>
          <canvas
            ref={canvasRef}
            className="w-full h-full"
            style={{ background: BACKGROUND_COLOR }}
          />
          
          {/* Overlay Info */}
          <div className="absolute top-4 right-4 bg-black/50 backdrop-blur-sm rounded-lg p-3 text-white">
            <div className="text-sm space-y-1">
              <div>25mm/s • 10mm/mV</div>
              <div>Duration: {data.duration}s</div>
              {isRealTime && (
                <div>Time: {currentTime.toFixed(1)}s</div>
              )}
            </div>
          </div>
        </div>

        {/* Analysis Results */}
        {showAnalysis && data.analysis && (
          <AIGlow active={true}>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              
              {/* Heart Rate */}
              <div className="bg-gradient-to-br from-red-50 to-pink-50 rounded-lg p-4 border border-red-200">
                <div className="flex items-center space-x-2 mb-2">
                  <span className="text-red-500 text-lg">💓</span>
                  <Typography variant="body2" className="font-medium text-red-600">
                    Frequência Cardíaca
                  </Typography>
                </div>
                <Typography variant="h5" className="font-bold text-red-700">
                  {data.analysis.heartRate} bpm
                </Typography>
              </div>

              {/* Rhythm */}
              <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg p-4 border border-blue-200">
                <div className="flex items-center space-x-2 mb-2">
                  <span className="text-blue-500 text-lg">🎵</span>
                  <Typography variant="body2" className="font-medium text-blue-600">
                    Ritmo
                  </Typography>
                </div>
                <Typography variant="h6" className="font-bold text-blue-700">
                  {data.analysis.rhythm}
                </Typography>
              </div>

              {/* AI Confidence */}
              <div className="bg-gradient-to-br from-purple-50 to-blue-50 rounded-lg p-4 border border-purple-200">
                <div className="flex items-center space-x-2 mb-2">
                  <span className="text-purple-500 text-lg">🧠</span>
                  <Typography variant="body2" className="font-medium text-purple-600">
                    Confiança IA
                  </Typography>
                </div>
                <Typography variant="h6" className="font-bold text-purple-700">
                  {(data.analysis.aiConfidence * 100).toFixed(1)}%
                </Typography>
              </div>
            </div>

            {/* Abnormalities */}
            {data.analysis.abnormalities.length > 0 && (
              <div className="mt-4 p-4 bg-yellow-50 rounded-lg border border-yellow-200">
                <Typography variant="body2" className="font-medium text-yellow-700 mb-2">
                  ⚠️ Anormalidades Detectadas:
                </Typography>
                <div className="space-y-1">
                  {data.analysis.abnormalities.map((abnormality, index) => (
                    <Badge key={index} variant="warning" className="mr-2">
                      {abnormality}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </AIGlow>
        )}
      </CardContent>
    </Card>
  )
}

export default ModernECGVisualization


import React, { useCallback, useEffect, useRef, useState } from 'react'
import { Activity, Heart, Zap, TrendingUp } from 'lucide-react'

interface ECGDataPoint {
  time: number
  amplitude: number
}

interface ECGVisualizationProps {
  data?: ECGDataPoint[]
  heartRate?: number
  rhythm?: string
  isRealTime?: boolean
  className?: string
  height?: number
}

const ECGVisualization: React.FC<ECGVisualizationProps> = ({
  data,
  heartRate = 72,
  rhythm = 'Sinusal Normal',
  isRealTime = false,
  className = '',
  height = 200,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()
  const [isActive, setIsActive] = useState(false)

  const generateSampleECGData = (): ECGDataPoint[] => {
    const points: ECGDataPoint[] = []
    const duration = 10 // seconds
    const sampleRate = 250 // Hz
    const totalPoints = duration * sampleRate

    for (let i = 0; i < totalPoints; i++) {
      const time = i / sampleRate
      let amplitude = 0

      const heartPeriod = 60 / heartRate // seconds per beat
      const beatPhase = (time % heartPeriod) / heartPeriod

      if (beatPhase < 0.1) {
        amplitude = 0.2 * Math.sin(beatPhase * 10 * Math.PI)
      } else if (beatPhase >= 0.15 && beatPhase < 0.25) {
        const qrsPhase = (beatPhase - 0.15) / 0.1
        if (qrsPhase < 0.3) {
          amplitude = -0.3 * Math.sin((qrsPhase * Math.PI) / 0.3)
        } else if (qrsPhase < 0.7) {
          amplitude = 1.0 * Math.sin(((qrsPhase - 0.3) * Math.PI) / 0.4)
        } else {
          amplitude = -0.4 * Math.sin(((qrsPhase - 0.7) * Math.PI) / 0.3)
        }
      } else if (beatPhase >= 0.4 && beatPhase < 0.6) {
        const tPhase = (beatPhase - 0.4) / 0.2
        amplitude = 0.3 * Math.sin(tPhase * Math.PI)
      }

      amplitude += (Math.random() - 0.5) * 0.05

      points.push({ time, amplitude })
    }

    return points
  }

  const ecgData = data || generateSampleECGData()

  const drawECG = useCallback(
    (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement): void => {
      const { width, height } = canvas

      ctx.fillStyle = 'rgba(17, 24, 39, 0.95)'
      ctx.fillRect(0, 0, width, height)

      ctx.strokeStyle = 'rgba(34, 197, 94, 0.1)'
      ctx.lineWidth = 1

      for (let x = 0; x < width; x += 20) {
        ctx.beginPath()
        ctx.moveTo(x, 0)
        ctx.lineTo(x, height)
        ctx.stroke()
      }

      for (let y = 0; y < height; y += 20) {
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(width, y)
        ctx.stroke()
      }

      if (ecgData.length > 0) {
        ctx.strokeStyle = '#22d3ee'
        ctx.lineWidth = 2
        ctx.shadowColor = '#22d3ee'
        ctx.shadowBlur = 10

        ctx.beginPath()

        ecgData.forEach((point, index) => {
          const x = (point.time / 10) * width
          const y = height / 2 - (point.amplitude * height) / 4

          if (index === 0) {
            ctx.moveTo(x, y)
          } else {
            ctx.lineTo(x, y)
          }
        })

        ctx.stroke()
        ctx.shadowBlur = 0
      }

      if (isRealTime && isActive) {
        const pulseX = (((Date.now() / 1000) % 10) / 10) * width
        ctx.fillStyle = '#ef4444'
        ctx.shadowColor = '#ef4444'
        ctx.shadowBlur = 20
        ctx.beginPath()
        ctx.arc(pulseX, height / 2, 4, 0, 2 * Math.PI)
        ctx.fill()
        ctx.shadowBlur = 0
      }
    },
    [ecgData, isRealTime, isActive]
  )

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = canvas.offsetWidth * window.devicePixelRatio
    canvas.height = height * window.devicePixelRatio
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio)

    const animate = (): void => {
      drawECG(ctx, canvas)
      if (isRealTime) {
        animationRef.current = requestAnimationFrame(animate)
      }
    }

    animate()

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [ecgData, isRealTime, isActive, height, drawECG])

  const getRhythmColor = (): string => {
    if (rhythm.toLowerCase().includes('normal')) return 'text-green-400'
    if (rhythm.toLowerCase().includes('arritmia')) return 'text-red-400'
    return 'text-yellow-400'
  }

  const getHeartRateColor = (): string => {
    if (heartRate < 60) return 'text-blue-400'
    if (heartRate > 100) return 'text-red-400'
    return 'text-green-400'
  }

  return (
    <div
      className={`bg-gray-900/40 backdrop-blur-xl rounded-2xl border border-gray-700/50 shadow-2xl overflow-hidden ${className}`}
    >
      {/* Header */}
      <div className="p-4 border-b border-gray-700/50">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 shadow-lg">
              <Activity className="w-5 h-5 text-white" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white">ECG Monitor</h3>
              <p className="text-sm text-gray-400">Eletrocardiograma em tempo real</p>
            </div>
          </div>

          <button
            onClick={() => setIsActive(!isActive)}
            className={`
              px-4 py-2 rounded-xl font-medium transition-all
              ${
                isActive
                  ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                  : 'bg-green-500/20 text-green-400 border border-green-500/30'
              }
            `}
          >
            {isActive ? 'Pausar' : 'Iniciar'}
          </button>
        </div>
      </div>

      {/* ECG Display */}
      <div className="relative">
        <canvas
          ref={canvasRef}
          style={{ width: '100%', height: `${height}px` }}
          className="block"
        />

        {/* Overlay indicators */}
        <div className="absolute top-4 left-4 space-y-2">
          <div className="flex items-center space-x-2 px-3 py-1 bg-gray-900/80 rounded-lg backdrop-blur-sm">
            <Heart className={`w-4 h-4 ${getHeartRateColor()}`} />
            <span className={`text-sm font-semibold ${getHeartRateColor()}`}>{heartRate} BPM</span>
          </div>

          <div className="flex items-center space-x-2 px-3 py-1 bg-gray-900/80 rounded-lg backdrop-blur-sm">
            <Zap className={`w-4 h-4 ${getRhythmColor()}`} />
            <span className={`text-sm font-medium ${getRhythmColor()}`}>{rhythm}</span>
          </div>
        </div>

        {/* Real-time indicator */}
        {isRealTime && (
          <div className="absolute top-4 right-4">
            <div className="flex items-center space-x-2 px-3 py-1 bg-gray-900/80 rounded-lg backdrop-blur-sm">
              <div
                className={`w-2 h-2 rounded-full ${isActive ? 'bg-green-400 animate-pulse' : 'bg-gray-400'}`}
              ></div>
              <span className="text-sm text-gray-300">{isActive ? 'AO VIVO' : 'PAUSADO'}</span>
            </div>
          </div>
        )}
      </div>

      {/* Footer with additional info */}
      <div className="p-4 border-t border-gray-700/50">
        <div className="flex items-center justify-between text-sm text-gray-400">
          <div className="flex items-center space-x-4">
            <span>Velocidade: 25mm/s</span>
            <span>Amplitude: 10mm/mV</span>
          </div>
          <div className="flex items-center space-x-2">
            <TrendingUp className="w-4 h-4" />
            <span>Qualidade do sinal: Boa</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ECGVisualization

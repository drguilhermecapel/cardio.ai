// Real-time Heartbeat Animation Component
// CardioAI Pro - Fusão entre cardiologia e inteligência artificial

import React, { useState, useEffect } from 'react'
import { Typography, Card, CardContent, Badge } from '../ui/BasicComponents'

interface HeartbeatData {
  bpm: number
  rhythm: 'normal' | 'irregular' | 'tachycardia' | 'bradycardia'
  status: 'stable' | 'warning' | 'critical'
  timestamp: string
}

interface HeartbeatAnimationProps {
  data: HeartbeatData
  isRealTime?: boolean
  showWaveform?: boolean
  size?: 'small' | 'medium' | 'large'
  className?: string
}

export const HeartbeatAnimation: React.FC<HeartbeatAnimationProps> = ({
  data,
  isRealTime = true,
  showWaveform = true,
  size = 'medium',
  className = ''
}) => {
  const [currentBeat, setCurrentBeat] = useState(0)
  const [isBeating, setIsBeating] = useState(false)

  // Calculate animation timing based on BPM
  const beatInterval = 60000 / data.bpm // milliseconds between beats
  
  const sizeClasses = {
    small: 'w-16 h-16',
    medium: 'w-24 h-24',
    large: 'w-32 h-32'
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'stable': return 'text-green-500'
      case 'warning': return 'text-yellow-500'
      case 'critical': return 'text-red-500'
      default: return 'text-gray-500'
    }
  }

  const getRhythmIcon = (rhythm: string) => {
    switch (rhythm) {
      case 'normal': return '💚'
      case 'irregular': return '💛'
      case 'tachycardia': return '🔴'
      case 'bradycardia': return '🔵'
      default: return '💓'
    }
  }

  // Heartbeat animation effect
  useEffect(() => {
    if (!isRealTime) return

    const interval = setInterval(() => {
      setIsBeating(true)
      setCurrentBeat(prev => prev + 1)
      
      setTimeout(() => {
        setIsBeating(false)
      }, 200) // Beat duration
      
    }, beatInterval)

    return () => clearInterval(interval)
  }, [beatInterval, isRealTime])

  return (
    <Card variant="medical" className={`heartbeat-monitor ${className}`}>
      <CardContent className="p-6">
        
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-2">
            <span className="text-2xl">💓</span>
            <Typography variant="h6" className="font-bold text-gray-900">
              Monitor Cardíaco
            </Typography>
          </div>
          <Badge variant={data.status === 'stable' ? 'success' : data.status === 'warning' ? 'warning' : 'critical'}>
            {data.status.toUpperCase()}
          </Badge>
        </div>

        {/* Main Display */}
        <div className="flex items-center justify-center mb-6">
          <div className="relative">
            
            {/* Animated Heart */}
            <div className={`
              ${sizeClasses[size]} 
              flex items-center justify-center
              transition-transform duration-200
              ${isBeating ? 'scale-125' : 'scale-100'}
              ${isBeating ? 'animate-pulse' : ''}
            `}>
              <div className={`
                text-6xl
                ${getStatusColor(data.status)}
                ${isBeating ? 'drop-shadow-lg' : ''}
                transition-all duration-200
              `}>
                💓
              </div>
            </div>

            {/* Pulse Rings */}
            {isBeating && (
              <>
                <div className="absolute inset-0 rounded-full border-2 border-red-400 animate-ping opacity-75"></div>
                <div className="absolute inset-0 rounded-full border border-red-300 animate-ping opacity-50" style={{ animationDelay: '0.1s' }}></div>
              </>
            )}
          </div>
        </div>

        {/* BPM Display */}
        <div className="text-center mb-4">
          <Typography variant="h3" className="font-bold text-gray-900 mb-1">
            {data.bpm}
          </Typography>
          <Typography variant="body2" className="text-gray-600">
            batimentos por minuto
          </Typography>
        </div>

        {/* Rhythm Info */}
        <div className="flex items-center justify-center space-x-2 mb-4">
          <span className="text-xl">{getRhythmIcon(data.rhythm)}</span>
          <Typography variant="body2" className="text-gray-700 capitalize">
            Ritmo {data.rhythm}
          </Typography>
        </div>

        {/* Mini Waveform */}
        {showWaveform && (
          <div className="relative h-16 bg-gray-900 rounded-lg overflow-hidden mb-4">
            <div className="absolute inset-0 flex items-center justify-center">
              <svg width="100%" height="100%" viewBox="0 0 400 60" className="text-green-400">
                <defs>
                  <linearGradient id="waveGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="currentColor" stopOpacity="0" />
                    <stop offset="50%" stopColor="currentColor" stopOpacity="1" />
                    <stop offset="100%" stopColor="currentColor" stopOpacity="0" />
                  </linearGradient>
                </defs>
                
                {/* ECG-like waveform */}
                <path
                  d="M0,30 L50,30 L60,10 L70,50 L80,20 L90,30 L150,30 L160,25 L170,35 L180,30 L240,30 L250,10 L260,50 L270,20 L280,30 L340,30 L350,25 L360,35 L370,30 L400,30"
                  stroke="url(#waveGradient)"
                  strokeWidth="2"
                  fill="none"
                  className={`${isBeating ? 'animate-pulse' : ''}`}
                />
                
                {/* Moving cursor */}
                {isRealTime && (
                  <line
                    x1={((currentBeat * 50) % 400)}
                    y1="0"
                    x2={((currentBeat * 50) % 400)}
                    y2="60"
                    stroke="#00ff88"
                    strokeWidth="2"
                    className="animate-pulse"
                  />
                )}
              </svg>
            </div>
          </div>
        )}

        {/* Stats Grid */}
        <div className="grid grid-cols-2 gap-4">
          <div className="text-center p-3 bg-gray-50 rounded-lg">
            <Typography variant="caption" className="text-gray-600">
              Última Atualização
            </Typography>
            <Typography variant="body2" className="font-medium text-gray-900">
              {data.timestamp}
            </Typography>
          </div>
          <div className="text-center p-3 bg-gray-50 rounded-lg">
            <Typography variant="caption" className="text-gray-600">
              Total de Batimentos
            </Typography>
            <Typography variant="body2" className="font-medium text-gray-900">
              {currentBeat.toLocaleString()}
            </Typography>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

// Heart Rate Trend Component
interface HeartRateTrendProps {
  data: Array<{
    time: string
    bpm: number
    status: 'stable' | 'warning' | 'critical'
  }>
  className?: string
}

export const HeartRateTrend: React.FC<HeartRateTrendProps> = ({
  data,
  className = ''
}) => {
  const maxBpm = Math.max(...data.map(d => d.bpm))
  const minBpm = Math.min(...data.map(d => d.bpm))
  const range = maxBpm - minBpm

  return (
    <Card variant="medical" className={className}>
      <CardContent className="p-6">
        <div className="flex items-center space-x-2 mb-4">
          <span className="text-2xl">📈</span>
          <Typography variant="h6" className="font-bold text-gray-900">
            Tendência da Frequência Cardíaca
          </Typography>
        </div>

        <div className="relative h-32 bg-gray-50 rounded-lg p-4">
          <svg width="100%" height="100%" viewBox="0 0 400 100" className="overflow-visible">
            <defs>
              <linearGradient id="trendGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.3" />
                <stop offset="100%" stopColor="#3b82f6" stopOpacity="0" />
              </linearGradient>
            </defs>
            
            {/* Trend line */}
            <path
              d={data.map((point, index) => {
                const x = (index / (data.length - 1)) * 380 + 10
                const y = 90 - ((point.bpm - minBpm) / range) * 80
                return `${index === 0 ? 'M' : 'L'} ${x} ${y}`
              }).join(' ')}
              stroke="#3b82f6"
              strokeWidth="3"
              fill="none"
              className="drop-shadow-sm"
            />
            
            {/* Fill area */}
            <path
              d={`${data.map((point, index) => {
                const x = (index / (data.length - 1)) * 380 + 10
                const y = 90 - ((point.bpm - minBpm) / range) * 80
                return `${index === 0 ? 'M' : 'L'} ${x} ${y}`
              }).join(' ')} L 390 90 L 10 90 Z`}
              fill="url(#trendGradient)"
            />
            
            {/* Data points */}
            {data.map((point, index) => {
              const x = (index / (data.length - 1)) * 380 + 10
              const y = 90 - ((point.bpm - minBpm) / range) * 80
              const color = point.status === 'stable' ? '#10b981' : 
                           point.status === 'warning' ? '#f59e0b' : '#ef4444'
              
              return (
                <circle
                  key={index}
                  cx={x}
                  cy={y}
                  r="4"
                  fill={color}
                  className="drop-shadow-sm animate-pulse"
                />
              )
            })}
          </svg>
        </div>

        {/* Legend */}
        <div className="flex justify-between items-center mt-4 text-sm text-gray-600">
          <span>{data[0]?.time}</span>
          <div className="flex space-x-4">
            <span>Min: {minBpm} bpm</span>
            <span>Max: {maxBpm} bpm</span>
          </div>
          <span>{data[data.length - 1]?.time}</span>
        </div>
      </CardContent>
    </Card>
  )
}

export default HeartbeatAnimation


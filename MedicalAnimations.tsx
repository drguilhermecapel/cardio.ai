// Medical Micro-animations Component
// CardioAI Pro - Fusão entre cardiologia e inteligência artificial

import React, { useState, useEffect } from 'react'
import { Typography } from '../ui/BasicComponents'

// Pulse Animation Component
interface PulseAnimationProps {
  children: React.ReactNode
  intensity?: 'low' | 'medium' | 'high'
  color?: string
  className?: string
}

export const PulseAnimation: React.FC<PulseAnimationProps> = ({
  children,
  intensity = 'medium',
  color = '#ef4444',
  className = ''
}) => {
  const intensityClasses = {
    low: 'animate-pulse',
    medium: 'animate-pulse',
    high: 'animate-bounce'
  }

  return (
    <div 
      className={`${intensityClasses[intensity]} ${className}`}
      style={{ 
        filter: `drop-shadow(0 0 8px ${color}40)`,
        animationDuration: intensity === 'high' ? '0.8s' : '1.5s'
      }}
    >
      {children}
    </div>
  )
}

// Breathing Animation Component
interface BreathingAnimationProps {
  children: React.ReactNode
  rate?: number // breaths per minute
  className?: string
}

export const BreathingAnimation: React.FC<BreathingAnimationProps> = ({
  children,
  rate = 16, // normal breathing rate
  className = ''
}) => {
  const [isInhaling, setIsInhaling] = useState(true)
  
  useEffect(() => {
    const breathInterval = (60 / rate) * 1000 // milliseconds per breath
    const interval = setInterval(() => {
      setIsInhaling(prev => !prev)
    }, breathInterval / 2) // half cycle for inhale/exhale
    
    return () => clearInterval(interval)
  }, [rate])

  return (
    <div 
      className={`transition-transform duration-1000 ease-in-out ${className}`}
      style={{
        transform: isInhaling ? 'scale(1.05)' : 'scale(0.95)',
        opacity: isInhaling ? 1 : 0.8
      }}
    >
      {children}
    </div>
  )
}

// ECG Pulse Line Animation
interface ECGPulseProps {
  isActive?: boolean
  color?: string
  width?: number
  height?: number
  className?: string
}

export const ECGPulse: React.FC<ECGPulseProps> = ({
  isActive = true,
  color = '#00ff88',
  width = 200,
  height = 60,
  className = ''
}) => {
  const [currentPhase, setCurrentPhase] = useState(0)

  useEffect(() => {
    if (!isActive) return

    const interval = setInterval(() => {
      setCurrentPhase(prev => (prev + 1) % 100)
    }, 50) // 20fps animation

    return () => clearInterval(interval)
  }, [isActive])

  // Generate ECG-like path
  const generateECGPath = (phase: number) => {
    const points = []
    for (let i = 0; i < width; i++) {
      const x = i
      let y = height / 2
      
      // Create ECG pattern
      const beatPosition = (i + phase * 2) % 80
      if (beatPosition < 5) {
        // P wave
        y += Math.sin((beatPosition / 5) * Math.PI) * 5
      } else if (beatPosition < 15) {
        // QRS complex
        const qrsPhase = (beatPosition - 5) / 10
        if (qrsPhase < 0.3) {
          y -= Math.sin(qrsPhase * 10 * Math.PI) * 3
        } else if (qrsPhase < 0.7) {
          y += Math.sin((qrsPhase - 0.3) * 7.5 * Math.PI) * 20
        } else {
          y -= Math.sin((qrsPhase - 0.7) * 10 * Math.PI) * 8
        }
      } else if (beatPosition < 35) {
        // T wave
        const tPhase = (beatPosition - 15) / 20
        y += Math.sin(tPhase * Math.PI) * 8
      }
      
      points.push(`${x},${y}`)
    }
    return `M ${points.join(' L ')}`
  }

  return (
    <div className={className}>
      <svg width={width} height={height} className="overflow-visible">
        <defs>
          <linearGradient id="ecgGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor={color} stopOpacity="0" />
            <stop offset="30%" stopColor={color} stopOpacity="0.8" />
            <stop offset="70%" stopColor={color} stopOpacity="1" />
            <stop offset="100%" stopColor={color} stopOpacity="0" />
          </linearGradient>
        </defs>
        
        <path
          d={generateECGPath(currentPhase)}
          stroke="url(#ecgGradient)"
          strokeWidth="2"
          fill="none"
          className={isActive ? 'animate-pulse' : ''}
        />
        
        {/* Moving cursor */}
        {isActive && (
          <line
            x1={(currentPhase * 2) % width}
            y1="0"
            x2={(currentPhase * 2) % width}
            y2={height}
            stroke={color}
            strokeWidth="1"
            opacity="0.6"
            className="animate-pulse"
          />
        )}
      </svg>
    </div>
  )
}

// Data Flow Animation
interface DataFlowProps {
  direction?: 'horizontal' | 'vertical'
  speed?: 'slow' | 'medium' | 'fast'
  color?: string
  className?: string
}

export const DataFlowAnimation: React.FC<DataFlowProps> = ({
  direction = 'horizontal',
  speed = 'medium',
  color = '#3b82f6',
  className = ''
}) => {
  const speedDurations = {
    slow: '3s',
    medium: '2s',
    fast: '1s'
  }

  const isHorizontal = direction === 'horizontal'

  return (
    <div className={`relative overflow-hidden ${className}`}>
      {/* Flowing particles */}
      {[...Array(5)].map((_, i) => (
        <div
          key={i}
          className="absolute w-2 h-2 rounded-full opacity-60"
          style={{
            backgroundColor: color,
            animation: `dataFlow${direction} ${speedDurations[speed]} linear infinite`,
            animationDelay: `${i * 0.4}s`,
            [isHorizontal ? 'top' : 'left']: `${20 + i * 15}%`,
            [isHorizontal ? 'left' : 'top']: '-10px'
          }}
        />
      ))}
      
      {/* Flowing line */}
      <div
        className="absolute opacity-30"
        style={{
          backgroundColor: color,
          [isHorizontal ? 'width' : 'height']: '100%',
          [isHorizontal ? 'height' : 'width']: '1px',
          [isHorizontal ? 'top' : 'left']: '50%',
          transform: isHorizontal ? 'translateY(-50%)' : 'translateX(-50%)'
        }}
      />
      
      <style jsx>{`
        @keyframes dataFlowhorizontal {
          from { left: -10px; }
          to { left: calc(100% + 10px); }
        }
        
        @keyframes dataFlowvertical {
          from { top: -10px; }
          to { top: calc(100% + 10px); }
        }
      `}</style>
    </div>
  )
}

// Loading Heartbeat
interface LoadingHeartbeatProps {
  size?: 'small' | 'medium' | 'large'
  message?: string
  className?: string
}

export const LoadingHeartbeat: React.FC<LoadingHeartbeatProps> = ({
  size = 'medium',
  message = 'Processando dados médicos...',
  className = ''
}) => {
  const sizeClasses = {
    small: 'text-2xl',
    medium: 'text-4xl',
    large: 'text-6xl'
  }

  return (
    <div className={`flex flex-col items-center justify-center space-y-4 ${className}`}>
      <PulseAnimation intensity="high" color="#ef4444">
        <div className={`${sizeClasses[size]} text-red-500`}>
          💓
        </div>
      </PulseAnimation>
      
      <Typography variant="body2" className="text-gray-600 text-center">
        {message}
      </Typography>
      
      <ECGPulse width={150} height={40} />
    </div>
  )
}

// AI Processing Animation
interface AIProcessingProps {
  stage?: 'analyzing' | 'learning' | 'predicting' | 'completed'
  progress?: number
  className?: string
}

export const AIProcessingAnimation: React.FC<AIProcessingProps> = ({
  stage = 'analyzing',
  progress = 0,
  className = ''
}) => {
  const [particles, setParticles] = useState<Array<{ id: number; x: number; y: number; delay: number }>>([])

  useEffect(() => {
    const newParticles = Array.from({ length: 20 }, (_, i) => ({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      delay: Math.random() * 2
    }))
    setParticles(newParticles)
  }, [])

  const stageColors = {
    analyzing: '#3b82f6',
    learning: '#8b5cf6',
    predicting: '#06b6d4',
    completed: '#10b981'
  }

  const stageIcons = {
    analyzing: '🔍',
    learning: '🧠',
    predicting: '🔮',
    completed: '✅'
  }

  return (
    <div className={`relative w-32 h-32 ${className}`}>
      {/* Central brain icon */}
      <div className="absolute inset-0 flex items-center justify-center">
        <PulseAnimation intensity="medium" color={stageColors[stage]}>
          <div className="text-4xl">
            {stageIcons[stage]}
          </div>
        </PulseAnimation>
      </div>
      
      {/* Floating particles */}
      {particles.map(particle => (
        <div
          key={particle.id}
          className="absolute w-1 h-1 rounded-full opacity-60 animate-ping"
          style={{
            backgroundColor: stageColors[stage],
            left: `${particle.x}%`,
            top: `${particle.y}%`,
            animationDelay: `${particle.delay}s`,
            animationDuration: '2s'
          }}
        />
      ))}
      
      {/* Progress ring */}
      <svg className="absolute inset-0 w-full h-full transform -rotate-90">
        <circle
          cx="50%"
          cy="50%"
          r="45%"
          fill="none"
          stroke="#e5e7eb"
          strokeWidth="2"
        />
        <circle
          cx="50%"
          cy="50%"
          r="45%"
          fill="none"
          stroke={stageColors[stage]}
          strokeWidth="3"
          strokeLinecap="round"
          strokeDasharray={`${2 * Math.PI * 45} ${2 * Math.PI * 45}`}
          strokeDashoffset={`${2 * Math.PI * 45 * (1 - progress / 100)}`}
          className="transition-all duration-500"
        />
      </svg>
    </div>
  )
}

export default {
  PulseAnimation,
  BreathingAnimation,
  ECGPulse,
  DataFlowAnimation,
  LoadingHeartbeat,
  AIProcessingAnimation
}

